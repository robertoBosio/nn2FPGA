import onnx
import numpy as np
from onnx import helper, numpy_helper
from qonnx.util.basic import get_by_name
from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.general import SortGraph
import logging

logger = logging.getLogger(__name__)

class SlicesToSplitTree(Transformation):
    """
    Replace multiple Slice ops that share the same data input with a (roughly) balanced binary tree
    of 2-way split ops (SplitC / SplitH / SplitW).

    Assumptions (matching your current Slice pattern policy):
      - We only handle families of Slice nodes that:
          * all slice the same single axis (C/H/W)
          * have constant starts/ends/axes/steps with step=1
          * fully tile the axis (no gaps, no discard): [0..D) covered contiguously
      - Slice data input is quantized (producer is Quant/IntQuant passing check_act_quant)
        NOTE: you can keep that check in the pattern and call this transform only after partitioning.
    """

    def __init__(self):
        super().__init__()
        self.axis_to_split_op = [1, 2, 3]  # C, H, W axes

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        modified = False

        # Ensure shapes exist (needed for negative index normalization and axis size)
        model = model.transform(InferShapes())

        # Build: data_tensor_name -> list[Slice nodes]
        families = {}
        for n in list(graph.node):
            if n.op_type != "Slice":
                continue
            if len(n.input) < 3:
                continue
            data_in = n.input[0]
            families.setdefault(data_in, []).append(n)

        # We'll append new nodes; later we'll remove old Slice nodes.
        slice_nodes_to_remove = set()

        for data_in, slices in families.items():
            if not slices:
                continue

            # Parse all slices into (axis, start, end, slice_node), and collect Constant param nodes
            data_shape = model.get_tensor_shape(data_in)
            if data_shape is None:
                logger.info(f"Skipping Slice family on {data_in}: unknown shape")
                continue

            parsed = []
            const_param_node_names = set()
            failed_reasons = []

            for s in slices:
                ok, axis, st, en, why, used_const_nodes = self._parse_slice_to_interval(model, s, data_shape)
                const_param_node_names.update(used_const_nodes)
                if not ok:
                    failed_reasons.append(f"{s.name}: {why}")
                    continue
                parsed.append((axis, st, en, s))

            if failed_reasons:
                logger.info(f"Skipping Slice family on {data_in}: " + "; ".join(failed_reasons))
                continue

            # Axis consistency
            axes = {a for (a, _, _, _) in parsed}
            if len(axes) != 1:
                logger.info(f"Skipping Slice family on {data_in}: mixed axes {sorted(list(axes))}")
                continue
            axis = next(iter(axes))
            if axis not in self.axis_to_split_op:
                logger.info(f"Skipping Slice family on {data_in}: axis {axis} not supported")
                continue

            dim = int(data_shape[axis])

            # Sort by start
            parsed.sort(key=lambda x: x[1])
            intervals = [(st, en, s) for (_, st, en, s) in parsed]

            # Enforce "no discard" tiling (same as your pattern)
            if not self._is_full_tiling(intervals, dim):
                logger.info(f"Skipping Slice family on {data_in}: slices do not fully tile axis")
                continue

            # Good: we can rewrite this family
            logger.info(
                f"Rewriting {len(intervals)} Slice nodes on tensor {data_in} into split tree "
            )

            # Mark slice nodes for removal
            for (_, _, s) in intervals:
                slice_nodes_to_remove.add(s.name)

            # Build leaves in order. Each leaf corresponds to exactly one slice interval.
            # We want the final produced tensors to have the SAME names as original Slice outputs,
            # so consumers remain unchanged.
            leaves = []
            for (st, en, s) in intervals:
                if len(s.output) != 1:
                    logger.info(f"Skipping Slice {s.name}: expected single output")
                    continue
                leaves.append(
                    {
                        "start": st,
                        "end": en,
                        "out_name": s.output[0],  # preserve original output name
                        "slice_node": s,
                    }
                )

            if not leaves:
                continue

            # Special case: only one slice (identity rename)
            if len(leaves) == 1:
                out_name = leaves[0]["out_name"]
                if out_name != data_in:
                    ident_name = f"Identity__{out_name}"
                    graph.node.append(
                        helper.make_node("Identity", inputs=[data_in], outputs=[out_name], name=ident_name)
                    )
                    modified = True
                continue

            # Build a balanced split tree that produces the leaf tensors.
            # Internal split outputs use generated intermediate tensor names.
            self._build_split_tree(
                model=model,
                axis=axis,
                input_tensor=data_in,
                leaves=leaves,
            )
            modified = True

        # Remove old Slice nodes (by name)
        if slice_nodes_to_remove:
            new_nodes = [n for n in model.graph.node if n.name not in slice_nodes_to_remove]
            del model.graph.node[:]
            model.graph.node.extend(new_nodes)
            modified = True

        if modified:
            # Re-run shape inference and topological sort for cleanliness
            model = model.transform(SortGraph())
            model = model.transform(InferShapes())

        return (model, False)

    # ------------------------ Tree building ------------------------
    def _build_split_tree(self, model, axis: int, input_tensor: str, leaves: list):
        """
        Build a binary split tree using *ONNX Split* (not custom ops).

        Assumes:
        - NCHW-like axis indexing (axis passed in)
        - leaves are contiguous and fully tile [0..D) along `axis`
        - we know D from inferred shapes
        - opset >= 13 (Split takes second input `split` tensor). If you use an older opset,
            you must adapt to attribute-based split.

        leaves: list of dicts with keys {start, end, out_name}
            ordered by start, and contiguous.
        """
        def get_default_opset(model: ModelWrapper) -> int:
            for opset in model.model.opset_import:
                # Default (standard ONNX) domain is empty string
                if opset.domain == "" or opset.domain is None:
                    return opset.version
            # Fallback: ONNX guarantees at least one default opset
            raise RuntimeError("No default ONNX opset import found")

        def fresh_init_name(base: str) -> str:
            used = {i.name for i in model.graph.initializer}
            name = base
            k = 0
            while name in used:
                k += 1
                name = f"{base}_{k}"
            return name

        def make_split_sizes_init(k_left: int, seg_len: int, base: str) -> str:
            """
            Create an INT64 initializer for Split sizes [k_left, seg_len-k_left].
            Returns the initializer tensor name.
            """
            k_right = seg_len - k_left
            assert k_left >= 0 and k_right >= 0, "Invalid split sizes"
            init_name = fresh_init_name(base)
            t = helper.make_tensor(
                name=init_name,
                data_type=onnx.TensorProto.INT64,
                dims=[2],
                vals=[int(k_left), int(k_right)],
            )
            model.graph.initializer.append(t)
            return init_name

        # Recursive build over leaf indices
        def rec(inp_tensor: str, lo: int, hi: int, seg_len: int, opset_version: int):
            n = hi - lo
            if n == 1:
                # Rename segment to the original Slice output tensor name
                out_name = leaves[lo]["out_name"]
                if inp_tensor != out_name:
                    model.graph.node.append(
                        helper.make_node(
                            "Identity",
                            inputs=[inp_tensor],
                            outputs=[out_name],
                            name=self._fresh_node_name(model, f"Identity__leaf_{out_name}"),
                        )
                    )
                return

            mid = (lo + hi) // 2

            # Left length = sum of leaf lengths in [lo, mid)
            left_len = 0
            for i in range(lo, mid):
                left_len += (leaves[i]["end"] - leaves[i]["start"])
            right_len = seg_len - left_len
            assert left_len > 0 and right_len > 0, "Degenerate split (check leaf tiling)"

            # Choose output tensor names. If subtree is a single leaf, output directly to leaf tensor name.
            if (mid - lo) == 1:
                left_out = leaves[lo]["out_name"]
            else:
                left_out = self._fresh_tensor_name(model)

            if (hi - mid) == 1:
                right_out = leaves[mid]["out_name"]
            else:
                right_out = self._fresh_tensor_name(model)

            # Build Split initializer and node
            if opset_version < 13:

                # Below opset 13, Split uses attribute-based split sizes.
                split_node = helper.make_node(
                    "Split",
                    inputs=[inp_tensor],
                    outputs=[left_out, right_out],
                    name=self._fresh_node_name(model, f"Split_{axis}_{lo}_{hi}"),
                    axis=axis,
                    split=[int(left_len), int(seg_len - left_len)],
                )
            else:
                # Above opset 13, Split uses second input tensor for split sizes.
                split_sizes_name = make_split_sizes_init(
                    k_left=left_len,
                    seg_len=seg_len,
                    base=f"{inp_tensor}__split_sizes_{lo}_{hi}",
                )

                split_node = helper.make_node(
                    "Split",
                    inputs=[inp_tensor, split_sizes_name],
                    outputs=[left_out, right_out],
                    name=self._fresh_node_name(model, f"Split_{axis}_{lo}_{hi}"),
                    axis=axis,
                )

            model.graph.node.append(split_node)

            # Recurse on children
            rec(left_out, lo, mid, left_len, opset_version)
            rec(right_out, mid, hi, right_len, opset_version)

        opset_version = get_default_opset(model)

        # Total segment length along axis
        in_shape = model.get_tensor_shape(input_tensor)
        if in_shape is None:
            raise RuntimeError(f"Cannot build split tree: unknown shape for {input_tensor}")
        dim = int(in_shape[axis])

        rec(input_tensor, 0, len(leaves), dim, opset_version)

    def _fresh_tensor_name(self, model: ModelWrapper) -> str:
        """Generate a tensor name not used in the graph."""
        return model.make_new_valueinfo_name()

    def _fresh_node_name(self, model: ModelWrapper, base: str) -> str:
        """Generate a node name not used in the graph."""
        used = {n.name for n in model.graph.node}
        name = base
        k = 0
        while name in used:
            k += 1
            name = f"{base}_{k}"
        return name

    def _get_const_array_and_constnode(self, model, tensor_name: str):
        """Return (np.array or None, const_node_name_or_None)."""
        if tensor_name is None or tensor_name == "":
            return None, None

        init = get_by_name(model.graph.initializer, tensor_name)
        if init is not None:
            return numpy_helper.to_array(init), None

        prod = model.find_producer(tensor_name)
        if prod is None or prod.op_type != "Constant":
            return None, None

        v = get_by_name(prod.attribute, "value")
        if v is None:
            return None, prod.name  # constant node exists but not supported format
        return numpy_helper.to_array(v.t), prod.name

    def _parse_slice_to_interval(self, model, slice_node, data_shape):
        """
        Returns:
          ok, axis, start, end, reason, used_const_nodes(set)
        """
        used_const_nodes = set()

        # Slice inputs: data, starts, ends, axes?, steps?
        starts_arr, cn1 = self._get_const_array_and_constnode(model, slice_node.input[1])
        ends_arr, cn2 = self._get_const_array_and_constnode(model, slice_node.input[2])
        if cn1: used_const_nodes.add(cn1)
        if cn2: used_const_nodes.add(cn2)

        if starts_arr is None or ends_arr is None:
            return False, None, None, None, "Dynamic starts/ends (not constant)", used_const_nodes

        starts = [int(x) for x in np.array(starts_arr).flatten().tolist()]
        ends = [int(x) for x in np.array(ends_arr).flatten().tolist()]
        if len(starts) != len(ends):
            return False, None, None, None, "Starts/ends length mismatch", used_const_nodes

        # axes
        if len(slice_node.input) >= 4 and slice_node.input[3] != "":
            axes_arr, cn3 = self._get_const_array_and_constnode(model, slice_node.input[3])
            if cn3: used_const_nodes.add(cn3)
            if axes_arr is None:
                return False, None, None, None, "Dynamic axes (not constant)", used_const_nodes
            axes = [int(x) for x in np.array(axes_arr).flatten().tolist()]
        else:
            axes = list(range(len(starts)))  # default

        # steps
        if len(slice_node.input) >= 5 and slice_node.input[4] != "":
            steps_arr, cn4 = self._get_const_array_and_constnode(model, slice_node.input[4])
            if cn4: used_const_nodes.add(cn4)
            if steps_arr is None:
                return False, None, None, None, "Dynamic steps (not constant)", used_const_nodes
            steps = [int(x) for x in np.array(steps_arr).flatten().tolist()]
        else:
            steps = [1] * len(starts)

        if not (len(axes) == len(starts) == len(steps)):
            return False, None, None, None, "Axes/starts/steps length mismatch", used_const_nodes

        # single-axis only
        if len(axes) != 1:
            return False, None, None, None, f"Multi-axis Slice not supported (axes={axes})", used_const_nodes

        axis = int(axes[0])
        if axis < 0:
            axis += len(data_shape)
        if axis < 0 or axis >= len(data_shape):
            return False, None, None, None, "Axis out of range", used_const_nodes

        step = int(steps[0])
        if step != 1:
            return False, None, None, None, f"Non-unit step not supported (step={step})", used_const_nodes

        dim = int(data_shape[axis])
        st = int(starts[0])
        en = int(ends[0])

        # normalize negatives
        if st < 0:
            st += dim
        if en < 0:
            en += dim

        # clamp
        st = max(0, min(dim, st))
        en = max(0, min(dim, en))

        if en < st:
            return False, None, None, None, f"Invalid interval [{st},{en})", used_const_nodes

        return True, axis, st, en, "", used_const_nodes

    def _is_full_tiling(self, intervals, dim: int) -> bool:
        """
        intervals: list of (start, end, slice_node) sorted by start
        Checks they tile [0..dim) contiguously with no gaps/overlaps.
        """
        if not intervals:
            return False
        if intervals[0][0] != 0:
            return False
        prev_end = intervals[0][1]
        for (st, en, _) in intervals[1:]:
            if st != prev_end:
                return False
            prev_end = en
        return prev_end == dim
