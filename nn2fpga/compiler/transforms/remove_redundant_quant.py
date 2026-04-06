from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from nn2fpga.compiler.core.tensor_quant import TensorQuant

logger = logging.getLogger(__name__)


class RemoveRedundantQuant(Transformation):
    """Remove adjacent redundant IntQuant/Quant nodes with identical quantization
    parameters.

    Rules:
    - Normal case:
        Q1 -> Q2, same quant params, Q2 not feeding graph output
        => remove Q2, redirect Q2.output to Q1.output

    - Graph-output case:
        Q1 -> Q2, same quant params, Q2 feeds graph output
        => remove Q1, rewire Q2.input to Q1.input, and redirect all other consumers
           of Q1.output to Q2.output

    This preserves graph output tensor names while still eliminating the duplicate.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.model.graph

        def is_quant(node) -> bool:
            return node.op_type in ("IntQuant", "Quant")

        def get_node_name(node, idx: int) -> str:
            if node.name:
                return node.name
            return f"__unnamed_node_{idx}"

        # ----------------------------
        # Pass 0: analyze graph
        # ----------------------------
        graph_output_names = {o.name for o in graph.output if o.name}

        producer_map: dict[str, object] = {}
        consumers_map: dict[str, list[object]] = defaultdict(list)
        node_name_to_node: dict[str, object] = {}
        node_order: list[object] = []
        node_index_by_name: dict[str, int] = {}

        for idx, node in enumerate(graph.node):
            nname = get_node_name(node, idx)
            node_name_to_node[nname] = node
            node_order.append(node)
            node_index_by_name[nname] = idx
            for out in node.output:
                if out:
                    producer_map[out] = node
            for inp in node.input:
                if inp:
                    consumers_map[inp].append(node)

        # ----------------------------
        # Transformation decisions
        # ----------------------------
        remove_node_names: set[str] = set()

        # generic tensor aliasing, used in normal "remove Q2" case
        alias_map: dict[str, str] = {}

        # special-case rewiring for kept Q2 in graph-output case:
        # forced_input0[q2_name] = new_input_tensor
        forced_input0: dict[str, str] = {}

        # tensors whose consumers (except some excluded nodes) must be rewritten
        # consumer_rewrite[src_tensor] = dst_tensor
        consumer_rewrite: dict[str, str] = {}

        # excluded_consumer_rewrite[(node_name, input_index)] means:
        # do NOT apply consumer_rewrite to this exact input slot
        excluded_consumer_rewrite: set[tuple[str, int]] = set()

        run_again = False

        def resolve_alias(t: str) -> str:
            seen = set()
            while t in alias_map and t not in seen:
                seen.add(t)
                t = alias_map[t]
            return t

        # ----------------------------
        # Pass 1: decide removals
        # ----------------------------
        for idx, node in enumerate(graph.node):
            node_name = get_node_name(node, idx)

            if not is_quant(node):
                continue
            if len(node.input) == 0 or not node.input[0]:
                continue
            if len(node.output) == 0 or not node.output[0]:
                continue

            inp0 = resolve_alias(node.input[0])
            prev_node = producer_map.get(inp0)
            if prev_node is None or not is_quant(prev_node):
                continue

            prev_name = get_node_name(
                prev_node,
                node_index_by_name[
                    get_node_name(
                        prev_node,
                        node_index_by_name.get(get_node_name(prev_node, 0), 0),
                    )
                ],
            )

            if prev_name in remove_node_names or node_name in remove_node_names:
                continue

            # true direct producer check
            if inp0 not in prev_node.output:
                continue
            if len(prev_node.input) == 0 or not prev_node.input[0]:
                continue
            if len(prev_node.output) == 0 or not prev_node.output[0]:
                continue

            q1 = TensorQuant.from_quant_node(prev_node, model)
            q2 = TensorQuant.from_quant_node(node, model)

            if q1 != q2:
                logger.info(
                    "Keeping quant node %s as quant params differ from %s",
                    node.name,
                    prev_node.name,
                )
                continue

            prev_inp = prev_node.input[0]
            prev_out = prev_node.output[0]
            node_out = node.output[0]

            if node_out in graph_output_names:
                # Special case: keep downstream quant so graph output tensor name stays stable.
                logger.info(
                    "Removing redundant upstream quant node %s and keeping %s "
                    "because %s feeds a graph output",
                    prev_node.name,
                    node.name,
                    node.name,
                )

                remove_node_names.add(prev_name)
                forced_input0[node_name] = prev_inp

                # Move all other consumers of prev_out from prev_out -> node_out.
                consumer_rewrite[prev_out] = node_out

                # But do NOT rewrite node.input[0] through that mapping; it must become prev_inp.
                for k, inp in enumerate(node.input):
                    if inp == prev_out:
                        excluded_consumer_rewrite.add((node_name, k))

                run_again = True
            else:
                # Normal case: remove downstream quant and alias its output to prev_out.
                logger.info(
                    "Removing redundant quant node %s with identical quant params to %s",
                    node.name,
                    prev_node.name,
                )
                remove_node_names.add(node_name)
                alias_map[node_out] = prev_out
                run_again = True

        if not run_again:
            return (model, False)

        # ----------------------------
        # Pass 2: rebuild graph
        # ----------------------------
        new_nodes = []

        def rewrite_tensor_for_input(
            tensor_name: str, node_name: str, input_index: int
        ) -> str:
            if not tensor_name:
                return tensor_name

            # First apply normal alias-chain resolution
            tensor_name = resolve_alias(tensor_name)

            # Then apply consumer-side rewrite, unless this exact slot is excluded
            if (node_name, input_index) not in excluded_consumer_rewrite:
                if tensor_name in consumer_rewrite:
                    tensor_name = consumer_rewrite[tensor_name]

            # Final alias cleanup in case consumer_rewrite target itself is aliased
            tensor_name = resolve_alias(tensor_name)
            return tensor_name

        for idx, node in enumerate(graph.node):
            node_name = get_node_name(node, idx)
            if node_name in remove_node_names:
                continue

            new_node = deepcopy(node)

            # Special case first: forced input0 for kept downstream quant
            if node_name in forced_input0 and len(new_node.input) > 0:
                new_node.input[0] = forced_input0[node_name]

            # Rewrite all inputs
            for k, inp in enumerate(new_node.input):
                if not inp:
                    continue

                # If input0 was forced above, preserve it from the generic rewrite source value.
                if k == 0 and node_name in forced_input0:
                    inp = new_node.input[0]

                new_node.input[k] = rewrite_tensor_for_input(inp, node_name, k)

            new_nodes.append(new_node)

        # Replace graph nodes
        graph.ClearField("node")
        graph.node.extend(new_nodes)

        # Do NOT rewrite graph.output names.
        # We intentionally preserve terminal tensor names.

        # ----------------------------
        # Cleanup stale ValueInfo / initializers / annotations for dead tensors
        # ----------------------------
        live_tensors: set[str] = set()

        for node in graph.node:
            for inp in node.input:
                if inp:
                    live_tensors.add(inp)
            for out in node.output:
                if out:
                    live_tensors.add(out)

        for vi in graph.input:
            if vi.name:
                live_tensors.add(vi.name)
        for vi in graph.output:
            if vi.name:
                live_tensors.add(vi.name)
        for init in graph.initializer:
            if init.name:
                live_tensors.add(init.name)

        def filter_valueinfo_container(container):
            keep = [vi for vi in container if vi.name in live_tensors]
            del container[:]
            container.extend(keep)

        filter_valueinfo_container(graph.value_info)

        # graph.input / graph.output must not be filtered away, only kept as-is.
        # But duplicate names can still be problematic, so remove exact duplicates
        # from value_info if a name already exists in graph.input or graph.output.
        boundary_names = {x.name for x in graph.input if x.name} | {
            x.name for x in graph.output if x.name
        }

        deduped_vi = []
        seen_vi = set()
        for vi in graph.value_info:
            if not vi.name:
                deduped_vi.append(vi)
                continue
            if vi.name in boundary_names:
                continue
            if vi.name in seen_vi:
                continue
            seen_vi.add(vi.name)
            deduped_vi.append(vi)

        graph.ClearField("value_info")
        graph.value_info.extend(deduped_vi)

        return (model, True)
