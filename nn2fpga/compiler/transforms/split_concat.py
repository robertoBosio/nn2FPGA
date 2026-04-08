from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from onnx import helper
import logging

logger = logging.getLogger(__name__)


class SplitConcat(Transformation):
    """
    Replace:
        Concat(x0, x1, x2, ..., xN, axis)
    with:
        a chain of binary Concat operations that preserves input order:
            t0 = Concat(x0, x1)
            t1 = Concat(t0, x2)
            ...
            out = Concat(t{k}, xN)
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        changed = False
        nodes_to_process = list(graph.node)

        for node in nodes_to_process:
            if node.op_type != "Concat" or len(node.input) <= 2:
                continue

            logger.info(
                "Splitting Concat node '%s' with %d inputs.",
                node.name,
                len(node.input),
            )
            changed = True

            inputs = list(node.input)
            output = node.output[0]

            axis_attr = next((attr for attr in node.attribute if attr.name == "axis"), None)
            axis = axis_attr.i if axis_attr is not None else 0

            new_nodes = []

            current = inputs[0]
            split_idx = 0

            for next_input in inputs[1:-1]:
                tmp_out = f"{node.name}_split_{split_idx}"
                new_node = helper.make_node(
                    "Concat",
                    inputs=[current, next_input],
                    outputs=[tmp_out],
                    name=f"{node.name}_split_{split_idx}",
                    axis=axis,
                )
                new_nodes.append(new_node)
                current = tmp_out
                split_idx += 1

            final_node = helper.make_node(
                "Concat",
                inputs=[current, inputs[-1]],
                outputs=[output],
                name=f"{node.name}_split_{split_idx}",
                axis=axis,
            )
            new_nodes.append(final_node)

            insert_idx = list(graph.node).index(node)
            graph.node.remove(node)
            for offset, new_node in enumerate(new_nodes):
                graph.node.insert(insert_idx + offset, new_node)

        return model, changed