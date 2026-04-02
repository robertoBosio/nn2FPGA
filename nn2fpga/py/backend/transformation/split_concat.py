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
        multiple binary Concat operations.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        changed = False
        nodes_to_process = list(graph.node)

        for node in nodes_to_process:
            if node.op_type != "Concat" or len(node.input) <= 2:
                continue

            logger.info(f"Splitting Concat node '{node.name}' with {len(node.input)} inputs.")
            changed = True

            inputs = list(node.input)
            output = node.output[0]
            axis = next(attr.i for attr in node.attribute if attr.name == "axis")

            new_nodes = []
            split_idx = 0

            while len(inputs) > 2:
                in0 = inputs.pop(0)
                in1 = inputs.pop(0)
                tmp_out = f"{node.name}_split_{split_idx}"

                new_node = helper.make_node(
                    "Concat",
                    inputs=[in0, in1],
                    outputs=[tmp_out],
                    name=f"{node.name}_split_{split_idx}",
                    axis=axis,
                )
                new_nodes.append(new_node)
                inputs.append(tmp_out)
                split_idx += 1

            # Final binary concat writes to the original output name
            final_node = helper.make_node(
                "Concat",
                inputs=[inputs[0], inputs[1]],
                outputs=[output],
                name=f"{node.name}_split_{split_idx}",
                axis=axis,
            )
            new_nodes.append(final_node)

            graph.node.remove(node)
            graph.node.extend(new_nodes)

        return model, changed
