from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from fractions import Fraction
import logging

logger = logging.getLogger(__name__)

class FuseElementwiseOps(Transformation):

    """Fuse Elementwise operations like Relu into preceding operators like Conv or Gemm."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        fused = 0

        # Find all Elementwise nodes in the model
        elementwise_ops = ["StreamingReLU", "StreamingLeakyReLU"]
        for op_type in elementwise_ops:
            nodes = model.get_nodes_by_op_type(op_type)
            for node in nodes:
                # Check if the input is from a StreamingConv, StreamingDepthwiseConv, or StreamingAdd node
                producer = model.find_producer(node.input[0])
                if producer is None or producer.op_type not in ["StreamingConv", "StreamingDepthwiseConv", "StreamingAdd"]:
                    continue

                producer_op = getCustomOp(producer)
                if node.op_type == "StreamingReLU":
                    producer_op.set_nodeattr("activation", "ReLU")
                elif node.op_type == "StreamingLeakyReLU":
                    alpha = Fraction(getCustomOp(node).get_nodeattr("alpha")).limit_denominator()
                    producer_op.set_nodeattr("activation", "LeakyReLU")
                    producer_op.set_nodeattr("activation_alpha_num", alpha.numerator)
                    producer_op.set_nodeattr("activation_alpha_den", alpha.denominator)
                else:
                    continue

                # Redirect outputs
                for i, out in enumerate(producer.output):
                    if out == node.input[0]:
                        producer.output[i] = node.output[0]

                # Remove the elementwise node
                graph.node.remove(node)
                logger.info(f"Fused {node.name} into {producer.name}")
                fused += 1

        return model, fused > 0
