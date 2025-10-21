from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import logging

logger = logging.getLogger(__name__)

class FuseElementwiseOps(Transformation):

    """Fuse Elementwise operations like Relu into preceding operators like Conv or Gemm."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        fused = 0

        # Find all Elementwise nodes in the model
        elementwise_ops = ["StreamingReLU"]
        for op_type in elementwise_ops:
            nodes = model.get_nodes_by_op_type(op_type)
            for node in nodes:
                if node.op_type == "StreamingReLU":
                    # Check if the input is from a StreamingConv or StreamingDepthwiseConv node
                    producer = model.find_producer(node.input[0])
                    if producer.op_type in ["StreamingConv", "StreamingDepthwiseConv"]:
                        # Fuse Relu into the producer node
                        getCustomOp(producer).set_nodeattr("activation", "ReLU")
                        # Redirect outputs
                        for i, out in enumerate(producer.output):
                            if out == node.input[0]:
                                producer.output[i] = node.output[0]

                        # Remove the Relu node
                        graph.node.remove(node)
                        logger.info(f"Fused {node.name} into {producer.name}")
                        fused += 1
        
        return model, False