from onnx import helper
from qonnx.custom_op.base import CustomOp
from nn2fpga.compiler.core.simulation import simulate

class nn2fpgaPartition(CustomOp):
    """ Operator representing the nn2fpga partition in the wrapper model. """

    def execute_node(self, context, graph):

        return simulate(
            accelerator_package_serialized=self.get_nodeattr("accelerator_package"),
            context=context,
        )

    def get_nodeattr_types(self):
        """
        Get the attribute types for the nn2fpgaPartition node.
        The attribute "blob" is added for nn2fpga to handle additional data, such as HLS code,
        or input data for the partition.
        This allows to embed everything needed for nn2fpga simulation directly into the ONNX model.
        """

        return {
            "accelerator_package": ("s", False, ""),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        return helper.make_node(
            "Identity",
            [node.input[0]],
            [node.output[0]],
            name=f"{node.name}_shape_compatible",
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def verify_node(self):
        pass
