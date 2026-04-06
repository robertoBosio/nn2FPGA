from onnx import helper
from qonnx.custom_op.base import CustomOp
import base64

def _b64_encode(s: str) -> str:
    """Base64 encode."""
    return base64.b64encode(s.encode()).decode()

def _b64_decode(s: str) -> str:
    """Decode base64 string."""
    return base64.b64decode(s.encode()).decode()

class HLSKernel(CustomOp):
    """ Node that represents a custom HLS kernel. """

    @staticmethod
    def make_node(
        name,
        inputs,
        outputs,
        domain,
        original_op_type: str = "",
        hls_object_name: str = "",
        hls_tag: str = "",
        hls_variable_declarations: str = "",
        hls_object_declaration: str = "",
        hls_run_call: str = "",
        hls_step_call: str = "",
    ):
        return helper.make_node(
            "HLSKernel",
            inputs,
            outputs,
            name=name,
            domain=domain,
            original_op_type=original_op_type,
            hls_object_name=_b64_encode(hls_object_name),
            hls_tag=_b64_encode(str(hls_tag)),
            hls_variable_declarations=_b64_encode(hls_variable_declarations),
            hls_object_declaration=_b64_encode(hls_object_declaration),
            hls_run_call=_b64_encode(hls_run_call),
            hls_step_call=_b64_encode(hls_step_call),
        )
    
    def get_nodeattr(self, attr_name):
        encoded_attr = super().get_nodeattr(attr_name)
        if "hls" in attr_name:
            return _b64_decode(encoded_attr) if encoded_attr != "" else ""
        return encoded_attr

    def get_nodeattr_types(self):
        return {
            "original_op_type": ("s", True, ""),
            "read_skew": ("i", False, 0), # Maximum skew in cycles inside the pipeline between the fifo reads.
            "write_skew": ("i", False, 0), # Maximum skew in cycles inside the pipeline between the fifo writes.
            "pipeline_stages": ("i", False, 1), # Number of stages in the pipeline. 
            "hls_tag": ("s", True, ""),
            "hls_object_name": ("s", True, ""), # Name of the HLS object (class instance).
            "hls_variable_declarations": ("s", False, ""),
            "hls_object_declaration": ("s", True, ""),
            "hls_run_call": ("s", True, ""),
            "hls_step_call": ("s", True, ""),
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

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_name = node.input[0]
        out_name = node.output[0]
        inp = context[inp_name]
        context[out_name] = inp

    def verify_node(self):
        pass
