import numpy as np
import onnxruntime as rt
from onnx import TensorProto, helper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import get_custom_tensor_datatype
from backend.core.tensor_fifo import TensorFifo
from backend.custom_op.hlskernel import HLSKernel
from backend.util.codegen_utils import (
    cpp_function,
    cpp_variable,
    cpp_object,
    get_struct_type,
    get_stream_type,
    get_hls_quant_type,
)
from backend.core.tensor_quant import TensorQuant
from backend.util.par_utils import get_par_attributes
from backend.custom_op.register_rewrite_rule import register_rules
from onnxscript.rewriter import pattern

class StreamingReLU(CustomOp):
    """ Node implementing the ReLU operation. """

    @staticmethod
    def pattern(op, x):
        return op.Relu(x, _allow_other_attributes=True)

    @staticmethod
    def rewrite(op, x):
        return op.StreamingReLU(
            x,
            _domain="backend.custom_op",
        )

    @register_rules
    def register_rules():
        return [
            pattern.RewriteRule(
                StreamingReLU.pattern, StreamingReLU.rewrite
            )
        ]

    def get_nodeattr_types(self):
        return {
            "in_ch_par": ("i", False, 1),  # Input channel parallelization
            "out_ch_par": ("i", False, 1),  # Output channel parallelization
            "in_w_par": ("i", False, 1),  # Input width parallelization
            "out_w_par": ("i", False, 1),  # Output width parallelization
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node

        return helper.make_node(
            "Relu",
            inputs=node.input,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        # create a standard conv node to compute the result
        node = self.onnx_node
        node_conv = helper.make_node(
            "Relu",
            inputs=node.input,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
        )

        # Make single node graph for execution
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)

        graph_globaleaveragepool = helper.make_graph(
            nodes=[node_conv],
            name="single-conv-exec",
            inputs=[inp],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_relu = qonnx_make_model(graph_globaleaveragepool, **onnx_kwargs)
        idict = {node.input[0]: inp_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_relu.SerializeToString())
        result = np.array(sess.run(None, idict)[0])
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass
