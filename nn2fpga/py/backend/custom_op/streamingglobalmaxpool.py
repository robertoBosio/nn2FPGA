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

class StreamingGlobalMaxPool(CustomOp):
    """Node implementing the streaming global max pooling operation."""

    @staticmethod
    def pattern(op, x):
        return op.MaxPool(x, _allow_other_attributes=True)
    @staticmethod
    def rewrite(op, x):
        return op.StreamingGlobalMaxPool(
            x,
            _domain="backend.custom_op",
        )

    @register_rules
    def register_rules():
        return [
            pattern.RewriteRule(
                StreamingGlobalMaxPool.pattern, StreamingGlobalMaxPool.rewrite
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
            "GlobalMaxPool",
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
            "GlobalMaxPool",
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
        model_globalaveragepool = qonnx_make_model(graph_globaleaveragepool, **onnx_kwargs)
        idict = {node.input[0]: inp_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_globalaveragepool.SerializeToString())
        result = np.array(sess.run(None, idict)[0])
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass

    def __get_stream_name(self, name: str) -> str:
        """
        Returns the name of the stream for the tensor.
        """
        return f"{name}_stream"

    def __get_accumulator(self, input_quant, input_shape) -> str:
        """ Returns the accumulator type for the StreamingGlobalAveragePool operation. """

        add_ops = input_shape[2] * input_shape[3] # H * W
        acc_bitwidth = input_quant.bitwidth + int(np.ceil(np.log2(add_ops)))
        acc_quant = TensorQuant(
            bitwidth=acc_bitwidth,
            signed=input_quant.signed,
            scale=input_quant.scale,
            zeropt=input_quant.zeropt,
        )

        return f"{get_hls_quant_type(acc_quant)}"

    def __get_divisor(self, input_shape) -> str:
        """ Returns the divisor type for the StreamingGlobalAveragePool operation. """

        divisor = input_shape[2] * input_shape[3]
        divisor_quant = TensorQuant(
            bitwidth=int(np.ceil(np.log2(divisor + 1))),
            signed=False,
            scale=1.0,
            zeropt=0,
        )
        return f"{get_hls_quant_type(divisor_quant)}"

    def __is_power_of_two(self, value) -> bool:
        """Check if a value is a power of two."""
        return value > 0 and float(np.log2(value)).is_integer()

    def __get_quantizer(self, input_quant, output_quant, input_shape) -> str:
        """ Returns the quantizer type for the StreamingGlobalAveragePool operation. """

        # Check if the scale is a power of two
        if self.__is_power_of_two(input_quant.scale) and self.__is_power_of_two(output_quant.scale):
            shift = int(np.log2(input_quant.scale)) - int(np.log2(output_quant.scale))
            return f"DequantQuantPo2<{shift}, {self.__get_accumulator(input_quant, input_shape)}, {get_hls_quant_type(output_quant)}>"
        else:
            raise ValueError(
                "Float quantization is currently not supported for StreamingGlobalAveragePool.  "
            )

    def __get_object_declaration(self, model) -> cpp_object:
        """ Generate the cpp_object for the StreamingGlobalAveragePool operation. """

        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model.")

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        # Retrieve parallelization attributes.
        par_attribute = get_par_attributes(self.onnx_node)

        # Retrieve tensor shape.
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model.")
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model.")

        # Create the StreamingGlobalAveragePool object.
        StreamingGlobalAveragePool = cpp_object(
            "StreamingGlobalAveragePool",
            f"{self.onnx_node.name}",
            template_args=[
                (f"{get_struct_type(input_quant, par_attribute['in_ch_par'])}", "TInputStruct"),
                (f"{get_hls_quant_type(input_quant)}", "TInput"),
                (f"{get_struct_type(output_quant, par_attribute['out_ch_par'])}", "TOutputStruct"),
                (f"{get_hls_quant_type(output_quant)}", "TOutput"),
                (self.__get_accumulator(input_quant, input_shape), "TAcc"),
                (self.__get_divisor(input_shape), "TDiv"),
                (self.__get_quantizer(input_quant, output_quant, input_shape), "Quantizer"),
                (input_shape[2], "IN_HEIGHT"),
                (input_shape[3], "IN_WIDTH"),
                (output_shape[1], "OUT_CH"),
                (par_attribute["out_ch_par"], "OUT_CH_PAR"),
            ])

        return StreamingGlobalAveragePool.generate_declaration()

    def __get_variable_declaration(self, model) -> str:
        """ Get the internal cpp variables of the StreamingGlobalAveragePool node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __get_run_call(self, hls_tag: int) -> str:
        """ Generates the C++ code necessary to run the StreamingGlobalAveragePool node. """

        # Generate the call to the StreamingGlobalAveragePool run method.
        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(
                (
                    f"input_data_stream",
                    f"hls::stream<TInputStruct>",
                ),
                (
                    f"output_data_stream",
                    f"hls::stream<TOutputStruct>",
                ),
            ),
        )

        return run.generate_call(
            [hls_tag],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def __get_step_call(self) -> str:
        """ Generates the C++ code necessary to step the StreamingGlobalAveragePool node. """

        # Generate the call to the StreamingGlobalAveragePool step method.
        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=(
                (
                    f"input_data_stream",
                    f"hls::stream<TInputStruct>",
                ),
                (
                    f"output_data_stream",
                    f"hls::stream<TOutputStruct>",
                ),
            ),
        )

        return step.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int):
        """
        Returns:
          nodes: List[onnx.NodeProto]
          initializers: List[onnx.TensorProto]
          fifo: Dict[str, TensorFifo]
        """

        par = get_par_attributes(self.onnx_node)
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(par["in_w_par"])
        ]

        output_names = [
            f"{self.__get_stream_name(self.onnx_node.output[0])}_{i}_"
            for i in range(par["out_w_par"])
        ]

        tensors_fifo_metadata = {}
        for output in output_names:
            tensors_fifo_metadata[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(output_quant, par['out_ch_par'])}",
                n_array=par["out_w_par"],
            )

        hls_kernel = HLSKernel.make_node(
            inputs=input_names,
            outputs=output_names,
            name=f"{self.onnx_node.name}_hls",
            domain="backend.custom_op",
            original_op_type=self.onnx_node.op_type,
            hls_object_name=self.onnx_node.name,
            hls_tag=hls_tag,
            hls_variable_declarations=self.__get_variable_declaration(model),
            hls_run_call=self.__get_run_call(hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )
        hls_tag += 1

        return [hls_kernel], [], tensors_fifo_metadata, hls_tag
