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

class StreamingAdd(CustomOp):
    """ Node implementing the output-stationary convolution operation. """

    @staticmethod
    def pattern(op, a, b):
        return op.Add(a, b, _allow_other_attributes=True)

    @staticmethod
    def rewrite(op, a, b):
        return op.StreamingAdd(
            a,
            b,
            _domain="backend.custom_op",
        )

    @register_rules
    def register_rules():
        return [
            pattern.RewriteRule(
                StreamingAdd.pattern, StreamingAdd.rewrite
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
            "Add",
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
            "Add",
            inputs=node.input,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
        )

        # Make single node graph for execution
        inpA_values = context[node.input[0]]
        inpB_values = context[node.input[1]]
        oshape = context[node.output[0]].shape
        inpA = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, inpA_values.shape)
        inpB = helper.make_tensor_value_info(node.input[1], TensorProto.FLOAT, inpB_values.shape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)

        graph_conv = helper.make_graph(
            nodes=[node_conv],
            name="single-conv-exec",
            inputs=[inpA, inpB],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_conv = qonnx_make_model(graph_conv, **onnx_kwargs)
        idict = {node.input[0]: inpA_values, node.input[1]: inpB_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_conv.SerializeToString())
        result = np.array(sess.run(None, idict)[0])
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass
    
    def __get_stream_name(self, name: str) -> str:
        """
        Returns the name of the stream for the tensor.
        """
        return f"{name}_stream"

    def __get_variable_declaration(self, model) -> str:
        """ Get the internal cpp variables of the StreamingConv node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""
    
    def __is_power_of_two(self, value) -> bool:
        """Check if a value is a power of two."""
        return value > 0 and float(np.log2(value)).is_integer()

    def __get_accumulator(self, input_quant0, input_quant1):
        signed = input_quant0.signed or input_quant1.signed
        acc_bits = max(input_quant0.bitwidth, input_quant1.bitwidth) + 1
        acc_quant = TensorQuant(
            bitwidth=acc_bits,
            signed=signed,
            scale=input_quant0.scale,
            zeropt=input_quant0.zeropt
        )
        return f"{get_hls_quant_type(acc_quant)}"

    def __get_quantizer(self, input_quant0, input_quant1, output_quant) -> str:
        """ Returns the quantizer type for the Add operation. """

        if (
            self.__is_power_of_two(input_quant0.scale)
            and self.__is_power_of_two(input_quant1.scale)
            and self.__is_power_of_two(output_quant.scale)
        ):
            shift = -1 * (
                int(np.log2(input_quant0.scale))
                - int(np.log2(output_quant.scale))
            )
            return f"DequantQuantPo2<{shift}, {self.__get_accumulator(input_quant0, input_quant1)}, {get_hls_quant_type(output_quant)}>"
        else:
            raise ValueError(
                "Float quantization is currently not supported for StreamingGlobalAveragePool.  "
            )

    def __get_object_declaration(self, model) -> cpp_object:

        input_quant0 = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if (input_quant0 is None):
            raise ValueError(f"Input {self.onnx_node.input[0]} has no quantization info")
        input_quant1 = get_custom_tensor_datatype(model, self.onnx_node.input[1])
        if (input_quant1 is None):
            raise ValueError(f"Input {self.onnx_node.input[1]} has no quantization info")
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if (output_quant is None):
            raise ValueError(f"Output {self.onnx_node.output[0]} has no quantization info")
        
        # Retrieve parallelization attributes.
        par_attribute = get_par_attributes(self.onnx_node)

        input_shape0 = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape0 is None:
            raise ValueError(f"Input {self.onnx_node.input[0]} has no shape info")
        input_shape1 = model.get_tensor_shape(self.onnx_node.input[1])
        if input_shape1 is None:
            raise ValueError(f"Input {self.onnx_node.input[1]} has no shape info")
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Output {self.onnx_node.output[0]} has no shape info")
        
        StreamingAdd = cpp_object(
            "StreamingAdd",
            f"{self.onnx_node.name}",
            template_args=[
                (
                    f"{get_struct_type(input_quant0, par_attribute['in_ch_par'])}",
                    f"TInputWord",
                ),
                (
                    f"{get_hls_quant_type(input_quant0)}",
                    f"TInput",
                ),
                (
                    f"{get_struct_type(output_quant, par_attribute['out_ch_par'])}",
                    f"TOutputWord",
                ),
                (
                    f"{get_hls_quant_type(output_quant)}",
                    f"TOutput",
                ),
                (
                    f"{self.__get_accumulator(input_quant0, input_quant1)}",
                    f"TAcc",
                ),
                (
                    f"{self.__get_quantizer(input_quant0, input_quant1, output_quant)}",
                    f"Quantizer",
                ),
                (f"{input_shape0[2]}", "IN_HEIGHT"),
                (f"{input_shape0[3]}", "IN_WIDTH"),
                (f"{input_shape0[1]}", "IN_CH"),
                (f"{par_attribute['in_w_par']}", "W_PAR"),
                (f"{par_attribute['in_ch_par']}", "CH_PAR"),
            ]
        )

        return StreamingAdd.generate_declaration()

    def __get_run_call(self, hls_tag: int) -> str:

        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(
                (
                    f"i_data0",
                    f"hls::stream<TInputWord>", 
                ),
                (
                    f"i_data1",
                    f"hls::stream<TInputWord>", 
                ),
                (
                    f"o_data",
                    f"hls::stream<TOutputWord>", 
                ),
            )
        )

        return run.generate_call(
            [hls_tag],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.input[1]),
            self.__get_stream_name(self.onnx_node.output[0]),
        )
    
    def __get_step_call(self) -> str:

        run = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=(
                (
                    f"i_data0",
                    f"hls::stream<TInputWord>", 
                ),
                (
                    f"i_data1",
                    f"hls::stream<TInputWord>", 
                ),
                (
                    f"o_data",
                    f"hls::stream<TOutputWord>", 
                ),
            )
        )

        return run.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.input[1]),
            self.__get_stream_name(self.onnx_node.output[0]),
        )
    
    def lower_to_hls(self, model: ModelWrapper, hls_tag: int) -> None:
        """ Lower the node to HLS code.
        """

        par = get_par_attributes(self.onnx_node)
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(par['in_w_par'])
        ]
        input_names.extend([
            f"{self.__get_stream_name(self.onnx_node.input[1])}_{i}_"
            for i in range(par['in_w_par'])
        ])

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
            original_op_type="StreamingAdd",
            hls_tag=hls_tag,
            hls_object_name=self.onnx_node.name,
            hls_variable_declarations=self.__get_variable_declaration(model),
            hls_run_call=self.__get_run_call(hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )
        hls_tag += 1

        return [hls_kernel], [], tensors_fifo_metadata, hls_tag