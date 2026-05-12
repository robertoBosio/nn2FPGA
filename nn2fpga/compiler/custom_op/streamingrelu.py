import numpy as np
import onnxruntime as rt
from onnxscript.rewriter import pattern
from onnx import TensorProto, helper
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.core.tensor_type import require_tensor_type
from nn2fpga.compiler.core.tensor_layout import require_tensor_layout
from nn2fpga.compiler.core.tensor_fifo import TensorFifo
from nn2fpga.compiler.custom_op.hlskernel import HLSKernel
from nn2fpga.compiler.custom_op.op_base import NN2FPGAOp, NodeInterface
from nn2fpga.compiler.custom_op.register_rewrite_rule import register_rules
from nn2fpga.compiler.utils.codegen_utils import (
    cpp_function,
    cpp_object,
    get_word_type,
)


class StreamingReLU(NN2FPGAOp):
    """Node implementing the ReLU operation."""

    @staticmethod
    def pattern(op, x):
        return op.Relu(x, _allow_other_attributes=True)

    @staticmethod
    def rewrite(op, x):
        return op.StreamingReLU(
            x,
            _domain="nn2fpga.compiler.custom_op",
        )

    @register_rules
    def register_rules():
        return [pattern.RewriteRule(StreamingReLU.pattern, StreamingReLU.rewrite)]

    def get_nodeattr_types(self):
        return {
            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),
            "dim2_unroll": ("i", False, 1),
            "dim1_unroll": ("i", False, 1),
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
        # create a standard relu node to compute the result
        node = self.onnx_node
        node_relu = helper.make_node(
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

        graph_relu = helper.make_graph(
            nodes=[node_relu],
            name="single-relu-exec",
            inputs=[inp],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_relu = qonnx_make_model(graph_relu, **onnx_kwargs)
        idict = {node.input[0]: inp_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_relu.SerializeToString())
        result = np.array(sess.run(None, idict)[0])
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass

    def __is_power_of_two(self, value) -> bool:
        """Check if a value is a power of two."""
        return value > 0 and float(np.log2(value)).is_integer()

    def __get_stream_name(self, name: str) -> str:
        """
        Returns the name of the stream for the tensor.
        """
        return f"{name}_stream"

    def __get_variable_declaration(self, model) -> str:
        """Get the internal cpp variables of the StreamingReLU node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __get_quantizer(self, input_quant, output_quant) -> str:
        """Returns the quantizer type for the ReLU operation."""

        if self.__is_power_of_two(input_quant.scale) and self.__is_power_of_two(
            output_quant.scale
        ):
            shift = -1 * int(np.log2(output_quant.scale) - np.log2(input_quant.scale))
            if (
                shift == 0
                and input_quant.bitwidth == output_quant.bitwidth
                and input_quant.signed == output_quant.signed
            ):
                return f"DequantQuantEqual<{input_quant.get_hls_data_type()}>"
            return f"DequantQuantPo2<{shift}, {input_quant.get_hls_data_type()}, {output_quant.get_hls_data_type()}>"
        else:
            raise ValueError(
                "Float quantization is currently not supported for StreamingReLU."
            )

    def __get_object_declaration(self, model) -> cpp_object:

        input_quant = require_tensor_type(model, self.onnx_node.input[0])
        output_quant = require_tensor_type(model, self.onnx_node.output[0])
        input_layout = require_tensor_layout(model, self.onnx_node.input[0])
        input_shape = self.require_4d_input_shape(model, 0, input_layout)

        StreamingReLU = cpp_object(
            "StreamingReLU",
            f"{self.onnx_node.name}",
            template_args=[
                (
                    f"{get_word_type(input_quant, self.get_nodeattr('in_word_array'))}",
                    f"TInputWord",
                ),
                (
                    f"{input_quant.get_hls_data_type()}",
                    f"TInput",
                ),
                (
                    f"{get_word_type(output_quant, self.get_nodeattr('out_word_array'))}",
                    f"TOutputWord",
                ),
                (
                    f"{output_quant.get_hls_data_type()}",
                    f"TOutput",
                ),
                (
                    f"{self.__get_quantizer(input_quant, output_quant)}",
                    f"Quantizer",
                ),
                (f"{input_shape[-3]}", "DIM0"),
                (f"{input_shape[-2]}", "DIM1"),
                (f"{input_shape[-1]}", "DIM2"),
                (f"{self.get_nodeattr('dim1_unroll')}", "DIM1_UNROLL"),
                (f"{self.get_nodeattr('dim2_unroll')}", "DIM2_UNROLL"),
            ],
        )

        return StreamingReLU.generate_declaration()

    def __get_run_call(self, hls_tag: int) -> str:

        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(
                (
                    f"i_data",
                    f"hls::stream<TInputWord>",
                ),
                (
                    f"o_data",
                    f"hls::stream<TOutputWord>",
                ),
            ),
        )

        return run.generate_call(
            [hls_tag],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def __get_step_call(self) -> str:

        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=(
                (
                    f"i_data",
                    f"hls::stream<TInputWord>",
                ),
                (
                    f"o_data",
                    f"hls::stream<TOutputWord>",
                ),
            ),
        )

        return step.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.output[0]),
        )
    
    def accepted_input_layout(self) -> tuple | None:
        """ StreamingReLU is layout-agnostic, any layout is accepted. """
        return None
    
    def produced_output_layout(self, input_layout: tuple | None) -> tuple | None:
        """ StreamingReLU is layout-agnostic, output layout matches input layout. """
        return input_layout

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int) -> None:
        """Lower the node to HLS code."""

        output_quant = require_tensor_type(model, self.onnx_node.output[0])
        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(self.get_nodeattr("in_stream_array"))
        ]

        output_names = [
            f"{self.__get_stream_name(self.onnx_node.output[0])}_{i}_"
            for i in range(self.get_nodeattr("out_stream_array"))
        ]

        tensors_fifo_metadata = {}
        for output in output_names:
            tensors_fifo_metadata[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_word_type(output_quant, self.get_nodeattr('out_word_array'))}",
                n_array=self.get_nodeattr("out_stream_array"),
            )

        hls_kernel = HLSKernel.make_node(
            inputs=input_names,
            outputs=output_names,
            name=f"{self.onnx_node.name}_hls",
            domain="nn2fpga.compiler.custom_op",
            original_op_type="StreamingReLU",
            hls_tag=hls_tag,
            hls_object_name=self.onnx_node.name,
            hls_variable_declarations=self.__get_variable_declaration(model),
            hls_run_call=self.__get_run_call(hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )
        hls_tag += 1

        return [hls_kernel], [], tensors_fifo_metadata, hls_tag

    def get_latency(self, model: ModelWrapper) -> int:
        """Estimate the latency of the StreamingReLU operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated latency in clock cycles.
        """
        input_shape = self.require_4d_input_shape(model, 0)

        unroll_factor = self.get_nodeattr("dim2_unroll") * self.get_nodeattr(
            "dim1_unroll"
        )
        return np.prod(input_shape) // unroll_factor

    def get_brams(self, model: ModelWrapper) -> int:
        """Estimate the BRAM usage of the StreamingReLU operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated BRAM usage.
        """
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        """Estimate the DSP usage of the StreamingReLU operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated DSP usage.
        """
        return 0

    def has_linebuffer(self) -> bool:
        """Check if the StreamingReLU operation requires a line buffer.
        Returns:
            bool: True if Line Buffering is required, False otherwise.
        """
        return False

    def can_inherit_interface(self):
        return True

    def inherit_interface(self, model: ModelWrapper, upstream: NodeInterface) -> None:
        """Inherit the interface from the upstream node."""
        self.set_nodeattr("in_stream_array", upstream.out_stream_array)
        self.set_nodeattr("out_stream_array", upstream.out_stream_array)
        self.set_nodeattr("in_word_array", upstream.out_word_array)
        self.set_nodeattr("out_word_array", upstream.out_word_array)

        self.set_nodeattr("dim2_unroll", upstream.out_word_array)
        self.set_nodeattr("dim1_unroll", upstream.out_stream_array)
