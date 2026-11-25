import numpy as np
import onnxruntime as rt
from fractions import Fraction
from onnxscript.rewriter import pattern
from onnx import TensorProto, helper
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.core.tensor_type import QuantizedTensorType, require_tensor_type
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
from nn2fpga.compiler.utils.scale_ratio import ratio_to_shift_form

class StreamingAdd(NN2FPGAOp):
    """Node implementing the Add operation."""

    @staticmethod
    def pattern(op, a, b):
        return op.Add(a, b, _allow_other_attributes=True)

    @staticmethod
    def rewrite(op, a, b):
        return op.StreamingAdd(
            a,
            b,
            _domain="nn2fpga.compiler.custom_op",
        )

    @register_rules
    def register_rules():
        return [pattern.RewriteRule(StreamingAdd.pattern, StreamingAdd.rewrite)]

    def get_nodeattr_types(self):
        return {
            # Custom attributes for unroll factors
            "dim2_unroll": ("i", False, 1),
            "dim1_unroll": ("i", False, 1),
            # Custom attributes for input/output streams
            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),
            "activation": ("s", False, "NoOp"),  # NoOp, ReLU, LeakyReLU
            "activation_alpha_num": ("i", False, 1),
            "activation_alpha_den": ("i", False, 1),
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
        # create a standard add node to compute the result
        node = self.onnx_node
        node_add = helper.make_node(
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

        graph_add = helper.make_graph(
            nodes=[node_add],
            name="single-add-exec",
            inputs=[inpA, inpB],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_add = qonnx_make_model(graph_add, **onnx_kwargs)
        idict = {node.input[0]: inpA_values, node.input[1]: inpB_values}

        # Execute the model using ONNX Runtime
        sess = rt.InferenceSession(model_add.SerializeToString())
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
        """ Get the internal cpp variables of the StreamingAdd node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __is_power_of_two(self, value) -> bool:
        """Check if a value is a power of two."""
        return value > 0 and float(np.log2(value)).is_integer()

    def __ratio_to_fraction(self, ratio: float) -> tuple[int, int]:
        frac = Fraction(float(ratio)).limit_denominator(1 << 20)
        return int(frac.numerator), int(frac.denominator)

    def __quant_range(self, tensor_quant: QuantizedTensorType) -> tuple[int, int]:
        if tensor_quant.signed:
            return -(1 << (tensor_quant.bitwidth - 1)), (1 << (tensor_quant.bitwidth - 1)) - 1
        return 0, (1 << tensor_quant.bitwidth) - 1

    def __max_centered_magnitude(self, tensor_quant: QuantizedTensorType) -> int:
        qmin, qmax = self.__quant_range(tensor_quant)
        return max(abs(qmin - tensor_quant.zeropt), abs(qmax - tensor_quant.zeropt))

    def __aligned_bound(self, scale: float, acc_scale: float, magnitude: int) -> int:
        ratio = float(scale) / float(acc_scale)
        return int(np.ceil(abs(ratio) * magnitude + 0.5))

    def __signed_range_bits(self, bound: int) -> int:
        bound = max(1, int(bound))
        return int(np.ceil(np.log2(bound + 1))) + 1

    def __get_accumulator_quant(self, input_quantA, input_quantB) -> QuantizedTensorType:
        acc_scale = min(input_quantA.scale, input_quantB.scale)
        bound_a = self.__aligned_bound(
            input_quantA.scale,
            acc_scale,
            self.__max_centered_magnitude(input_quantA),
        )
        bound_b = self.__aligned_bound(
            input_quantB.scale,
            acc_scale,
            self.__max_centered_magnitude(input_quantB),
        )
        signed = (
            bool(input_quantA.signed)
            or bool(input_quantB.signed)
            or int(input_quantA.zeropt) != 0
            or int(input_quantB.zeropt) != 0
        )
        acc_bits = self.__signed_range_bits(bound_a + bound_b + 1)
        return QuantizedTensorType(
            bitwidth=acc_bits,
            signed=signed,
            scale=acc_scale,
            zeropt=0,
        )

    def __get_accumulator(self, input_quantA, input_quantB):
        """ Returns the accumulator type for the Add operation. """
        return self.__get_accumulator_quant(input_quantA, input_quantB).get_hls_data_type()

    def __get_input_aligner(self, input_quant, acc_quant) -> str:
        ratio = float(input_quant.scale) / float(acc_quant.scale)
        input_type = input_quant.get_hls_data_type()
        acc_type = acc_quant.get_hls_data_type()
        zeropt = int(input_quant.zeropt)

        if np.isclose(ratio, 1.0, atol=1e-12):
            if zeropt == 0:
                return f"DequantQuantEqual<{input_type}>"
            return f"DequantAlignEqual<{zeropt}, {input_type}, {acc_type}>"

        if self.__is_power_of_two(ratio):
            shift = -1 * int(np.log2(ratio))
            if zeropt == 0:
                return f"DequantQuantPo2<{shift}, {input_type}, {acc_type}>"
            return f"DequantAlignPo2<{shift}, {zeropt}, {input_type}, {acc_type}>"

        num, den = self.__ratio_to_fraction(ratio)
        if zeropt == 0:
            shift_form = ratio_to_shift_form(ratio)
            if shift_form is not None:
                n, s = shift_form
                return f"DequantQuantFloatShift<{n}, {s}, {input_type}, {acc_type}>"
            return f"DequantQuantFloat<{num}, {den}, {input_type}, {acc_type}>"
        return f"DequantAlignFloat<{num}, {den}, {zeropt}, {input_type}, {acc_type}>"

    def __get_aligners(self, input_quantA, input_quantB) -> tuple[str, str]:
        acc_quant = self.__get_accumulator_quant(input_quantA, input_quantB)
        return (
            self.__get_input_aligner(input_quantA, acc_quant),
            self.__get_input_aligner(input_quantB, acc_quant),
        )

    def __get_quantizer(self, input_quantA, input_quantB, output_type) -> str:
        """ Returns the quantizer type for the Add operation. """
        acc_quant = self.__get_accumulator_quant(input_quantA, input_quantB)
        ratio = float(acc_quant.scale) / float(output_type.scale)
        accumulator = acc_quant.get_hls_data_type()
        hls_output_type = output_type.get_hls_data_type()
        output_zeropt = int(np.asarray(output_type.zeropt).reshape(()).item())
        activation = self.get_nodeattr("activation")

        if self.__is_power_of_two(ratio):
            shift = -1 * int(np.log2(ratio))
            if activation == "NoOp":
                return f"DequantQuantPo2<{shift}, {accumulator}, {hls_output_type}, {output_zeropt}>"
            if activation == "ReLU":
                return f"ReLUQuantPo2<{shift}, {accumulator}, {hls_output_type}, {output_zeropt}>"
            if activation == "LeakyReLU":
                alpha_num = self.get_nodeattr("activation_alpha_num")
                alpha_den = self.get_nodeattr("activation_alpha_den")
                if self.__is_power_of_two(alpha_den):
                    alpha_den_shift = int(np.log2(alpha_den))
                    return f"LeakyReLUQuantPo2<{shift}, {alpha_num}, {alpha_den_shift}, {accumulator}, {hls_output_type}, {output_zeropt}>"
                raise ValueError(
                    "Fused LeakyReLU currently requires a power-of-two alpha denominator."
                )
            raise ValueError(
                f"Unsupported activation function '{activation}' for StreamingAdd."
            )

        num, den = self.__ratio_to_fraction(ratio)
        if activation == "NoOp":
            shift_form = ratio_to_shift_form(ratio)
            if shift_form is not None:
                n, s = shift_form
                return f"DequantQuantFloatShift<{n}, {s}, {accumulator}, {hls_output_type}, {output_zeropt}>"
            return f"DequantQuantFloat<{num}, {den}, {accumulator}, {hls_output_type}, {output_zeropt}>"
        if activation == "ReLU":
            return f"ReLUQuantFloat<{num}, {den}, {accumulator}, {hls_output_type}, {output_zeropt}>"
        if activation == "LeakyReLU":
            raise ValueError(
                "Fused LeakyReLU currently requires power-of-two quantization scales in StreamingAdd."
            )
        raise ValueError(
            f"Unsupported activation function '{activation}' for StreamingAdd."
        )

    def __get_object_declaration(self, model) -> cpp_object:

        input_typeA = require_tensor_type(model, self.onnx_node.input[0])
        input_typeB = require_tensor_type(model, self.onnx_node.input[1])
        output_type = require_tensor_type(model, self.onnx_node.output[0])

        input_shapeA_layout = require_tensor_layout(model, self.onnx_node.input[0])
        input_shapeB_layout = require_tensor_layout(model, self.onnx_node.input[1])
        if input_shapeA_layout != input_shapeB_layout:
            raise ValueError(
                f"Input layouts for StreamingAdd must be the same. Got {input_shapeA_layout} and {input_shapeB_layout}."
            )
        input_shapeA = self.require_4d_input_shape(model, 0, input_layout=input_shapeA_layout)

        StreamingAdd = cpp_object(
            "StreamingAdd",
            f"{self.onnx_node.name}",
            template_args=[
                (
                    f"{get_word_type(input_typeA, self.get_nodeattr('in_word_array'))}",
                    f"TInputWordA",
                ),
                (
                    f"{input_typeA.get_hls_data_type()}",
                    f"TInputA",
                ),
                (
                    f"{get_word_type(input_typeB, self.get_nodeattr('in_word_array'))}",
                    f"TInputWordB",
                ),
                (
                    f"{input_typeB.get_hls_data_type()}",
                    f"TInputB",
                ),
                (
                    f"{get_word_type(output_type, self.get_nodeattr('out_word_array'))}",
                    f"TOutputWord",
                ),
                (
                    f"{output_type.get_hls_data_type()}",
                    f"TOutput",
                ),
                (
                    f"{self.__get_accumulator(input_typeA, input_typeB)}",
                    f"TAcc",
                ),
                (
                    f"{self.__get_quantizer(input_typeA, input_typeB, output_type)}",
                    f"OutputTransform",
                ),
                (f"{self.__get_aligners(input_typeA, input_typeB)[0]}", "AlignA"),
                (f"{self.__get_aligners(input_typeA, input_typeB)[1]}", "AlignB"),
                (f"{input_shapeA[1]}", "DIM0"),
                (f"{input_shapeA[2]}", "DIM1"),
                (f"{input_shapeA[3]}", "DIM2"),
                (f"{self.get_nodeattr('dim1_unroll')}", "DIM1_UNROLL"),
                (f"{self.get_nodeattr('dim2_unroll')}", "DIM2_UNROLL"),
            ]
        )

        return StreamingAdd.generate_declaration()

    def __get_run_call(self, hls_tag: int) -> str:

        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(
                (
                    f"i_dataA",
                    f"hls::stream<TInputWordA>", 
                ),
                (
                    f"i_dataB",
                    f"hls::stream<TInputWordB>", 
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
                    f"i_dataA",
                    f"hls::stream<TInputWordA>", 
                ),
                (
                    f"i_dataB",
                    f"hls::stream<TInputWordB>", 
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

    def accepted_input_layout(self) -> tuple | None:
        """ StreamingAdd is layout-agnostic, so it accepts any input layout. """
        return None

    def produced_output_layout(self, input_layout: tuple | None) -> tuple | None:
        """ StreamingAdd is layout-agnostic, so it produces the same layout as input. """
        return input_layout

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int) -> None:
        """Lower the node to HLS code."""

        output_type = require_tensor_type(model, self.onnx_node.output[0])

        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(self.get_nodeattr("in_stream_array"))
        ]
        input_names.extend(
            [
                f"{self.__get_stream_name(self.onnx_node.input[1])}_{i}_"
                for i in range(self.get_nodeattr("in_stream_array"))
            ]
        )

        output_names = [
            f"{self.__get_stream_name(self.onnx_node.output[0])}_{i}_"
            for i in range(self.get_nodeattr("out_stream_array"))
        ]

        tensors_fifo_metadata = {}
        for output in output_names:
            tensors_fifo_metadata[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_word_type(output_type, self.get_nodeattr('out_word_array'))}",
                n_array=self.get_nodeattr("out_stream_array"),
            )

        hls_kernel = HLSKernel.make_node(
            inputs=input_names,
            outputs=output_names,
            name=f"{self.onnx_node.name}_hls",
            domain="nn2fpga.compiler.custom_op",
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

    def get_latency(self, model: ModelWrapper) -> int:
        """ Estimate the latency of the StreamingAdd operation.
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
        """ Estimate the BRAM usage of the StreamingAdd operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated BRAM usage.
        """
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        """ Estimate the DSP usage of the StreamingAdd operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated DSP usage.
        """
        dim1_unroll = self.get_nodeattr("dim1_unroll")
        dim2_unroll = self.get_nodeattr("dim2_unroll")
        act_dsp = 0
        if self.get_nodeattr("activation") == "LeakyReLU":
            act_dsp = dim1_unroll * dim2_unroll

        return act_dsp

    def has_linebuffer(self) -> bool:
        """ Check if the StreamingAdd operation requires a line buffer.
        Returns:
            bool: True if Line Buffering is required, False otherwise.
        """
        return False

    def can_inherit_interface(self):
        return True

    def inherit_interface(self, model: ModelWrapper, upstream: NodeInterface) -> None:
        """ Inherit the interface from the upstream node."""
        self.set_nodeattr("in_stream_array", upstream.out_stream_array)
        self.set_nodeattr("out_stream_array", upstream.out_stream_array)
        self.set_nodeattr("in_word_array", upstream.out_word_array)
        self.set_nodeattr("out_word_array", upstream.out_word_array)

        self.set_nodeattr("dim2_unroll", upstream.out_word_array)
        self.set_nodeattr("dim1_unroll", upstream.out_stream_array)
