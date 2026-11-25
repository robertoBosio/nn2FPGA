from fractions import Fraction

import numpy as np
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from onnx_ir import convenience as ir_convenience
from onnxscript.rewriter import pattern

from nn2fpga.compiler.core.tensor_fifo import TensorFifo
from nn2fpga.compiler.core.tensor_layout import require_tensor_layout
from nn2fpga.compiler.core.tensor_type import require_tensor_type
from nn2fpga.compiler.custom_op.hlskernel import HLSKernel
from nn2fpga.compiler.custom_op.op_base import NN2FPGAOp, NodeInterface
from nn2fpga.compiler.custom_op.register_rewrite_rule import PRule, register_rules
from nn2fpga.compiler.utils.codegen_utils import cpp_function, cpp_object, get_word_type
from nn2fpga.compiler.utils.linebuffer_utils import circular_linebuffer_latency, normalize_pads
from nn2fpga.compiler.utils.scale_ratio import ratio_to_shift_form


def _constant_values(value) -> list | None:
    tensor = ir_convenience.get_const_tensor(value)
    if tensor is None:
        return None
    return tensor.numpy().reshape(-1).tolist()


def _attribute_value(value):
    return getattr(value, "value", value)


class StreamingWindowToChannels(NN2FPGAOp):
    """Flatten spatial windows into the output channel dimension."""

    def get_nodeattr_types(self):
        return {
            "kernel_shape": ("ints", True, [1, 1]),
            "pads": ("ints", True, [0, 0]),
            "strides": ("ints", True, [1, 1]),
            "dim2_unroll": ("i", False, 1),
            "dim1_unroll": ("i", False, 1),
            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),
        }

    @staticmethod
    def focus_pattern(
        op,
        x,
        h_even_starts,
        h_even_ends,
        h_even_axes,
        h_even_steps,
        h_odd_starts,
        h_odd_ends,
        h_odd_axes,
        h_odd_steps,
        ee_starts,
        ee_ends,
        ee_axes,
        ee_steps,
        eo_starts,
        eo_ends,
        eo_axes,
        eo_steps,
        oe_starts,
        oe_ends,
        oe_axes,
        oe_steps,
        oo_starts,
        oo_ends,
        oo_axes,
        oo_steps,
        qh0_scale,
        qh0_zeropt,
        qh0_bitwidth,
        qh0_signed,
        qh0_narrow,
        qh0_rounding_mode,
        qh1_scale,
        qh1_zeropt,
        qh1_bitwidth,
        qh1_signed,
        qh1_narrow,
        qh1_rounding_mode,
        q0_scale,
        q0_zeropt,
        q0_bitwidth,
        q0_signed,
        q0_narrow,
        q0_rounding_mode,
        q1_scale,
        q1_zeropt,
        q1_bitwidth,
        q1_signed,
        q1_narrow,
        q1_rounding_mode,
        q2_scale,
        q2_zeropt,
        q2_bitwidth,
        q2_signed,
        q2_narrow,
        q2_rounding_mode,
        q3_scale,
        q3_zeropt,
        q3_bitwidth,
        q3_signed,
        q3_narrow,
        q3_rounding_mode,
        q01_scale,
        q01_zeropt,
        q01_bitwidth,
        q01_signed,
        q01_narrow,
        q01_rounding_mode,
        q012_scale,
        q012_zeropt,
        q012_bitwidth,
        q012_signed,
        q012_narrow,
        q012_rounding_mode,
        axis0,
        axis1,
        axis2,
    ):
        h_even = op.Slice(x, h_even_starts, h_even_ends, h_even_axes, h_even_steps)
        h_odd = op.Slice(x, h_odd_starts, h_odd_ends, h_odd_axes, h_odd_steps)
        q_h_even = op.Quant(
            h_even, qh0_scale, qh0_zeropt, qh0_bitwidth, signed=qh0_signed,
            narrow=qh0_narrow, rounding_mode=qh0_rounding_mode,
            _allow_other_attributes=True, _domain="qonnx.custom_op.general",
        )
        q_h_odd = op.Quant(
            h_odd, qh1_scale, qh1_zeropt, qh1_bitwidth, signed=qh1_signed,
            narrow=qh1_narrow, rounding_mode=qh1_rounding_mode,
            _allow_other_attributes=True, _domain="qonnx.custom_op.general",
        )
        ee = op.Slice(q_h_even, ee_starts, ee_ends, ee_axes, ee_steps)
        eo = op.Slice(q_h_even, eo_starts, eo_ends, eo_axes, eo_steps)
        oe = op.Slice(q_h_odd, oe_starts, oe_ends, oe_axes, oe_steps)
        oo = op.Slice(q_h_odd, oo_starts, oo_ends, oo_axes, oo_steps)

        q_ee = op.Quant(
            ee, q0_scale, q0_zeropt, q0_bitwidth, signed=q0_signed,
            narrow=q0_narrow, rounding_mode=q0_rounding_mode,
            _allow_other_attributes=True, _domain="qonnx.custom_op.general",
        )
        q_eo = op.Quant(
            eo, q1_scale, q1_zeropt, q1_bitwidth, signed=q1_signed,
            narrow=q1_narrow, rounding_mode=q1_rounding_mode,
            _allow_other_attributes=True, _domain="qonnx.custom_op.general",
        )
        q_oe = op.Quant(
            oe, q2_scale, q2_zeropt, q2_bitwidth, signed=q2_signed,
            narrow=q2_narrow, rounding_mode=q2_rounding_mode,
            _allow_other_attributes=True, _domain="qonnx.custom_op.general",
        )
        q_oo = op.Quant(
            oo, q3_scale, q3_zeropt, q3_bitwidth, signed=q3_signed,
            narrow=q3_narrow, rounding_mode=q3_rounding_mode,
            _allow_other_attributes=True, _domain="qonnx.custom_op.general",
        )

        concat01 = op.Concat(q_ee, q_oe, axis=axis0)
        q_concat01 = op.Quant(
            concat01, q01_scale, q01_zeropt, q01_bitwidth, signed=q01_signed,
            narrow=q01_narrow, rounding_mode=q01_rounding_mode,
            _allow_other_attributes=True, _domain="qonnx.custom_op.general",
        )
        concat012 = op.Concat(q_concat01, q_eo, axis=axis1)
        q_concat012 = op.Quant(
            concat012, q012_scale, q012_zeropt, q012_bitwidth, signed=q012_signed,
            narrow=q012_narrow, rounding_mode=q012_rounding_mode,
            _allow_other_attributes=True, _domain="qonnx.custom_op.general",
        )
        return op.Concat(q_concat012, q_oo, axis=axis2)

    @staticmethod
    def _condition(
        context,
        h_even_starts, h_even_ends, h_even_axes, h_even_steps,
        h_odd_starts, h_odd_ends, h_odd_axes, h_odd_steps,
        ee_starts, ee_ends, ee_axes, ee_steps,
        eo_starts, eo_ends, eo_axes, eo_steps,
        oe_starts, oe_ends, oe_axes, oe_steps,
        oo_starts, oo_ends, oo_axes, oo_steps,
        qh0_scale, qh0_zeropt, qh0_bitwidth, qh0_signed, qh0_narrow, qh0_rounding_mode,
        qh1_scale, qh1_zeropt, qh1_bitwidth, qh1_signed, qh1_narrow, qh1_rounding_mode,
        q0_scale, q0_zeropt, q0_bitwidth, q0_signed, q0_narrow, q0_rounding_mode,
        q1_scale, q1_zeropt, q1_bitwidth, q1_signed, q1_narrow, q1_rounding_mode,
        q2_scale, q2_zeropt, q2_bitwidth, q2_signed, q2_narrow, q2_rounding_mode,
        q3_scale, q3_zeropt, q3_bitwidth, q3_signed, q3_narrow, q3_rounding_mode,
        q01_scale, q01_zeropt, q01_bitwidth, q01_signed, q01_narrow, q01_rounding_mode,
        q012_scale, q012_zeropt, q012_bitwidth, q012_signed, q012_narrow, q012_rounding_mode,
        axis0, axis1, axis2,
        **_,
    ):
        end_sentinels = {2147483647, 9223372036854775807}
        slices = (
            (h_even_starts, h_even_ends, h_even_axes, h_even_steps, 0, 2),
            (h_odd_starts, h_odd_ends, h_odd_axes, h_odd_steps, 1, 2),
            (ee_starts, ee_ends, ee_axes, ee_steps, 0, 3),
            (eo_starts, eo_ends, eo_axes, eo_steps, 1, 3),
            (oe_starts, oe_ends, oe_axes, oe_steps, 0, 3),
            (oo_starts, oo_ends, oo_axes, oo_steps, 1, 3),
        )
        for starts, ends, axes, steps, start, axis in slices:
            end_values = _constant_values(ends)
            if (
                _constant_values(starts) != [start]
                or end_values is None
                or len(end_values) != 1
                or end_values[0] not in end_sentinels
                or _constant_values(axes) != [axis]
                or _constant_values(steps) != [2]
            ):
                return False

        quantizers = (
            (qh0_scale, qh0_zeropt, qh0_bitwidth, qh0_signed, qh0_narrow, qh0_rounding_mode),
            (qh1_scale, qh1_zeropt, qh1_bitwidth, qh1_signed, qh1_narrow, qh1_rounding_mode),
            (q0_scale, q0_zeropt, q0_bitwidth, q0_signed, q0_narrow, q0_rounding_mode),
            (q1_scale, q1_zeropt, q1_bitwidth, q1_signed, q1_narrow, q1_rounding_mode),
            (q2_scale, q2_zeropt, q2_bitwidth, q2_signed, q2_narrow, q2_rounding_mode),
            (q3_scale, q3_zeropt, q3_bitwidth, q3_signed, q3_narrow, q3_rounding_mode),
            (q01_scale, q01_zeropt, q01_bitwidth, q01_signed, q01_narrow, q01_rounding_mode),
            (q012_scale, q012_zeropt, q012_bitwidth, q012_signed, q012_narrow, q012_rounding_mode),
        )
        reference = quantizers[0]
        if any(_constant_values(value) is None for value in reference[:3]):
            return False
        for quantizer in quantizers[1:]:
            if any(_constant_values(value) is None for value in quantizer[:3]):
                return False
            if any(
                _constant_values(value) != _constant_values(reference_value)
                for value, reference_value in zip(quantizer[:3], reference[:3])
            ):
                return False
            if any(
                _attribute_value(value) != _attribute_value(reference_value)
                for value, reference_value in zip(quantizer[3:], reference[3:])
            ):
                return False

        return all(_attribute_value(axis) == 1 for axis in (axis0, axis1, axis2))

    @staticmethod
    def rewrite(op, x, **_):
        return op.StreamingWindowToChannels(
            x,
            kernel_shape=[2, 2],
            pads=[0, 0],
            strides=[2, 2],
            _domain="nn2fpga.compiler.custom_op",
        )

    @register_rules
    def register_rules():
        return [
            PRule(
                pattern.RewriteRule(
                    StreamingWindowToChannels.focus_pattern,
                    StreamingWindowToChannels.rewrite,
                    StreamingWindowToChannels._condition,
                ),
                priority=1,
            )
        ]

    def make_shape_compatible_op(self, model):
        kernel_shape = self.get_nodeattr("kernel_shape")
        strides = self.get_nodeattr("strides")
        pads = normalize_pads(self.get_nodeattr("pads"))
        if (
            kernel_shape[0] != kernel_shape[1]
            or list(strides) != list(kernel_shape)
            or any(pads)
        ):
            raise ValueError(
                "StreamingWindowToChannels shape inference currently requires "
                "a square, unpadded window with stride equal to its kernel"
            )

        return helper.make_node(
            "SpaceToDepth",
            inputs=self.onnx_node.input,
            outputs=self.onnx_node.output,
            name=f"{self.onnx_node.name}_shape_compatible",
            blocksize=kernel_shape[0],
            mode="CRD",
        )

    def infer_node_datatype(self, model):
        model.set_tensor_datatype(
            self.onnx_node.output[0],
            model.get_tensor_datatype(self.onnx_node.input[0]),
        )

    def execute_node(self, context, graph):
        pass

    def verify_node(self):
        pass

    def accepted_input_layout(self) -> tuple | None:
        return (0, 2, 3, 1)

    def produced_output_layout(self, input_layout: tuple | None) -> tuple | None:
        return (0, 2, 3, 1)

    def __get_stream_name(self, name: str) -> str:
        return f"{name}_stream"

    def __get_quantizer(self, input_type, output_type) -> str:
        ratio = float(input_type.scale) / float(output_type.scale)
        input_hls_type = input_type.get_hls_data_type()
        output_hls_type = output_type.get_hls_data_type()
        input_zeropt = int(input_type.zeropt)
        output_zeropt = int(output_type.zeropt)

        if np.isclose(ratio, 1.0, atol=1e-12):
            if input_hls_type == output_hls_type and input_zeropt == output_zeropt:
                return f"DequantQuantEqual<{output_hls_type}>"
            return (
                f"RequantizeEqual<{input_zeropt}, {output_zeropt}, "
                f"{input_hls_type}, {output_hls_type}>"
            )

        if ratio > 0 and float(np.log2(ratio)).is_integer():
            shift = -int(np.log2(ratio))
            if input_zeropt == 0:
                return (
                    f"DequantQuantPo2<{shift}, {input_hls_type}, {output_hls_type}, "
                    f"{output_zeropt}>"
                )
            return (
                f"RequantizePo2<{shift}, {input_zeropt}, {output_zeropt}, "
                f"{input_hls_type}, {output_hls_type}>"
            )

        if input_zeropt == 0:
            shift_form = ratio_to_shift_form(ratio)
            if shift_form is not None:
                numerator, shift = shift_form
                return (
                    f"DequantQuantFloatShift<{numerator}, {shift}, "
                    f"{input_hls_type}, {output_hls_type}, {output_zeropt}>"
                )
            numerator, denominator = Fraction(ratio).limit_denominator(1 << 20).as_integer_ratio()
            return (
                f"DequantQuantFloat<{numerator}, {denominator}, "
                f"{input_hls_type}, {output_hls_type}, {output_zeropt}>"
            )

        numerator, denominator = Fraction(ratio).limit_denominator(1 << 20).as_integer_ratio()
        return (
            f"RequantizeFloat<{numerator}, {denominator}, {input_zeropt}, "
            f"{output_zeropt}, {input_hls_type}, {output_hls_type}>"
        )

    def __get_object_declaration(self, model: ModelWrapper) -> str:
        input_type = require_tensor_type(model, self.onnx_node.input[0])
        output_type = require_tensor_type(model, self.onnx_node.output[0])
        output_layout = require_tensor_layout(model, self.onnx_node.output[0])
        output_shape = self.require_4d_output_shape(model, 0, output_layout)
        window_to_channels = cpp_object(
            "StreamingWindowToChannels",
            self.onnx_node.name,
            template_args=[
                (get_word_type(input_type, self.get_nodeattr("in_word_array")), "TInputWord"),
                (input_type.get_hls_data_type(), "TInput"),
                (get_word_type(output_type, self.get_nodeattr("out_word_array")), "TOutputWord"),
                (output_type.get_hls_data_type(), "TOutput"),
                (self.__get_quantizer(input_type, output_type), "Quantizer"),
                (output_shape[-3], "OUT_DIM0"),
                (output_shape[-2], "OUT_DIM1"),
                (output_shape[-1], "OUT_DIM2"),
                (self.get_nodeattr("kernel_shape")[0], "FH"),
                (self.get_nodeattr("kernel_shape")[1], "FW"),
                (self.get_nodeattr("strides")[1], "STRIDE_W"),
                (self.get_nodeattr("dim1_unroll"), "DIM1_UNROLL"),
                (self.get_nodeattr("dim2_unroll"), "DIM2_UNROLL"),
            ],
        )
        return window_to_channels.generate_declaration()

    def __get_run_call(self, hls_tag: int) -> str:
        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(
                ("i_data", "hls::stream<TInputWord>"),
                ("o_data", "hls::stream<TOutputWord>"),
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
            return_type="ActorStatus",
            arguments=(
                ("i_data", "hls::stream<TInputWord>"),
                ("o_data", "hls::stream<TOutputWord>"),
            ),
        )
        return step.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int):
        filter_height, filter_width = self.get_nodeattr("kernel_shape")
        stride_width = self.get_nodeattr("strides")[1]
        dim1_unroll = self.get_nodeattr("dim1_unroll")
        filter_width_extended = filter_width + (dim1_unroll - 1) * stride_width
        output_type = require_tensor_type(model, self.onnx_node.output[0])

        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{index}_"
            for index in range(filter_height * filter_width_extended)
        ]
        output_names = [
            f"{self.__get_stream_name(self.onnx_node.output[0])}_{index}_"
            for index in range(self.get_nodeattr("out_stream_array"))
        ]
        fifos = {
            output: TensorFifo(
                depth=0,
                hls_type=get_word_type(output_type, self.get_nodeattr("out_word_array")),
                n_array=self.get_nodeattr("out_stream_array"),
            )
            for output in output_names
        }

        hls_kernel = HLSKernel.make_node(
            inputs=input_names,
            outputs=output_names,
            name=f"{self.onnx_node.name}_hls",
            domain="nn2fpga.compiler.custom_op",
            original_op_type="StreamingWindowToChannels",
            hls_tag=hls_tag,
            hls_object_name=self.onnx_node.name,
            hls_variable_declarations="",
            hls_run_call=self.__get_run_call(hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )
        return [hls_kernel], [], fifos, hls_tag + 1

    def get_latency(self, model: ModelWrapper) -> int:
        input_layout = require_tensor_layout(model, self.onnx_node.input[0])
        output_layout = require_tensor_layout(model, self.onnx_node.output[0])
        input_shape = self.require_4d_input_shape(model, 0, input_layout)
        output_shape = self.require_4d_output_shape(model, 0, output_layout)
        operator_latency = int(np.prod(output_shape)) // (
            self.get_nodeattr("dim1_unroll") * self.get_nodeattr("dim2_unroll")
        )
        if not self.has_linebuffer():
            return operator_latency
        linebuffer_latency = circular_linebuffer_latency(
            input_shape,
            self.get_nodeattr("pads"),
            self.get_nodeattr("dim1_unroll"),
            self.get_nodeattr("dim2_unroll"),
        )
        return max(operator_latency, linebuffer_latency)

    def get_brams(self, model: ModelWrapper) -> int:
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        return 0

    def can_inherit_interface(self):
        return True

    def inherit_interface(self, model: ModelWrapper, upstream: NodeInterface) -> None:
        self.set_nodeattr("in_stream_array", upstream.out_stream_array)
        self.set_nodeattr("out_stream_array", upstream.out_stream_array)
        self.set_nodeattr("in_word_array", upstream.out_word_array)
        self.set_nodeattr("out_word_array", upstream.out_word_array)
        self.set_nodeattr("dim1_unroll", upstream.out_stream_array)
        self.set_nodeattr("dim2_unroll", upstream.out_word_array)

    def has_linebuffer(self) -> bool:
        kernel_shape = self.get_nodeattr("kernel_shape")
        strides = self.get_nodeattr("strides")
        pads = normalize_pads(self.get_nodeattr("pads"))
        return not (
            all(kernel == 1 for kernel in kernel_shape)
            and all(stride == 1 for stride in strides)
            and self.get_nodeattr("dim1_unroll") == 1
            and not any(pads)
        )
