from attr import dataclass

import numpy as np
import onnxruntime as rt
from onnx import TensorProto, helper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import get_custom_tensor_datatype
from backend.core.tensor_fifo import TensorFifo
from backend.custom_op.hlskernel import HLSKernel
from backend.custom_op.op_base import DSECapable, NN2FPGAOp, NodeInterface
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
from backend.custom_op.register_rewrite_rule import register_rules, PRule
from onnxscript import ir
from onnx_ir import convenience as ir_convenience
from onnxscript.rewriter import pattern
from typing import Optional, Sequence
import logging

logger = logging.getLogger(__name__)

EXP_PRECISION = 12  # Number of bits for LUT output (Q0.12 format for max precision)
DIV_PRECISION = 32 # Number of bits for division result (Q0.32 format for max precision)

def _as_list(x):
    if x is None:
        return None
    arr = np.asarray(x)
    return [int(v) for v in arr.flatten().tolist()]


def _get_const_i64_list(x) -> Optional[list[int]]:
    t = ir_convenience.get_const_tensor(x)
    if t is None:
        return None
    return _as_list(t)


def _get_const_scalar(x):
    t = ir_convenience.get_const_tensor(x)
    if t is None:
        return None
    arr = np.asarray(t)
    if arr.size != 1:
        return None
    return arr.reshape(-1)[0].item()


def _check_quant_node_attrs(scale, zeropt, bitwidth, signed, narrow, rounding_mode):
    return (
        ir_convenience.get_const_tensor(scale) is not None
        and ir_convenience.get_const_tensor(zeropt) is not None
        and ir_convenience.get_const_tensor(bitwidth) is not None
    )


class StreamingYoloAttention(NN2FPGAOp, DSECapable):
    @staticmethod
    def yolo_attention_pattern(
        op,
        x,

        # input quant before reshape
        # in_scale, in_zeropt, in_bitwidth, in_signed, in_narrow, in_rounding_mode,

        # reshape
        shape_in,

        # quant after reshape
        rq_scale, rq_zeropt, rq_bitwidth, rq_signed, rq_narrow, rq_rounding_mode,

        # split tree constants: first Q | KV, then K | V
        split0,   # expected [32, 96]
        split1,   # expected [32, 64]
        axis0,
        axis1,

        # Q quant
        q_scale, q_zeropt, q_bitwidth, q_signed, q_narrow, q_rounding_mode,

        # K quant
        k_scale, k_zeropt, k_bitwidth, k_signed, k_narrow, k_rounding_mode,

        # V quant
        v_scale, v_zeropt, v_bitwidth, v_signed, v_narrow, v_rounding_mode,

        # Q transpose quant
        qt_scale, qt_zeropt, qt_bitwidth, qt_signed, qt_narrow, qt_rounding_mode,

        # QK quant
        qk_scale, qk_zeropt, qk_bitwidth, qk_signed, qk_narrow, qk_rounding_mode,

        # constant branch as Quant(const)
        const_value,
        const_scale,
        const_zeropt,
        const_bitwidth,
        const_signed,
        const_narrow,
        const_rounding_mode,

        # post-mul quant
        qks_scale, qks_zeropt, qks_bitwidth, qks_signed, qks_narrow, qks_rounding_mode,

        # softmax quant
        p_scale, p_zeropt, p_bitwidth, p_signed, p_narrow, p_rounding_mode,

        # P transpose quant
        pt_scale, pt_zeropt, pt_bitwidth, pt_signed, pt_narrow, pt_rounding_mode,

        # VP quant
        vp_scale, vp_zeropt, vp_bitwidth, vp_signed, vp_narrow, vp_rounding_mode,

        # output reshapes
        shape_out_y,
        shape_out_v,

        # final Y quant
        # y_scale, y_zeropt, y_bitwidth, y_signed, y_narrow, y_rounding_mode,

        # final V-out quant
        # vo_scale, vo_zeropt, vo_bitwidth, vo_signed, vo_narrow, vo_rounding_mode,
    ):
        # x_q = op.Quant(
        #     x,
        #     in_scale,
        #     in_zeropt,
        #     in_bitwidth,
        #     signed=in_signed,
        #     narrow=in_narrow,
        #     rounding_mode=in_rounding_mode,
        #     _allow_other_attributes=True,
        #     _domain="qonnx.custom_op.general",
        # )

        x_r = op.Reshape(x, shape_in)
        x_rq = op.Quant(
            x_r,
            rq_scale,
            rq_zeropt,
            rq_bitwidth,
            signed=rq_signed,
            narrow=rq_narrow,
            rounding_mode=rq_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )

        q, kv = op.Split(x_rq, split0, axis=axis0, _outputs=2)
        k, v = op.Split(kv, split1, axis=axis1, _outputs=2)

        q_q = op.Quant(
            q,
            q_scale,
            q_zeropt,
            q_bitwidth,
            signed=q_signed,
            narrow=q_narrow,
            rounding_mode=q_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )
        k_q = op.Quant(
            k,
            k_scale,
            k_zeropt,
            k_bitwidth,
            signed=k_signed,
            narrow=k_narrow,
            rounding_mode=k_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )
        v_q = op.Quant(
            v,
            v_scale,
            v_zeropt,
            v_bitwidth,
            signed=v_signed,
            narrow=v_narrow,
            rounding_mode=v_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )

        q_t = op.Transpose(q_q, perm=[0, 1, 3, 2])
        q_tq = op.Quant(
            q_t,
            qt_scale,
            qt_zeropt,
            qt_bitwidth,
            signed=qt_signed,
            narrow=qt_narrow,
            rounding_mode=qt_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )

        qk = op.MatMul(q_tq, k_q)
        qk_q = op.Quant(
            qk,
            qk_scale,
            qk_zeropt,
            qk_bitwidth,
            signed=qk_signed,
            narrow=qk_narrow,
            rounding_mode=qk_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )

        const_q = op.Quant(
            const_value,
            const_scale,
            const_zeropt,
            const_bitwidth,
            signed=const_signed,
            narrow=const_narrow,
            rounding_mode=const_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )

        qks = op.Mul(qk_q, const_q)
        qks_q = op.Quant(
            qks,
            qks_scale,
            qks_zeropt,
            qks_bitwidth,
            signed=qks_signed,
            narrow=qks_narrow,
            rounding_mode=qks_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )

        p = op.Softmax(qks_q)
        p_q = op.Quant(
            p,
            p_scale,
            p_zeropt,
            p_bitwidth,
            signed=p_signed,
            narrow=p_narrow,
            rounding_mode=p_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )

        p_t = op.Transpose(p_q, perm=[0, 1, 3, 2])
        p_tq = op.Quant(
            p_t,
            pt_scale,
            pt_zeropt,
            pt_bitwidth,
            signed=pt_signed,
            narrow=pt_narrow,
            rounding_mode=pt_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )

        y = op.MatMul(v_q, p_tq)
        y_q_pre = op.Quant(
            y,
            vp_scale,
            vp_zeropt,
            vp_bitwidth,
            signed=vp_signed,
            narrow=vp_narrow,
            rounding_mode=vp_rounding_mode,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )

        y_r = op.Reshape(y_q_pre, shape_out_y)
        # y_out = op.Quant(
        #     y_r,
        #     y_scale,
        #     y_zeropt,
        #     y_bitwidth,
        #     signed=y_signed,
        #     narrow=y_narrow,
        #     rounding_mode=y_rounding_mode,
        #     _allow_other_attributes=True,
        #     _domain="qonnx.custom_op.general",
        # )

        v_r = op.Reshape(v_q, shape_out_v)
        # v_out = op.Quant(
        #     v_r,
        #     vo_scale,
        #     vo_zeropt,
        #     vo_bitwidth,
        #     signed=vo_signed,
        #     narrow=vo_narrow,
        #     rounding_mode=vo_rounding_mode,
        #     _allow_other_attributes=True,
        #     _domain="qonnx.custom_op.general",
        # )

        return y_r, v_r

    @staticmethod
    def _condition(
        context,
        x,
        # in_scale, in_zeropt, in_bitwidth, in_signed, in_narrow, in_rounding_mode,
        shape_in,
        rq_scale, rq_zeropt, rq_bitwidth, rq_signed, rq_narrow, rq_rounding_mode,
        split0, split1, axis0, axis1,
        q_scale, q_zeropt, q_bitwidth, q_signed, q_narrow, q_rounding_mode,
        k_scale, k_zeropt, k_bitwidth, k_signed, k_narrow, k_rounding_mode,
        v_scale, v_zeropt, v_bitwidth, v_signed, v_narrow, v_rounding_mode,
        qt_scale, qt_zeropt, qt_bitwidth, qt_signed, qt_narrow, qt_rounding_mode,
        qk_scale, qk_zeropt, qk_bitwidth, qk_signed, qk_narrow, qk_rounding_mode,
        const_value, const_scale, const_zeropt, const_bitwidth, const_signed, const_narrow, const_rounding_mode,
        qks_scale, qks_zeropt, qks_bitwidth, qks_signed, qks_narrow, qks_rounding_mode,
        p_scale, p_zeropt, p_bitwidth, p_signed, p_narrow, p_rounding_mode,
        pt_scale, pt_zeropt, pt_bitwidth, pt_signed, pt_narrow, pt_rounding_mode,
        vp_scale, vp_zeropt, vp_bitwidth, vp_signed, vp_narrow, vp_rounding_mode,
        shape_out_y, shape_out_v,
        # y_scale, y_zeropt, y_bitwidth, y_signed, y_narrow, y_rounding_mode,
        # vo_scale, vo_zeropt, vo_bitwidth, vo_signed, vo_narrow, vo_rounding_mode,
        **_,
    ):
        if _get_const_scalar(const_value) is None:
            return False

        if _get_const_i64_list(shape_in) is None:
            return False
        if _get_const_i64_list(shape_out_y) is None:
            return False
        if _get_const_i64_list(shape_out_v) is None:
            return False

        if _get_const_i64_list(split0) != [32, 96]:
            return False
        if _get_const_i64_list(split1) != [32, 64]:
            return False

        a0 = int(axis0.value)
        a1 = int(axis1.value)
        if a0 != a1:
            return False
        if a0 not in (1, 2):
            return False

        quant_groups = [
            # (in_scale, in_zeropt, in_bitwidth, in_signed, in_narrow, in_rounding_mode),
            (rq_scale, rq_zeropt, rq_bitwidth, rq_signed, rq_narrow, rq_rounding_mode),
            (q_scale, q_zeropt, q_bitwidth, q_signed, q_narrow, q_rounding_mode),
            (k_scale, k_zeropt, k_bitwidth, k_signed, k_narrow, k_rounding_mode),
            (v_scale, v_zeropt, v_bitwidth, v_signed, v_narrow, v_rounding_mode),
            (qt_scale, qt_zeropt, qt_bitwidth, qt_signed, qt_narrow, qt_rounding_mode),
            (qk_scale, qk_zeropt, qk_bitwidth, qk_signed, qk_narrow, qk_rounding_mode),
            (const_scale, const_zeropt, const_bitwidth, const_signed, const_narrow, const_rounding_mode),
            (qks_scale, qks_zeropt, qks_bitwidth, qks_signed, qks_narrow, qks_rounding_mode),
            (p_scale, p_zeropt, p_bitwidth, p_signed, p_narrow, p_rounding_mode),
            (pt_scale, pt_zeropt, pt_bitwidth, pt_signed, pt_narrow, pt_rounding_mode),
            (vp_scale, vp_zeropt, vp_bitwidth, vp_signed, vp_narrow, vp_rounding_mode),
            # (y_scale, y_zeropt, y_bitwidth, y_signed, y_narrow, y_rounding_mode),
            # (vo_scale, vo_zeropt, vo_bitwidth, vo_signed, vo_narrow, vo_rounding_mode),
        ]
        if not all(_check_quant_node_attrs(*g) for g in quant_groups):
            return False

        return True

    @staticmethod
    def rewrite(
        op,
        x,
        # in_scale, in_zeropt, in_bitwidth, in_signed, in_narrow, in_rounding_mode,
        shape_in,
        rq_scale, rq_zeropt, rq_bitwidth, rq_signed, rq_narrow, rq_rounding_mode,
        split0, split1, axis0, axis1,
        q_scale, q_zeropt, q_bitwidth, q_signed, q_narrow, q_rounding_mode,
        k_scale, k_zeropt, k_bitwidth, k_signed, k_narrow, k_rounding_mode,
        v_scale, v_zeropt, v_bitwidth, v_signed, v_narrow, v_rounding_mode,
        qt_scale, qt_zeropt, qt_bitwidth, qt_signed, qt_narrow, qt_rounding_mode,
        qk_scale, qk_zeropt, qk_bitwidth, qk_signed, qk_narrow, qk_rounding_mode,
        const_value, const_scale, const_zeropt, const_bitwidth, const_signed, const_narrow, const_rounding_mode,
        qks_scale, qks_zeropt, qks_bitwidth, qks_signed, qks_narrow, qks_rounding_mode,
        p_scale, p_zeropt, p_bitwidth, p_signed, p_narrow, p_rounding_mode,
        pt_scale, pt_zeropt, pt_bitwidth, pt_signed, pt_narrow, pt_rounding_mode,
        vp_scale, vp_zeropt, vp_bitwidth, vp_signed, vp_narrow, vp_rounding_mode,
        shape_out_y, shape_out_v,
        # y_scale, y_zeropt, y_bitwidth, y_signed, y_narrow, y_rounding_mode,
        # vo_scale, vo_zeropt, vo_bitwidth, vo_signed, vo_narrow, vo_rounding_mode,
        **_,
    ):
        const_value_py = _get_const_scalar(const_value)

        y_out, v_out = op.StreamingYoloAttention(
            x,

            # in_scale, in_zeropt, in_bitwidth,
            rq_scale, rq_zeropt, rq_bitwidth,
            q_scale, q_zeropt, q_bitwidth,
            k_scale, k_zeropt, k_bitwidth,
            v_scale, v_zeropt, v_bitwidth,
            qt_scale, qt_zeropt, qt_bitwidth,
            qk_scale, qk_zeropt, qk_bitwidth,
            const_scale, const_zeropt, const_bitwidth,
            qks_scale, qks_zeropt, qks_bitwidth,
            p_scale, p_zeropt, p_bitwidth,
            pt_scale, pt_zeropt, pt_bitwidth,
            vp_scale, vp_zeropt, vp_bitwidth,
            # y_scale, y_zeropt, y_bitwidth,
            # vo_scale, vo_zeropt, vo_bitwidth,
            const_value=const_value_py,

            # in_signed=in_signed.value,
            # in_narrow=in_narrow.value,
            # in_rounding_mode=in_rounding_mode.value,

            rq_signed=rq_signed.value,
            rq_narrow=rq_narrow.value,
            rq_rounding_mode=rq_rounding_mode.value,

            q_signed=q_signed.value,
            q_narrow=q_narrow.value,
            q_rounding_mode=q_rounding_mode.value,

            k_signed=k_signed.value,
            k_narrow=k_narrow.value,
            k_rounding_mode=k_rounding_mode.value,

            v_signed=v_signed.value,
            v_narrow=v_narrow.value,
            v_rounding_mode=v_rounding_mode.value,

            qt_signed=qt_signed.value,
            qt_narrow=qt_narrow.value,
            qt_rounding_mode=qt_rounding_mode.value,

            qk_signed=qk_signed.value,
            qk_narrow=qk_narrow.value,
            qk_rounding_mode=qk_rounding_mode.value,

            const_signed=const_signed.value,
            const_narrow=const_narrow.value,
            const_rounding_mode=const_rounding_mode.value,

            qks_signed=qks_signed.value,
            qks_narrow=qks_narrow.value,
            qks_rounding_mode=qks_rounding_mode.value,

            p_signed=p_signed.value,
            p_narrow=p_narrow.value,
            p_rounding_mode=p_rounding_mode.value,

            pt_signed=pt_signed.value,
            pt_narrow=pt_narrow.value,
            pt_rounding_mode=pt_rounding_mode.value,

            vp_signed=vp_signed.value,
            vp_narrow=vp_narrow.value,
            vp_rounding_mode=vp_rounding_mode.value,

            # y_signed=y_signed.value,
            # y_narrow=y_narrow.value,
            # y_rounding_mode=y_rounding_mode.value,

            # vo_signed=vo_signed.value,
            # vo_narrow=vo_narrow.value,
            # vo_rounding_mode=vo_rounding_mode.value,

            _domain="backend.custom_op",
            _outputs=2,
        )
        return y_out, v_out

    @register_rules
    def register_rules():
        return [
            PRule(
                pattern.RewriteRule(
                    StreamingYoloAttention.yolo_attention_pattern,
                    StreamingYoloAttention.rewrite,
                    StreamingYoloAttention._condition,
                ),
                priority=1,
            )
        ]

    @dataclass(frozen=True)
    class DSEPoint:
        lanes_unroll: int
        reduction_unroll: int

        # optional helpers to interop with old code / ONNX storage
        def to_dict(self) -> dict:
            return {
                "lanes_unroll": self.lanes_unroll,
                "reduction_unroll": self.reduction_unroll,
            }

        @staticmethod
        def from_dict(d: dict) -> "StreamingYoloAttention.DSEPoint":
            return StreamingYoloAttention.DSEPoint(
                lanes_unroll=d["lanes_unroll"],
                reduction_unroll=d["reduction_unroll"],
            )

    def get_nodeattr_types(self):
        return {
            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),

            "lanes_unroll": ("i", False, 1),
            "reduction_unroll": ("i", False, 1),

            # YoloAttention structural attrs
            "const_value": ("f", True, 0.0),

            # input quant attrs
            # "in_signed": ("i", True, 0),
            # "in_narrow": ("i", True, 0),
            # "in_rounding_mode": ("s", True, ""),

            # reshape quant attrs
            "rq_signed": ("i", True, 0),
            "rq_narrow": ("i", True, 0),
            "rq_rounding_mode": ("s", True, ""),

            # Q quant attrs
            "q_signed": ("i", True, 0),
            "q_narrow": ("i", True, 0),
            "q_rounding_mode": ("s", True, ""),

            # K quant attrs
            "k_signed": ("i", True, 0),
            "k_narrow": ("i", True, 0),
            "k_rounding_mode": ("s", True, ""),

            # V quant attrs
            "v_signed": ("i", True, 0),
            "v_narrow": ("i", True, 0),
            "v_rounding_mode": ("s", True, ""),

            # Q transpose quant attrs
            "qt_signed": ("i", True, 0),
            "qt_narrow": ("i", True, 0),
            "qt_rounding_mode": ("s", True, ""),

            # QK quant attrs
            "qk_signed": ("i", True, 0),
            "qk_narrow": ("i", True, 0),
            "qk_rounding_mode": ("s", True, ""),

            # const quant attrs
            "const_signed": ("i", True, 0),
            "const_narrow": ("i", True, 0),
            "const_rounding_mode": ("s", True, ""),

            # post-mul quant attrs
            "qks_signed": ("i", True, 0),
            "qks_narrow": ("i", True, 0),
            "qks_rounding_mode": ("s", True, ""),

            # softmax quant attrs
            "p_signed": ("i", True, 0),
            "p_narrow": ("i", True, 0),
            "p_rounding_mode": ("s", True, ""),

            # P transpose quant attrs
            "pt_signed": ("i", True, 0),
            "pt_narrow": ("i", True, 0),
            "pt_rounding_mode": ("s", True, ""),

            # VP quant attrs
            "vp_signed": ("i", True, 0),
            "vp_narrow": ("i", True, 0),
            "vp_rounding_mode": ("s", True, ""),

            # Y output quant attrs
            # "y_signed": ("i", True, 0),
            # "y_narrow": ("i", True, 0),
            # "y_rounding_mode": ("s", True, ""),

            # V output quant attrs
            # "vo_signed": ("i", True, 0),
            # "vo_narrow": ("i", True, 0),
            # "vo_rounding_mode": ("s", True, ""),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node

        return helper.make_node(
            "MatMul",
            inputs=node.input,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        pass

    def verify_node(self):
        pass

    def __get_stream_name(self, name: str) -> str:
        """
        Returns the name of the stream for the tensor.
        """
        return f"{name}_stream"

    def __get_accumulator(self, sums, base, input_quant: TensorQuant) -> str:
        """
        Get the accumulator type for the given input quantization.
        """
        accumulator_bitwidth = base + int(np.floor(np.log2(sums) + 1))

        signed = False
        acc_quant = TensorQuant(
            bitwidth=accumulator_bitwidth,
            signed=signed,
            scale=input_quant.scale,
            zeropt=input_quant.zeropt,
        )
        return f"{get_hls_quant_type(acc_quant)}"

    def __get_lut_type(self) -> str:
        """
        Get the type for the LUT entries in the softmax computation.
        The LUT stores quantized exponentials.
        """
        lut_quant = TensorQuant(
            bitwidth=EXP_PRECISION,  # Use the output bitwidth for max precision
            signed=False,
            scale=0.0, # Not the actual scale, but it is usless for type generation.
            zeropt=0,
        )
        return f"{get_hls_quant_type(lut_quant)}"

    def __get_division_type(self) -> str:
        """
        Get the type for the division result in the softmax computation.
        """

        div_quant = TensorQuant(
            bitwidth=DIV_PRECISION,
            signed=False,
            scale=0.0, # Not the actual scale, but it is usless for type generation.
            zeropt=0,
        )
        return f"{get_hls_quant_type(div_quant)}"

    def __is_power_of_two(self, value) -> bool:
        """Check if a value is a power of two."""
        return value > 0 and float(np.log2(value)).is_integer()

    def __get_shift_quantizer(self, input_quant, output_quant) -> str:
        """Returns the quantizer type for the StreamingSoftmax operation."""

        # Check if the scale is a power of two
        if (
            self.__is_power_of_two(output_quant.scale)
            and self.__is_power_of_two(input_quant.scale)
        ):
            shift = int(np.log2(output_quant.scale)) - int(np.log2(input_quant.scale))
            if shift == 0 and input_quant.bitwidth == output_quant.bitwidth and input_quant.signed == output_quant.signed:
                return f"DequantQuantEqual<{get_hls_quant_type(input_quant)}>"
            return f"DequantQuantPo2<{shift}, {get_hls_quant_type(input_quant)}, {get_hls_quant_type(output_quant)}>"
        else:
            raise ValueError(
                "Float quantization is currently not supported for StreamingSoftmax."
            )

    def __generate_lut_memory(self, softmax_input_bits, exp_precision, softmax_scale, lut_bits):
        """
        Generate LUT contents for a softmax-style exponential using NumPy only:

            LUT[d] = Quantize_RNE_TIES_TO_EVEN( exp(-d * X_SCALE) )   in Q0.F

        where:
        - d is an unsigned integer index in [0 .. 2^INPUT_DATAWIDTH - 1]
        - X_SCALE is the real step per integer diff (e.g. 0.125)
        - F = EXP_PRECISION is the number of fractional bits (e.g. 12, 16, 24)

        This matches the common softmax kernel usage:
            diff = max - x    (>= 0)
            E    = LUT[diff]  ~= exp(x - max)

        Expected config_dict keys:
        INPUT_DATAWIDTH (int)
        EXP_PRECISION   (int)   # fractional bits F
        X_SCALE         (float)

        Optional keys:
        OUTPUT_IS_UNSIGNED (bool, default True)  # LUT is normally unsigned
        SATURATE           (bool, default True)  # saturate to the chosen bitwidth
        OUTPUT_TOTAL_BITS  (int, optional)       # if omitted, uses EXP_PRECISION+1
                                                # (enough to represent values up to ~1.0)

        Notes:
        - Output is an *integer table* representing Q0.F fixed point.
        - We use round-to-nearest, ties-to-even (banker's rounding), like ONNX QuantizeLinear.
        - For exp(0)=1.0: ideal value is 2^F, but if OUTPUT_TOTAL_BITS == F
            you can't represent it. Default OUTPUT_TOTAL_BITS is F+1 so 2^F fits.
        """
        # ---- read config ----
        nbits = int(softmax_input_bits)
        F = int(exp_precision)
        lut_entries = 1 << nbits

        x_scale = float(softmax_scale)

        # Total output bits for the LUT integer values.
        # Default to F+1 so exp(0)=2^F is representable.
        out_total_bits = int(lut_bits)

        if out_total_bits <= 0:
            raise ValueError("OUTPUT_TOTAL_BITS must be positive")
        if F < 0:
            raise ValueError("EXP_PRECISION must be >= 0")

        # ---- build d = 0..2^nbits-1 ----
        d = np.arange(lut_entries, dtype=np.float64)

        # ---- compute real exp(-d * x_scale) ----
        y_real = np.exp(-d * x_scale)

        # ---- quantize to Q0.F with RNE ties-to-even ----
        # y_q = round(y_real * 2^F) with banker's rounding
        scale = float(2 ** F)
        y_q = np.rint(y_real * scale).astype(np.int64)  # np.rint = ties-to-even
        qmin, qmax = 0, (1 << out_total_bits) - 1
        y_q = np.clip(y_q, qmin, qmax)

        # ---- emit C++ variable ----
        # Choose a reasonable C++ primitive based on out_total_bits + signedness
        # (you can override outside if you prefer a fixed type)
        primitive = f"ap_uint<{out_total_bits}>"

        lut_values = [int(v) for v in y_q.tolist()]

        lut_variable = cpp_variable(
            name="LUTmem",
            primitive=primitive,
            value=lut_values,
        )
        return lut_variable.generate_initialization()

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int) -> None:
        """Lower the node to HLS code."""

        hls_kernels = []
        fifos = {}
        q_shape = [1, 2, 32, 400]
        k_shape = [1, 2, 32, 400]
        v_shape = [1, 2, 64, 400]

        ####### SplitReshapeQKV ########
        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(
                f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model."
            )

        input_shape = model.get_tensor_shape(self.onnx_node.input[0])

        (rq_scale, rq_zeropt, rq_bitwidth) = (1,2,3)
        SplitReshapeQKV_output_quant = TensorQuant(
            scale=model.get_initializer(self.onnx_node.input[rq_scale]),
            zeropt=model.get_initializer(self.onnx_node.input[rq_zeropt]),
            bitwidth=model.get_initializer(self.onnx_node.input[rq_bitwidth]),
            signed=self.get_nodeattr("rq_signed"),
            narrow=self.get_nodeattr("rq_narrow"),
            rounding_mode=self.get_nodeattr("rq_rounding_mode"),
        )

        # Input stream names for SplitReshapeQKV
        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(self.get_nodeattr("in_stream_array"))
        ]

        # Output stream names for SplitReshapeQKV
        SplitReshapeQKV_output_names = [
            f"stream_q_{i}_"
            for i in range(2)
        ]
        SplitReshapeQKV_output_names += [
            f"stream_k_{i}_"
            for i in range(2)
        ]
        SplitReshapeQKV_output_names += [
            f"stream_v_{i}_"
            for i in range(2)
        ]

        for output in SplitReshapeQKV_output_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(SplitReshapeQKV_output_quant, 1)}",
                n_array=2,  # Split into 2 streams for Q, K, V each
            )

        SplitReshapeQKV_run_call = f"splitreshapeqkv.run<{hls_tag}>( {self.__get_stream_name(self.onnx_node.input[0])}, stream_q, stream_k, stream_v)"
        SplitReshapeQKV_step_call = f"splitreshapeqkv.step( {self.__get_stream_name(self.onnx_node.input[0])}, stream_q, stream_k, stream_v)"

        SplitReshapeQKV = cpp_object(
            f"SplitReshapeQKV",
            f"splitreshapeqkv",
            template_args=[
                (f"{get_struct_type(input_quant, self.get_nodeattr('in_word_array'))}", "TInputWord"),
                (f"{get_hls_quant_type(input_quant)}", "TInput"),
                (f"{get_struct_type(SplitReshapeQKV_output_quant, self.get_nodeattr('in_word_array'))}", "TSplitWord"),
                (f"{get_hls_quant_type(SplitReshapeQKV_output_quant)}", "TSplit"),
                (f"{self.__get_shift_quantizer(input_quant, SplitReshapeQKV_output_quant)}", "SplitQuantizer"),
                (f"{input_shape[2]}", "IN_HEIGHT"),
                (f"{input_shape[3]}", "IN_WIDTH"),
                (f"{input_shape[1]}", "IN_CHANNELS"),
                "1"
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=input_names,
                outputs=SplitReshapeQKV_output_names,
                name=f"splitreshapeqkv",
                domain="backend.custom_op",
                original_op_type="SplitReshapeQKV",
                hls_tag=hls_tag,
                hls_object_name=f"splitreshapeqkv",
                hls_variable_declarations="",
                hls_run_call=SplitReshapeQKV_run_call,
                hls_step_call=SplitReshapeQKV_step_call,
                hls_object_declaration=SplitReshapeQKV.generate_declaration(),
            )
        )
        hls_tag += 1

        # TensorDuplicator first head for V stream
        TensorDuplicator0_output_names = ["stream_v_out_0_", "stream_v_copy_0_"]
        for output in TensorDuplicator0_output_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(SplitReshapeQKV_output_quant, 1)}",
                n_array=2,  # Duplicate into 2 streams
            )
        TensorDuplicator0_run_call = f"tensorduplicator_head0.run<{hls_tag}>(&stream_v[0], &stream_v_out[0], &stream_v_copy[0])"
        TensorDuplicator0_step_call = "tensorduplicator_head0.step(&stream_v[0], &stream_v_out[0], &stream_v_copy[0])"

        TensorDuplicator0 = cpp_object(
            f"TensorDuplicator",
            f"tensorduplicator_head0",
            template_args=[
                (f"{get_struct_type(SplitReshapeQKV_output_quant, self.get_nodeattr('in_word_array'))}", "TSplitWord"),
                (f"{v_shape[2]}", "DIM_V"),
                (f"{v_shape[3]}", "DIM_SEQ_VP"),
                ("1", "DIM_HEADS"),
                ("1", "W_PAR"),
                ("1", "REDUCE_PAR"),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=["stream_v_0_"],
                outputs=TensorDuplicator0_output_names,
                name=f"tensorduplicator_head0",
                domain="backend.custom_op",
                original_op_type="TensorDuplicator",
                hls_tag=hls_tag,
                hls_object_name=f"tensorduplicator_head0",
                hls_variable_declarations="",
                hls_run_call=TensorDuplicator0_run_call,
                hls_step_call=TensorDuplicator0_step_call,
                hls_object_declaration=TensorDuplicator0.generate_declaration(),
            )
        )
        hls_tag += 1

        # TensorDuplicator second head for V stream
        TensorDuplicator1_output_names = ["stream_v_out_1_", "stream_v_copy_1_"]
        for output in TensorDuplicator1_output_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(SplitReshapeQKV_output_quant, self.get_nodeattr('in_word_array'))}",
                n_array=2,  # Duplicate into 2 streams
            )
        TensorDuplicator1_run_call = f"tensorduplicator_head1.run<{hls_tag}>(&stream_v[1], &stream_v_out[1], &stream_v_copy[1])"
        TensorDuplicator1_step_call = "tensorduplicator_head1.step(&stream_v[1], &stream_v_out[1], &stream_v_copy[1])"

        TensorDuplicator1 = cpp_object(
            f"TensorDuplicator",
            f"tensorduplicator_head1",
            template_args=[
                (f"{get_struct_type(SplitReshapeQKV_output_quant, self.get_nodeattr('in_word_array'))}", "TSplitWord"),
                (f"{v_shape[2]}", "DIM_V"),
                (f"{v_shape[3]}", "DIM_SEQ_VP"),
                ("1", "DIM_HEADS"),
                ("1", "W_PAR"),
                ("1", "REDUCE_PAR"),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=["stream_v_1_"],
                outputs=TensorDuplicator1_output_names,
                name=f"tensorduplicator_head1",
                domain="backend.custom_op",
                original_op_type="TensorDuplicator",
                hls_tag=hls_tag,
                hls_object_name=f"tensorduplicator_head1",
                hls_variable_declarations="",
                hls_run_call=TensorDuplicator1_run_call,
                hls_step_call=TensorDuplicator1_step_call,
                hls_object_declaration=TensorDuplicator1.generate_declaration(),
            )
        )
        hls_tag += 1

        # QK matmul first head
        (qk_scale, qk_zeropt, qk_bitwidth) = (16, 17, 18)
        QKMatMul_output_quant = TensorQuant(
            scale=model.get_initializer(self.onnx_node.input[qk_scale]),
            zeropt=model.get_initializer(self.onnx_node.input[qk_zeropt]),
            bitwidth=model.get_initializer(self.onnx_node.input[qk_bitwidth]),
            signed=self.get_nodeattr("qk_signed"),
            narrow=self.get_nodeattr("qk_narrow"),
            rounding_mode=self.get_nodeattr("qk_rounding_mode"),
        )
        QKMatmul_acc_quant = TensorQuant(
            bitwidth=SplitReshapeQKV_output_quant.bitwidth * 2 + int(np.ceil(np.log2(q_shape[2]))),
            signed=SplitReshapeQKV_output_quant.signed,
            scale=SplitReshapeQKV_output_quant.scale * SplitReshapeQKV_output_quant.scale,
            zeropt=0,
        )
        QKMatMul0_output_names = ["stream_qk_0_"]
        for output in QKMatMul0_output_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(QKMatMul_output_quant, 1)}",
                n_array=2, 
            )
        QKMatMul0_run_call = f"matmulqk_head0.run<{hls_tag}>(&stream_q[0], &stream_k[0], &stream_qk[0])"
        QKMatMul0_step_call = f"matmulqk_head0.step(&stream_q[0], &stream_k[0], &stream_qk[0])"

        QKMatMul0 = cpp_object(
            f"QKMatMul",
            f"matmulqk_head0",
            template_args=[
                (f"{get_struct_type(SplitReshapeQKV_output_quant, 1)}", "TQInputWord"),
                (f"{get_hls_quant_type(SplitReshapeQKV_output_quant)}", "TQInput"),
                (f"{get_struct_type(SplitReshapeQKV_output_quant, 1)}", "TKInputWord"),
                (f"{get_hls_quant_type(SplitReshapeQKV_output_quant)}", "TKInput"),
                (f"{get_struct_type(QKMatMul_output_quant, 1)}", "TQKWord"),
                (f"{get_hls_quant_type(QKMatMul_output_quant)}", "TQK"),
                (f"{get_hls_quant_type(QKMatmul_acc_quant)}", "TAccQK"),
                (f"{self.__get_shift_quantizer(QKMatmul_acc_quant, QKMatMul_output_quant)}", "QKQuantizer"),
                ("1", "DIM_HEADS"),
                (f"{q_shape[3]}", "DIM_Q"),
                (f"{k_shape[3]}", "DIM_K"),
                (f"{q_shape[2]}", "DIM_SEQ_QK"),
                ("1", "REDUCE_PAR"),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=["stream_q_0_", "stream_k_0_"],
                outputs=QKMatMul0_output_names,
                name=f"matmulqk_head0",
                domain="backend.custom_op",
                original_op_type="QKMatMul",
                hls_tag=hls_tag,
                hls_object_name=f"matmulqk_head0",
                hls_variable_declarations="",
                hls_run_call=QKMatMul0_run_call,
                hls_step_call=QKMatMul0_step_call,
                hls_object_declaration=QKMatMul0.generate_declaration(),
            )
        )
        hls_tag += 1

        # QK matmul second head
        QKMatMul1_output_names = ["stream_qk_1_"]
        for output in QKMatMul1_output_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(QKMatMul_output_quant, 1)}",
                n_array=2, 
            )
        QKMatMul1_run_call = f"matmulqk_head1.run<{hls_tag}>(&stream_q[1], &stream_k[1], &stream_qk[1])"
        QKMatMul1_step_call = f"matmulqk_head1.step(&stream_q[1], &stream_k[1], &stream_qk[1])"

        QKMatMul1 = cpp_object(
            f"QKMatMul",
            f"matmulqk_head1",
            template_args=[
                (f"{get_struct_type(SplitReshapeQKV_output_quant, 1)}", "TQInputWord"),
                (f"{get_hls_quant_type(SplitReshapeQKV_output_quant)}", "TQInput"),
                (f"{get_struct_type(SplitReshapeQKV_output_quant, 1)}", "TKInputWord"),
                (f"{get_hls_quant_type(SplitReshapeQKV_output_quant)}", "TKInput"),
                (f"{get_struct_type(QKMatMul_output_quant, 1)}", "TQKWord"),
                (f"{get_hls_quant_type(QKMatMul_output_quant)}", "TQK"),
                (f"{get_hls_quant_type(QKMatmul_acc_quant)}", "TAccQK"),
                (f"{self.__get_shift_quantizer(QKMatmul_acc_quant, QKMatMul_output_quant)}", "QKQuantizer"),
                ("1", "DIM_HEADS"),
                (f"{q_shape[3]}", "DIM_Q"),
                (f"{k_shape[3]}", "DIM_K"),
                (f"{q_shape[2]}", "DIM_SEQ_QK"),
                ("1", "REDUCE_PAR"),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=["stream_q_1_", "stream_k_1_"],
                outputs=QKMatMul1_output_names,
                name=f"matmulqk_head1",
                domain="backend.custom_op",
                original_op_type="QKMatMul",
                hls_tag=hls_tag,
                hls_object_name=f"matmulqk_head1",
                hls_variable_declarations="",
                hls_run_call=QKMatMul1_run_call,
                hls_step_call=QKMatMul1_step_call,
                hls_object_declaration=QKMatMul1.generate_declaration(),
            )
        )
        hls_tag += 1

        # Const scaling for first head
        (const_scale, const_zeropt, const_bitwidth) = (19, 20, 21)
        ConstScale_output_quant = TensorQuant(
            scale=model.get_initializer(self.onnx_node.input[const_scale]),
            zeropt=model.get_initializer(self.onnx_node.input[const_zeropt]),
            bitwidth=model.get_initializer(self.onnx_node.input[const_bitwidth]),
            signed=self.get_nodeattr("const_signed"),
            narrow=self.get_nodeattr("const_narrow"),
            rounding_mode=self.get_nodeattr("const_rounding_mode"),
        )
        ConstScale_value = (
            self.get_nodeattr("const_value") / ConstScale_output_quant.scale
        )
        ConstScale_variable_declaration = f"const {get_hls_quant_type(ConstScale_output_quant)} CONST_SCALE = {int(ConstScale_value)};"

        (qks_scale, qks_zeropt, qks_bitwidth) = (22, 23, 24)
        QKS_output_quant = TensorQuant(
            scale=model.get_initializer(self.onnx_node.input[qks_scale]),
            zeropt=model.get_initializer(self.onnx_node.input[qks_zeropt]),
            bitwidth=model.get_initializer(self.onnx_node.input[qks_bitwidth]),
            signed=self.get_nodeattr("qks_signed"),
            narrow=self.get_nodeattr("qks_narrow"),
            rounding_mode=self.get_nodeattr("qks_rounding_mode"),
        )

        QKS_mul_quant = TensorQuant(
            bitwidth=QKMatMul_output_quant.bitwidth + ConstScale_output_quant.bitwidth,
            signed=QKMatMul_output_quant.signed or ConstScale_output_quant.signed,
            scale=QKMatMul_output_quant.scale * ConstScale_output_quant.scale,
            zeropt=0,
        )

        ConstMul0_output_names = ["stream_qkscaled_0_"]
        for output in ConstMul0_output_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(QKS_output_quant, 1)}",
                n_array=2, 
            )
        ConstMul0_run_call = f"constmulqk_head0.run<{hls_tag}>(&stream_qk[0], CONST_SCALE, &stream_qkscaled[0])"
        ConstMul0_step_call = f"constmulqk_head0.step(&stream_qk[0], CONST_SCALE, &stream_qkscaled[0])"

        ConstMul0 = cpp_object(
            f"StreamingConstMul",
            f"constmulqk_head0",
            template_args=[
                (f"{get_struct_type(QKMatMul_output_quant, 1)}", "TInputWord"),
                (f"{get_hls_quant_type(QKMatMul_output_quant)}", "TInput"),
                (f"{get_hls_quant_type(ConstScale_output_quant)}", "TConst"),
                (f"{get_struct_type(QKS_output_quant, 1)}", "TOutputWord"),
                (f"{get_hls_quant_type(QKS_output_quant)}", "TOutput"),
                (f"{get_hls_quant_type(QKS_mul_quant)}", "TMul"),
                (f"DequantQuantEqual<{get_hls_quant_type(QKS_mul_quant)}>", "MulActivation"),
                (f"{self.__get_shift_quantizer(QKS_mul_quant, QKS_output_quant)}", "MulQuantizer"),
                (f"{q_shape[3]}", "MUL_HEIGHT"),
                (f"{k_shape[3]}", "MUL_WIDTH"),
                ("1", "MUL_CHANNELS"),
                ("1", "MUL_W_PAR"),
                ("1", "MUL_CH_PAR"),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=["stream_qk_0_"],
                outputs=ConstMul0_output_names,
                name=f"constmulqk_head0",
                domain="backend.custom_op",
                original_op_type="StreamingConstMul",
                hls_tag=hls_tag,
                hls_object_name=f"constmulqk_head0",
                hls_variable_declarations=ConstScale_variable_declaration,
                hls_run_call=ConstMul0_run_call,
                hls_step_call=ConstMul0_step_call,
                hls_object_declaration=ConstMul0.generate_declaration(),
            )
        )
        hls_tag += 1

        # Const scaling for second head
        ConstMul1_output_names = ["stream_qkscaled_1_"]
        for output in ConstMul1_output_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(QKS_output_quant, 1)}",
                n_array=2, 
            )
        ConstMul1_run_call = f"constmulqk_head1.run<{hls_tag}>(&stream_qk[1], CONST_SCALE, &stream_qkscaled[1])"
        ConstMul1_step_call = f"constmulqk_head1.step(&stream_qk[1], CONST_SCALE, &stream_qkscaled[1])" 
        ConstMul1 = cpp_object(
            f"StreamingConstMul",
            f"constmulqk_head1",
            template_args=[
                (f"{get_struct_type(QKMatMul_output_quant, 1)}", "TInputWord"),
                (f"{get_hls_quant_type(QKMatMul_output_quant)}", "TInput"),
                (f"{get_hls_quant_type(ConstScale_output_quant)}", "TConst"),
                (f"{get_struct_type(QKS_output_quant, 1)}", "TOutputWord"),
                (f"{get_hls_quant_type(QKS_output_quant)}", "TOutput"),
                (f"{get_hls_quant_type(QKS_mul_quant)}", "TMul"),
                (f"DequantQuantEqual<{get_hls_quant_type(QKS_mul_quant)}>", "MulActivation"),
                (f"{self.__get_shift_quantizer(QKS_mul_quant, QKS_output_quant)}", "MulQuantizer"),
                (f"{q_shape[3]}", "MUL_HEIGHT"),
                (f"{k_shape[3]}", "MUL_WIDTH"),
                ("1", "MUL_CHANNELS"),
                ("1", "MUL_W_PAR"),
                ("1", "MUL_CH_PAR"),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=["stream_qk_1_"],
                outputs=ConstMul1_output_names,
                name=f"constmulqk_head1",
                domain="backend.custom_op",
                original_op_type="StreamingConstMul",
                hls_tag=hls_tag,
                hls_object_name=f"constmulqk_head1",
                hls_variable_declarations="",
                hls_run_call=ConstMul1_run_call,
                hls_step_call=ConstMul1_step_call,
                hls_object_declaration=ConstMul1.generate_declaration(),
            )
        )
        hls_tag += 1

        # Softmax for first head
        (p_scale, p_zeropt, p_bitwidth) = (25, 26, 27)
        Softmax_output_quant = TensorQuant(
            scale=model.get_initializer(self.onnx_node.input[p_scale]),
            zeropt=model.get_initializer(self.onnx_node.input[p_zeropt]),
            bitwidth=model.get_initializer(self.onnx_node.input[p_bitwidth]),
            signed=self.get_nodeattr("p_signed"),
            narrow=self.get_nodeattr("p_narrow"),
            rounding_mode=self.get_nodeattr("p_rounding_mode"),
        )

        Softmax_acc_quant = TensorQuant(
            bitwidth=EXP_PRECISION + int(np.ceil(np.log2(q_shape[3]))),
            signed=False,
            scale=2 ** (-EXP_PRECISION),
            zeropt=0,
        )

        Softmax_div_quant = TensorQuant(
            bitwidth=DIV_PRECISION,
            signed=False,
            scale=2 ** -(DIV_PRECISION - EXP_PRECISION),
            zeropt=0,
        )

        Softmax_lut_quant = TensorQuant(
            bitwidth=EXP_PRECISION,
            signed=False,
            scale=2 ** (-EXP_PRECISION),
            zeropt=0,
        )

        Softmax_lut_variable_declaration = self.__generate_lut_memory(
            softmax_input_bits=QKS_output_quant.bitwidth,
            exp_precision=EXP_PRECISION,
            softmax_scale=QKS_output_quant.scale,
            lut_bits=Softmax_lut_quant.bitwidth,
        ).code

        Softmax0_output_names = ["stream_p_0_"]
        for output in Softmax0_output_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(Softmax_output_quant, 1)}",
                n_array=2, 
            )
        Softmax0_run_call = f"softmax_head0.run<{hls_tag}>(&stream_qkscaled[0], LUTmem, &stream_p[0])"
        Softmax0_step_call = f"softmax_head0.step(&stream_qkscaled[0], LUTmem, &stream_p[0])"

        Softmax0 = cpp_object(
            f"StreamingSoftmax",
            f"softmax_head0",
            template_args=[
                (f"{get_struct_type(QKS_output_quant, 1)}", "TInputWord"),
                (f"{get_hls_quant_type(QKS_output_quant)}", "TInput"),
                (f"{get_struct_type(Softmax_output_quant, 1)}", "TOutputWord"),
                (f"{get_hls_quant_type(Softmax_output_quant)}", "TOutput"),
                (f"{get_hls_quant_type(Softmax_lut_quant)}", "TLut"),
                (f"{get_hls_quant_type(Softmax_acc_quant)}", "TAcc"),
                (f"{get_hls_quant_type(Softmax_div_quant)}", "TDiv"),
                (f"{self.__get_shift_quantizer(Softmax_div_quant, Softmax_output_quant)}", "Quantizer"),
                (f"{2**QKS_output_quant.bitwidth}", "LUT_SIZE"),
                (f"{q_shape[3]}", "HEIGHT"),
                (f"{k_shape[3]}", "WIDTH"),
                ("1", "W_PAR"),
                ("1", "CH_PAR"),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=["stream_qkscaled_0_"],
                outputs=Softmax0_output_names,
                name=f"softmax_head0",
                domain="backend.custom_op",
                original_op_type="StreamingSoftmax",
                hls_tag=hls_tag,
                hls_object_name=f"softmax_head0",
                hls_variable_declarations=Softmax_lut_variable_declaration,
                hls_run_call=Softmax0_run_call,
                hls_step_call=Softmax0_step_call,
                hls_object_declaration=Softmax0.generate_declaration(),
            )
        )
        hls_tag += 1

        Softmax1_output_names = ["stream_p_1_"]
        for output in Softmax1_output_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(Softmax_output_quant, 1)}",
                n_array=2, 
            )
        Softmax1_run_call = f"softmax_head1.run<{hls_tag}>(&stream_qkscaled[1], LUTmem, &stream_p[1])"
        Softmax1_step_call = f"softmax_head1.step(&stream_qkscaled[1], LUTmem, &stream_p[1])"

        Softmax1 = cpp_object(
            f"StreamingSoftmax",
            f"softmax_head1",
            template_args=[
                (f"{get_struct_type(QKS_output_quant, 1)}", "TInputWord"),
                (f"{get_hls_quant_type(QKS_output_quant)}", "TInput"),
                (f"{get_struct_type(Softmax_output_quant, 1)}", "TOutputWord"),
                (f"{get_hls_quant_type(Softmax_output_quant)}", "TOutput"),
                (f"{get_hls_quant_type(Softmax_lut_quant)}", "TLut"),
                (f"{get_hls_quant_type(Softmax_acc_quant)}", "TAcc"),
                (f"{get_hls_quant_type(Softmax_div_quant)}", "TDiv"),
                (f"{self.__get_shift_quantizer(Softmax_div_quant, Softmax_output_quant)}", "Quantizer"),
                (f"{2**QKS_output_quant.bitwidth}", "LUT_SIZE"),
                (f"{q_shape[3]}", "HEIGHT"),
                (f"{k_shape[3]}", "WIDTH"),
                ("1", "W_PAR"),
                ("1", "CH_PAR"),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=["stream_qkscaled_1_"],
                outputs=Softmax1_output_names,
                name=f"softmax_head1",
                domain="backend.custom_op",
                original_op_type="StreamingSoftmax",
                hls_tag=hls_tag,
                hls_object_name=f"softmax_head1",
                hls_variable_declarations="",
                hls_run_call=Softmax1_run_call,
                hls_step_call=Softmax1_step_call,
                hls_object_declaration=Softmax1.generate_declaration(),
            )
        )
        hls_tag += 1

        # Transpose V for first head
        TransposeV0_output_names = ["stream_v_transposed_0_"]
        for output in TransposeV0_output_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(SplitReshapeQKV_output_quant, 1)}",
                n_array=2, 
            )

        TransposeV0_run_call = f"transposev_head0.run<{hls_tag}>(&stream_v_copy[0], &stream_v_transposed[0])"
        TransposeV0_step_call = f"transposev_head0.step(&stream_v_copy[0], &stream_v_transposed[0])"

        TransposeV0 = cpp_object(
            f"TransposeRowCol",
            f"transposev_head0",
            template_args=[
                (f"{get_struct_type(SplitReshapeQKV_output_quant, 1)}", "TInputWord"),
                (f"{v_shape[2]}", "DIM_V"),
                (f"{v_shape[3]}", "DIM_SEQ_VP"),
                ("1", "DIM_HEADS"),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=["stream_v_copy_0_"],
                outputs=TransposeV0_output_names,
                name=f"transposev_head0",
                domain="backend.custom_op",
                original_op_type="TransposeRowCol",
                hls_tag=hls_tag,
                hls_object_name=f"transposev_head0",
                hls_variable_declarations="",
                hls_run_call=TransposeV0_run_call,
                hls_step_call=TransposeV0_step_call,
                hls_object_declaration=TransposeV0.generate_declaration(),
            )
        )
        hls_tag += 1

        # Transpose V for second head
        TransposeV1_output_names = ["stream_v_transposed_1_"]
        for output in TransposeV1_output_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(SplitReshapeQKV_output_quant, 1)}",
                n_array=2, 
            )

        TransposeV1_run_call = f"transposev_head1.run<{hls_tag}>(&stream_v_copy[1], &stream_v_transposed[1])"
        TransposeV1_step_call = f"transposev_head1.step(&stream_v_copy[1], &stream_v_transposed[1])"

        TransposeV1 = cpp_object(
            f"TransposeRowCol",
            f"transposev_head1",
            template_args=[
                (f"{get_struct_type(SplitReshapeQKV_output_quant, 1)}", "TInputWord"),
                (f"{v_shape[2]}", "DIM_V"),
                (f"{v_shape[3]}", "DIM_SEQ_VP"),
                ("1", "DIM_HEADS"),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=["stream_v_copy_1_"],
                outputs=TransposeV1_output_names,
                name=f"transposev_head1",
                domain="backend.custom_op",
                original_op_type="TransposeRowCol",
                hls_tag=hls_tag,
                hls_object_name=f"transposev_head1",
                hls_variable_declarations="",
                hls_run_call=TransposeV1_run_call,
                hls_step_call=TransposeV1_step_call,
                hls_object_declaration=TransposeV1.generate_declaration(),
            )
        )
        hls_tag += 1

        # VP matmul for first head
        (vp_scale, vp_zeropt, vp_bitwidth) = (31, 32, 33)
        VPMatMul_output_quant = TensorQuant(
            scale=model.get_initializer(self.onnx_node.input[vp_scale]),
            zeropt=model.get_initializer(self.onnx_node.input[vp_zeropt]),
            bitwidth=model.get_initializer(self.onnx_node.input[vp_bitwidth]),
            signed=self.get_nodeattr("vp_signed"),
            narrow=self.get_nodeattr("vp_narrow"),
            rounding_mode=self.get_nodeattr("vp_rounding_mode"),
        )

        VPMatMul_acc_quant = TensorQuant(
            bitwidth=Softmax_output_quant.bitwidth + SplitReshapeQKV_output_quant.bitwidth + int(np.ceil(np.log2(k_shape[3]))),
            signed=Softmax_output_quant.signed or SplitReshapeQKV_output_quant.signed,
            scale=Softmax_output_quant.scale * SplitReshapeQKV_output_quant.scale,
            zeropt=0,
        )

        VPMatMul0_output_names = ["stream_y_0_"]
        for output in VPMatMul0_output_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(SplitReshapeQKV_output_quant, 1)}",
                n_array=2, 
            )
        VPMatMul0_run_call = f"matmulvp_head0.run<{hls_tag}>(&stream_v_transposed[0], &stream_p[0], &stream_y[0])"
        VPMatMul0_step_call = f"matmulvp_head0.step(&stream_v_transposed[0], &stream_p[0], &stream_y[0])"

        VPMatMul0 = cpp_object(
            f"VPMatMul",
            f"matmulvp_head0",
            template_args=[
                (f"{get_struct_type(SplitReshapeQKV_output_quant, 1)}", "TVInputWord"),
                (f"{get_hls_quant_type(SplitReshapeQKV_output_quant)}", "TVInput"),
                (f"{get_struct_type(Softmax_output_quant, 1)}", "TPInputWord"),
                (f"{get_hls_quant_type(Softmax_output_quant)}", "TPInput"),
                (f"{get_struct_type(VPMatMul_output_quant, 1)}", "TVPOutputWord"),
                (f"{get_hls_quant_type(VPMatMul_output_quant)}", "TVPOutput"),
                (f"{get_hls_quant_type(VPMatMul_acc_quant)}", "TAccVP"),
                (f"{self.__get_shift_quantizer(VPMatMul_acc_quant, VPMatMul_output_quant)}", "VPQuantizer"),
                ("1", "DIM_HEADS"),
                (f"{v_shape[2]}", "DIM_V"),
                (f"{q_shape[3]}", "DIM_P"),
                (f"{v_shape[3]}", "DIM_SEQ_VP"),
                ("1", "REDUCE_PAR"),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=["stream_v_transposed_0_", "stream_p_0_"],
                outputs=VPMatMul0_output_names,
                name=f"matmulvp_head0",
                domain="backend.custom_op",
                original_op_type="VPMatMul",
                hls_tag=hls_tag,
                hls_object_name=f"matmulvp_head0",
                hls_variable_declarations="",
                hls_run_call=VPMatMul0_run_call,
                hls_step_call=VPMatMul0_step_call,
                hls_object_declaration=VPMatMul0.generate_declaration(),
            )
        )
        hls_tag += 1

        # VP matmul for second head
        VPMatMul1_output_names = ["stream_y_1_"]
        for output in VPMatMul1_output_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(SplitReshapeQKV_output_quant, 1)}",
                n_array=2, 
            )
        VPMatMul1_run_call = f"matmulvp_head1.run<{hls_tag}>(&stream_v_transposed[1], &stream_p[1], &stream_y[1])"
        VPMatMul1_step_call = f"matmulvp_head1.step(&stream_v_transposed[1], &stream_p[1], &stream_y[1])"

        VPMatMul1 = cpp_object(
            f"VPMatMul",
            f"matmulvp_head1",
            template_args=[
                (f"{get_struct_type(SplitReshapeQKV_output_quant, 1)}", "TVInputWord"),
                (f"{get_hls_quant_type(SplitReshapeQKV_output_quant)}", "TVInput"),
                (f"{get_struct_type(Softmax_output_quant, 1)}", "TPInputWord"),
                (f"{get_hls_quant_type(Softmax_output_quant)}", "TPInput"),
                (f"{get_struct_type(VPMatMul_output_quant, 1)}", "TVPOutputWord"),
                (f"{get_hls_quant_type(VPMatMul_output_quant)}", "TVPOutput"),
                (f"{get_hls_quant_type(VPMatMul_acc_quant)}", "TAccVP"),
                (f"{self.__get_shift_quantizer(VPMatMul_acc_quant, VPMatMul_output_quant)}", "VPQuantizer"),
                ("1", "DIM_HEADS"),
                (f"{v_shape[2]}", "DIM_V"),
                (f"{q_shape[3]}", "DIM_P"),
                (f"{v_shape[3]}", "DIM_SEQ_VP"),
                ("1", "REDUCE_PAR"),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=["stream_v_transposed_1_", "stream_p_1_"],
                outputs=VPMatMul1_output_names,
                name=f"matmulvp_head1",
                domain="backend.custom_op",
                original_op_type="VPMatMul",
                hls_tag=hls_tag,
                hls_object_name=f"matmulvp_head1",
                hls_variable_declarations="",
                hls_run_call=VPMatMul1_run_call,
                hls_step_call=VPMatMul1_step_call,
                hls_object_declaration=VPMatMul1.generate_declaration(),
            )
        )
        hls_tag += 1

        # Reshape for V
        output_v_quant = get_custom_tensor_datatype(model, self.onnx_node.output[1])
        if output_v_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{self.onnx_node.output[1]}' not found in model."
            )
        output_v_names = [
            f"{self.__get_stream_name(self.onnx_node.output[1])}_{i}_"
            for i in range(self.get_nodeattr("out_stream_array"))
        ]
        for output in output_v_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(output_v_quant, self.get_nodeattr('out_word_array'))}",
                n_array=self.get_nodeattr("out_stream_array"),
            )
        ReshapeV_run_call = f"reshapev.run<{hls_tag}>(stream_v_out, {self.__get_stream_name(self.onnx_node.output[1])})"
        ReshapeV_step_call = f"reshapev.step(stream_v_out, {self.__get_stream_name(self.onnx_node.output[1])})"

        ReshapeV = cpp_object(
            f"ReshapeV",
            f"reshapev",
            template_args=[
                (
                    f"{get_struct_type(SplitReshapeQKV_output_quant, self.get_nodeattr('in_word_array'))}",
                    "TSplitWord",
                ),
                (f"{get_hls_quant_type(SplitReshapeQKV_output_quant)}", "TSplit"),
                (
                    f"{get_struct_type(SplitReshapeQKV_output_quant, self.get_nodeattr('out_word_array'))}",
                    "TReshapeWord",
                ),
                (f"{get_hls_quant_type(SplitReshapeQKV_output_quant)}", "TReshape"),
                (
                    f"DequantQuantEqual<{get_hls_quant_type(SplitReshapeQKV_output_quant)}>",
                    "Quantizer",
                ),
                (f"20", "OUT_HEIGHT"),
                (f"20", "OUT_WIDTH"),
                (f"128", "OUT_CH"),
                ("1", "REDUCE_PAR"),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=["stream_v_out_0_", "stream_v_out_1_"],
                outputs=output_v_names,
                name=f"reshapev",
                domain="backend.custom_op",
                original_op_type="ReshapeV",
                hls_tag=hls_tag,
                hls_object_name=f"reshapev",
                hls_variable_declarations="",
                hls_run_call=ReshapeV_run_call,
                hls_step_call=ReshapeV_step_call,
                hls_object_declaration=ReshapeV.generate_declaration(),
            )
        )
        hls_tag += 1

        # Reshape for Y
        output_y_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_y_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model."
            )
        output_y_names = [
            f"{self.__get_stream_name(self.onnx_node.output[0])}_{i}_"
            for i in range(self.get_nodeattr("out_stream_array"))
        ]
        for output in output_y_names:
            fifos[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(output_y_quant, self.get_nodeattr('out_word_array'))}",
                n_array=self.get_nodeattr("out_stream_array"),
            )

        ReshapeY_run_call = f"reshapey.run<{hls_tag}>(stream_y, {self.__get_stream_name(self.onnx_node.output[0])})"
        ReshapeY_step_call = f"reshapey.step(stream_y, {self.__get_stream_name(self.onnx_node.output[0])})"

        ReshapeY = cpp_object(
            f"ReshapeV",
            f"reshapey",
            template_args=[
                (
                    f"{get_struct_type(VPMatMul_output_quant, self.get_nodeattr('in_word_array'))}",
                    "TVPOutputWord",
                ),
                (f"{get_hls_quant_type(VPMatMul_output_quant)}", "TVPOutput"),
                (
                    f"{get_struct_type(output_y_quant, self.get_nodeattr('out_word_array'))}",
                    "TOutputWord",
                ),
                (f"{get_hls_quant_type(output_y_quant)}", "TOutput"),
                (f"DequantQuantEqual<{get_hls_quant_type(output_y_quant)}>",
                    "Quantizer"),
                (f"20", "OUT_HEIGHT"),
                (f"20", "OUT_WIDTH"),
                (f"128", "OUT_CH"),
                ("1", "REDUCE_PAR"),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=["stream_y_0_", "stream_y_1_"],
                outputs=output_y_names,
                name=f"reshapey",
                domain="backend.custom_op",
                original_op_type="ReshapeV",
                hls_tag=hls_tag,
                hls_object_name=f"reshapey",
                hls_variable_declarations="",
                hls_run_call=ReshapeY_run_call,
                hls_step_call=ReshapeY_step_call,
                hls_object_declaration=ReshapeY.generate_declaration(),
            )
        )

        return hls_kernels, [], fifos, hls_tag

    def get_latency(self, model: ModelWrapper) -> int:
        """ Estimate the latency of the StreamingSoftmax operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated latency in clock cycles.
        """
        return 64*400*400

    def get_brams(self, model: ModelWrapper) -> int:
        """ Estimate the BRAM usage of the StreamingSoftmax operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated BRAM usage.
        """
        return 80

    def get_dsps(self, model: ModelWrapper) -> int:
        """ Estimate the DSP usage of the StreamingSoftmax operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated DSP usage.
        """
        return 8

    def has_linebuffer(self) -> bool:
        """ Check if the StreamingSoftmax operation requires a line buffer.
        Returns:
            bool: True if Line Buffering is required, False otherwise.
        """
        return False

    def __current_dse_point(self) -> "StreamingYoloAttention.DSEPoint":
        """ Returns the current DSE point of the StreamingYoloAttention operation. """
        return StreamingYoloAttention.DSEPoint(
            lanes_unroll=self.get_nodeattr("lanes_unroll"),
            reduction_unroll=self.get_nodeattr("reduction_unroll"),
        )

    def get_dse_points(
        self, model: ModelWrapper
    ) -> list["StreamingYoloAttention.DSEPoint"]:
        """Generate the list of valid DSE points for the StreamingYoloAttention operation."""
        return [
            self.DSEPoint(lanes_unroll=1, reduction_unroll=1)
        ]

    def apply_point(self, model: ModelWrapper, point: "StreamingYoloAttention.DSEPoint"):
        """ Set the parallelization attributes for the StreamingYoloAttention operation.
        Args:
            point (StreamingYoloAttention.DSEPoint): The DSE point containing the unrolling parameters.
        """
        self.set_nodeattr("lanes_unroll", point.lanes_unroll)
        self.set_nodeattr("reduction_unroll", point.reduction_unroll)

        self.set_nodeattr("in_stream_array", point.lanes_unroll)
        self.set_nodeattr("out_stream_array", point.lanes_unroll)
        self.set_nodeattr("in_word_array", point.reduction_unroll)
        self.set_nodeattr("out_word_array", point.reduction_unroll)
