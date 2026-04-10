import numpy as np
import onnxruntime as rt
from onnxscript.rewriter import pattern
from onnx import TensorProto, helper
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import TensorQuant, get_custom_tensor_datatype
from backend.core.tensor_fifo import TensorFifo
from backend.custom_op.hlskernel import HLSKernel
from backend.custom_op.op_base import NN2FPGAOp, NodeInterface
from backend.custom_op.register_rewrite_rule import register_rules
from backend.util.codegen_utils import (
    cpp_function,
    cpp_object,
    get_struct_type,
    get_hls_quant_type,
)


class StreamingMatMul(NN2FPGAOp):
    """
    Node implementing MatMul with optional input-A transpose absorbed internally.

    Supports two graph patterns:
      1. Plain:     Quant(x) ──┐
                               MatMul → Quant(out)
                   Quant(y) ──┘

      2. Transpose: Quant(x) → Transpose([0,1,3,2]) → Quant(outer) ──┐
                                                                       MatMul → Quant(out)
                                                         Quant(y) ────┘
         In case 2, the Transpose+Quant_outer are absorbed: the node's
         input[0] is wired to Quant(x)'s output tensor, and transpose_a=1
         tells lower_to_hls to swap IN_HEIGHT↔IN_WIDTH in the template.
    """

    # ------------------------------------------------------------------ #
    #  Pattern 1 — plain Quant(x) @ Quant(y)                              #
    # ------------------------------------------------------------------ #
 
    @staticmethod
    def pattern_plain(op, x, y):
        x_q = op.Quant(x, _allow_other_inputs=True, _allow_other_attributes=True)
        y_q = op.Quant(y, _allow_other_inputs=True, _allow_other_attributes=True)
        return op.MatMul(x_q, y_q, _allow_other_attributes=True)

    @staticmethod
    def rewrite_plain(op, x, y):
        return op.StreamingMatMul(
            x, y,
            _domain="backend.custom_op",
            transpose_a=0,
        )
  
    # ------------------------------------------------------------------ #
    #  Pattern 2 — Quant(Transpose(Quant(x))) @ Quant(y)                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def pattern_transpose(op, x, y):
        x_q_inner = op.Quant(x, _allow_other_inputs=True, _allow_other_attributes=True)
        x_t       = op.Transpose(x_q_inner, _allow_other_attributes=True)
        x_q_outer = op.Quant(x_t, _allow_other_inputs=True, _allow_other_attributes=True)
        y_q       = op.Quant(y, _allow_other_inputs=True, _allow_other_attributes=True)
        return op.MatMul(x_q_outer, y_q, _allow_other_attributes=True)

    @staticmethod
    def rewrite_transpose(op, x, y):
        # Wire directly to x (pre-inner-Quant raw tensor) — the inner
        # Quant's output tensor is used as input so quantization info
        # is preserved; the Transpose is absorbed (transpose_a=1).
        x_q_inner = op.Quant(x, _allow_other_inputs=True, _allow_other_attributes=True)
        y_q       = op.Quant(y, _allow_other_inputs=True, _allow_other_attributes=True)
        return op.StreamingMatMul(
            x_q_inner, y_q,
            _domain="backend.custom_op",
            transpose_a=1,
        )

    @register_rules
   
    def register_rules():
        return [
            pattern.RewriteRule(
                StreamingMatMul.pattern_transpose,
                StreamingMatMul.rewrite_transpose,
            ),
            pattern.RewriteRule(
                StreamingMatMul.pattern_plain,
                StreamingMatMul.rewrite_plain,
            ),
        ]
    # ------------------------------------------------------------------ #
    #  Node attributes                                                     #
    # ------------------------------------------------------------------ #
    def get_nodeattr_types(self):
        return {
            "channel_unroll":   ("i", False, 1),
            "width_unroll":     ("i", False, 1),
            "in_stream_array":  ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array":    ("i", False, 1),
            "out_word_array":   ("i", False, 1),
            # 1 → input A was originally transposed [0,1,3,2]; swap H↔W
            "transpose_a":      ("i", False, 0),
        }

    # ------------------------------------------------------------------ #
    #  Shape / datatype inference                                          #
    # ------------------------------------------------------------------ #
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
        node = self.onnx_node
        a = context[node.input[0]]
        b = context[node.input[1]]
        if self.get_nodeattr("transpose_a"):
            # replicate the [0,1,3,2] transpose that was absorbed
            a = np.transpose(a, (0, 1, 3, 2))
        context[node.output[0]] = np.matmul(a, b).astype(np.float32)

    def verify_node(self):
        pass

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #
    def __get_stream_name(self, name: str) -> str:
        return f"{name}_stream"

    def __is_power_of_two(self, value) -> bool:
        return value > 0 and float(np.log2(value)).is_integer()

    def __get_accumulator(self, input_quantA, input_quantB) -> str:
        signed    = input_quantA.signed or input_quantB.signed
        acc_bits  = input_quantA.bitwidth + input_quantB.bitwidth + 1
        acc_scale = input_quantA.scale * input_quantB.scale
        acc_quant = TensorQuant(
            bitwidth=acc_bits,
            signed=signed,
            scale=acc_scale,
            zeropt=0,
        )
        return get_hls_quant_type(acc_quant)

    def __get_quantizer(self, input_quantA, input_quantB, output_quant) -> str:
        if (
            self.__is_power_of_two(input_quantA.scale)
            and self.__is_power_of_two(input_quantB.scale)
            and self.__is_power_of_two(output_quant.scale)
        ):
            acc_scale = input_quantA.scale * input_quantB.scale
            shift = int(np.log2(acc_scale)) - int(np.log2(output_quant.scale))
            return (
                f"DequantQuantPo2<{shift}, "
                f"{self.__get_accumulator(input_quantA, input_quantB)}, "
                f"{get_hls_quant_type(output_quant)}>"
            )
        raise ValueError("Float quantization not supported for StreamingMatMul.")

    def __get_object_declaration(self, model) -> str:
        node = self.onnx_node
        transpose_a = self.get_nodeattr("transpose_a")

        input_quantA = get_custom_tensor_datatype(model, node.input[0])
        if input_quantA is None:
            raise ValueError(f"Input {node.input[0]} has no quantization info")
        input_quantB = get_custom_tensor_datatype(model, node.input[1])
        if input_quantB is None:
            raise ValueError(f"Input {node.input[1]} has no quantization info")
        output_quant = get_custom_tensor_datatype(model, node.output[0])
        if output_quant is None:
            raise ValueError(f"Output {node.output[0]} has no quantization info")

        input_shapeA = model.get_tensor_shape(node.input[0])
        input_shapeB = model.get_tensor_shape(node.input[1])
        if input_shapeA is None:
            raise ValueError(f"Input {node.input[0]} has no shape info")
        if input_shapeB is None:
            raise ValueError(f"Input {node.input[1]} has no shape info")

        # If transpose_a=1, the raw tensor is [B,H,S,D] but MatMul sees [B,H,D,S]
        # → swap IN_HEIGHT ↔ IN_WIDTH so the HPP loop bounds are correct
        if transpose_a:
            in_height = input_shapeA[3]   # D  (was last dim before transpose)
            in_width  = input_shapeA[2]   # S  (was second-to-last)
        else:
            in_height = input_shapeA[2]
            in_width  = input_shapeA[3]

        out_width = input_shapeB[3]
        in_ch     = input_shapeA[1]

        obj = cpp_object(
            "StreamingMatMul",
            f"{node.name}",
            template_args=[
                (f"{get_struct_type(input_quantA, self.get_nodeattr('in_word_array'))}", "TInputWordA"),
                (f"{get_hls_quant_type(input_quantA)}",                                 "TInputA"),
                (f"{get_struct_type(input_quantB, self.get_nodeattr('in_word_array'))}", "TInputWordB"),
                (f"{get_hls_quant_type(input_quantB)}",                                 "TInputB"),
                (f"{get_struct_type(output_quant, self.get_nodeattr('out_word_array'))}", "TOutputWord"),
                (f"{get_hls_quant_type(output_quant)}",                                  "TOutput"),
                (f"{self.__get_accumulator(input_quantA, input_quantB)}",                "TAcc"),
                (f"{self.__get_quantizer(input_quantA, input_quantB, output_quant)}",    "Quantizer"),
                (f"{in_height}",                                 "IN_HEIGHT"),
                (f"{in_width}",                                  "IN_WIDTH"),
                (f"{in_ch}",                                     "IN_CH"),
                (f"{out_width}",                                 "OUT_WIDTH"),
                (f"{self.get_nodeattr('width_unroll')}",         "W_PAR"),
                (f"{self.get_nodeattr('channel_unroll')}",       "CH_PAR"),
                (f"{1 if transpose_a else 0}",                   "TRANSPOSE_A"),
            ],
        )
        return obj.generate_declaration()

    def __get_run_call(self, hls_tag: int) -> str:
        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(
                ("i_dataA", "hls::stream"),
                ("i_dataB", "hls::stream"),
                ("o_data",  "hls::stream"),
            ),
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
                ("i_dataA", "hls::stream"),
                ("i_dataB", "hls::stream"),
                ("o_data",  "hls::stream"),
            ),
        )
        return run.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.input[1]),
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    # ------------------------------------------------------------------ #
    #  lower_to_hls                                                        #
    # ------------------------------------------------------------------ #
    def lower_to_hls(self, model: ModelWrapper, hls_tag: int):
        node = self.onnx_node
        output_quant = get_custom_tensor_datatype(model, node.output[0])
        if output_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{node.output[0]}' not found."
            )

        # input[0] is always the pre-transpose tensor when transpose_a=1
        input_names = [
            f"{self.__get_stream_name(node.input[0])}_{i}_"
            for i in range(self.get_nodeattr("in_stream_array"))
        ]
        input_names.extend(
            f"{self.__get_stream_name(node.input[1])}_{i}_"
            for i in range(self.get_nodeattr("in_stream_array"))
        )
        output_names = [
            f"{self.__get_stream_name(node.output[0])}_{i}_"
            for i in range(self.get_nodeattr("out_stream_array"))
        ]

        tensors_fifo_metadata = {
            out: TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(output_quant, self.get_nodeattr('out_word_array'))}",
                n_array=self.get_nodeattr("out_stream_array"),
            )
            for out in output_names
        }

        hls_kernel = HLSKernel.make_node(
            inputs=input_names,
            outputs=output_names,
            name=f"{node.name}_hls",
            domain="backend.custom_op",
            original_op_type="StreamingMatMul",
            hls_tag=hls_tag,
            hls_object_name=node.name,
            hls_variable_declarations="",
            hls_run_call=self.__get_run_call(hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )
        hls_tag += 1
        return [hls_kernel], [], tensors_fifo_metadata, hls_tag

    # ------------------------------------------------------------------ #
    #  Resource / latency estimates                                        #
    # ------------------------------------------------------------------ #
    def get_latency(self, model: ModelWrapper) -> int:
        shape = model.get_tensor_shape(self.onnx_node.input[0])
        if shape is None:
            raise ValueError(f"No shape for '{self.onnx_node.input[0]}'")
        unroll = self.get_nodeattr("channel_unroll") * self.get_nodeattr("width_unroll")
        return int(np.prod(shape)) // unroll

    def get_brams(self, model: ModelWrapper) -> int:
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        return 0

    def has_linebuffer(self) -> bool:
        return False

    def can_inherit_interface(self) -> bool:
        return True

    def inherit_interface(self, model: ModelWrapper, upstream: NodeInterface) -> None:
        self.set_nodeattr("in_stream_array",  upstream.out_stream_array)
        self.set_nodeattr("out_stream_array", upstream.out_stream_array)
        self.set_nodeattr("in_word_array",    upstream.out_word_array)
        self.set_nodeattr("out_word_array",   upstream.out_word_array)
        self.set_nodeattr("channel_unroll",   upstream.out_word_array)
        self.set_nodeattr("width_unroll",     upstream.out_stream_array)
