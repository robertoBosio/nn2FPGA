import numpy as np
import onnxruntime as rt
from onnxscript.rewriter import pattern
from onnxscript import ir
from onnx import TensorProto, helper
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import TensorQuant, get_custom_tensor_datatype
from backend.core.tensor_fifo import TensorFifo
from backend.custom_op.hlskernel import HLSKernel
from backend.custom_op.op_base import NN2FPGAOp, NodeInterface
from backend.custom_op.register_rewrite_rule import register_rules, PRule
from backend.util.codegen_utils import (
    cpp_function,
    cpp_object,
    get_struct_type,
    get_hls_quant_type,
)
import logging


logger = logging.getLogger(__name__)


class StreamingMatMul(NN2FPGAOp):
    """Node implementing the MatMul operation.

    Matches and absorbs:

        a ──► Transpose([0,1,3,2]) ──► Quant(outer) ──┐
                                                        MatMul ──► Quant(out)
                                          b ────────────┘

    `a` is the pre-Transpose tensor (e.g. output of upstream Quant),
    with physical shape [B, CH, H, W].  The HLS kernel sees it as
    [B, CH, W, H] after the absorbed Transpose.
    """

    # ------------------------------------------------------------------ #
    #  Pattern B – Quant(A) → Transpose → Quant(outer) → MatMul → Quant(out)
    # ------------------------------------------------------------------ #
    @staticmethod
    def pattern(
        op,
        a,
        b,
        outer_scale,
        outer_zeropt,
        outer_bitwidth,
    ):
        """
        Matches (free vars `a`, `b` are already-quantised tensors):

            a ──► Transpose([0,1,3,2]) ──► Quant(outer) ──┐
                                                            MatMul ──► Quant(out)
                                              b ────────────┘
        """
        a_t = op.Transpose(a, perm=[0, 1, 3, 2])
        a_outer = op.Quant(
            a_t,
            outer_scale,
            outer_zeropt,
            outer_bitwidth,
            _allow_other_attributes=True,
            _domain="qonnx.custom_op.general",
        )
        return op.MatMul(a_outer, b, _allow_other_attributes=True)
    
    @staticmethod
    def _condition(context, a, **_):
        """
        Accept the rule only when `a` (the pre-Transpose tensor) has a
        statically known 4-D shape, so HLS shape inference can run without
        a prior shape-propagation pass.
        """
        if not isinstance(a, ir.Value):
            return False
        s = a.shape
        return s is not None and len(s) == 4 and all(d is not None for d in s)

    @staticmethod
    def rewrite(
        op,
        a,
        b,
        **_,
    ):
        """
        Replace the absorbed sub-graph with a single StreamingMatMul.

        `a`  — pre-Transpose tensor, physical shape [B, CH, H, W]
        `b`  — other MatMul input
        `transposed_a=1` — HLS kernel reads A as [B, CH, W, H]
        """
        logger.debug(
            "StreamingMatMul: absorbing Transpose + Quant(outer) + MatMul."
        )
        return op.StreamingMatMul(
            a,
            b,
            _domain="backend.custom_op",
            transposed_a=1,
        )

    @register_rules
    def register_rules():
        return [
            PRule(
                pattern.RewriteRule(
                    StreamingMatMul.pattern,
                    StreamingMatMul.rewrite,
                    StreamingMatMul._condition,
                ),
                priority=1,
            )
        ]

    # ------------------------------------------------------------------ #
    #  Node attributes
    # ------------------------------------------------------------------ #
    def get_nodeattr_types(self):
        return {
            "channel_unroll":    ("i", False, 1),
            "width_unroll":      ("i", False, 1),
            "in_stream_array":   ("i", False, 1),
            "out_stream_array":  ("i", False, 1),
            "in_word_array":     ("i", False, 1),
            "out_word_array":    ("i", False, 1),
            # Always 1: A arrives pre-Transpose, physical shape [B, CH, H, W].
            # The HLS kernel iterates over the logical shape [B, CH, W, H].
            "transposed_a":      ("i", False, 1),
            # Forwarded from the absorbed Quant(out)
            "out_signed":        ("i", False, 0),
            "out_narrow":        ("i", False, 0),
            "out_rounding_mode": ("s", False, "ROUND"),
        }

    # ------------------------------------------------------------------ #
    #  Shape / type inference (QONNX hooks)
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

    # ------------------------------------------------------------------ #
    #  Execution (functional reference via ONNX Runtime)
    # ------------------------------------------------------------------ #
    def execute_node(self, context, graph):
        node = self.onnx_node

        # A is stored pre-Transpose: apply it before the reference MatMul.
        inpA_values = np.transpose(context[node.input[0]], (0, 1, 3, 2))
        inpB_values = context[node.input[1]]
        oshape      = context[node.output[0]].shape

        node_mm = helper.make_node(
            "MatMul",
            inputs=node.input,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
        )
        inpA = helper.make_tensor_value_info(
            node.input[0], TensorProto.FLOAT, inpA_values.shape
        )
        inpB = helper.make_tensor_value_info(
            node.input[1], TensorProto.FLOAT, inpB_values.shape
        )
        outp = helper.make_tensor_value_info(
            node.output[0], TensorProto.FLOAT, oshape
        )
        graph_mm = helper.make_graph(
            nodes=[node_mm],
            name="single-matmul-exec",
            inputs=[inpA, inpB],
            outputs=[outp],
        )
        opset_imports = [helper.make_opsetid("", self.onnx_opset_version)]
        model_mm = qonnx_make_model(graph_mm, opset_imports=opset_imports)
        idict = {node.input[0]: inpA_values, node.input[1]: inpB_values}

        sess   = rt.InferenceSession(model_mm.SerializeToString())
        result = np.array(sess.run(None, idict)[0])
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def __get_stream_name(self, name: str) -> str:
        return f"{name}_stream"

    def _get_logical_shape_A(self, model=None) -> list:
        """Return A's logical shape as [B, CH, IN_H, IN_W].

        node.input[0] has physical shape [B, CH, H, W] (pre-Transpose).
        After the absorbed Transpose([0,1,3,2]) the HLS kernel sees
        [B, CH, W, H], so dims 2 and 3 are swapped.

        Example (from the graph in the attached images):
            physical a : [1, 2, 32, 400]
            logical  A : [1, 2, 400, 32]  → IN_HEIGHT=400, IN_WIDTH=32
        """
        src = model if model is not None else getattr(self, "_model_ref", None)
        if src is None:
            return None
        raw = src.get_tensor_shape(self.onnx_node.input[0])
        if raw is None or len(raw) < 4:
            return raw
        s = list(raw)
        s[2], s[3] = raw[3], raw[2]   # H ↔ W
        return s

    def __get_accumulator(
        self,
        input_quantA: TensorQuant,
        input_quantB: TensorQuant,
    ) -> str:
        """acc_bits = bwA + bwB + ceil(log2(IN_WIDTH)) + 1 (sign bit)."""
        logical  = self._get_logical_shape_A()
        in_width = int(logical[3]) if logical is not None and len(logical) >= 4 else 1

        acc_bits = (
            input_quantA.bitwidth
            + input_quantB.bitwidth
            + int(np.ceil(np.log2(max(in_width, 1))))
            + 1
        )
        acc_quant = TensorQuant(
            bitwidth=acc_bits,
            signed=input_quantA.signed or input_quantB.signed,
            scale=input_quantA.scale * input_quantB.scale,
            zeropt=0,
        )
        return get_hls_quant_type(acc_quant)

    def __get_quantizer(
        self,
        input_quantA: TensorQuant,
        input_quantB: TensorQuant,
        output_quant: TensorQuant,
    ) -> str:
        """Power-of-two requantizer: accumulator scale → output scale."""
        def is_po2(v):
            return v > 0 and float(np.log2(v)).is_integer()

        if (
            is_po2(input_quantA.scale)
            and is_po2(input_quantB.scale)
            and is_po2(output_quant.scale)
        ):
            acc_scale = input_quantA.scale * input_quantB.scale
            shift = int(np.log2(acc_scale)) - int(np.log2(output_quant.scale))
            return (
                f"DequantQuantPo2<{shift}, "
                f"{self.__get_accumulator(input_quantA, input_quantB)}, "
                f"{get_hls_quant_type(output_quant)}>"
            )
        raise ValueError(
            "Float quantization is currently not supported for StreamingMatMul."
        )

    def __get_variable_declaration(self, model) -> str:
        return ""

    def __get_object_declaration(self, model) -> str:
        node = self.onnx_node

        input_quantA = get_custom_tensor_datatype(model, node.input[0])
        if input_quantA is None:
            raise ValueError(f"Input {node.input[0]} has no quantization info")
        input_quantB = get_custom_tensor_datatype(model, node.input[1])
        if input_quantB is None:
            raise ValueError(f"Input {node.input[1]} has no quantization info")
        output_quant = get_custom_tensor_datatype(model, node.output[0])
        if output_quant is None:
            raise ValueError(f"Output {node.output[0]} has no quantization info")

        if model.get_tensor_shape(node.input[0]) is None:
            raise ValueError(f"Input {node.input[0]} has no shape info")
        if model.get_tensor_shape(node.input[1]) is None:
            raise ValueError(f"Input {node.input[1]} has no shape info")
        output_shape = model.get_tensor_shape(node.output[0])
        if output_shape is None:
            raise ValueError(f"Output {node.output[0]} has no shape info")

        self._model_ref = model   # used by __get_accumulator via _get_logical_shape_A
        logical_A = self._get_logical_shape_A(model)

        in_ch     = logical_A[1]
        in_height = logical_A[2]   # W of raw A  (after absorbed Transpose)
        in_width  = logical_A[3]   # H of raw A  (reduction dimension)
        out_width = output_shape[3]

        logger.debug(
            "%s: physical A %s → logical [B=%s, CH=%s, H=%s, W=%s], OUT_W=%s",
            node.name, model.get_tensor_shape(node.input[0]),
            logical_A[0], in_ch, in_height, in_width, out_width,
        )

        acc_type  = self.__get_accumulator(input_quantA, input_quantB)
        quantizer = self.__get_quantizer(input_quantA, input_quantB, output_quant)

        mm_obj = cpp_object(
            "StreamingMatMul",
            f"{node.name}",
            template_args=[
                (f"{get_struct_type(input_quantA, self.get_nodeattr('in_word_array'))}", "TInputWordA"),
                (f"{get_hls_quant_type(input_quantA)}",                                 "TInputA"),
                (f"{get_struct_type(input_quantB, self.get_nodeattr('in_word_array'))}", "TInputWordB"),
                (f"{get_hls_quant_type(input_quantB)}",                                 "TInputB"),
                (f"{get_struct_type(output_quant,  self.get_nodeattr('out_word_array'))}", "TOutputWord"),
                (f"{get_hls_quant_type(output_quant)}",                                   "TOutput"),
                (f"{acc_type}",                                                            "TAcc"),
                (f"{quantizer}",                                                           "Quantizer"),
                (f"{in_height}",                                                           "IN_HEIGHT"),
                (f"{in_width}",                                                            "IN_WIDTH"),
                (f"{out_width}",                                                           "OUT_WIDTH"),
                (f"{in_ch}",                                                               "IN_CH"),
                (f"{self.get_nodeattr('width_unroll')}",                                   "W_PAR"),
                (f"{self.get_nodeattr('channel_unroll')}",                                 "CH_PAR"),
            ],
        )
        return mm_obj.generate_declaration()

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
        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=(
                ("i_dataA", "hls::stream"),
                ("i_dataB", "hls::stream"),
                ("o_data",  "hls::stream"),
            ),
        )
        return step.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.input[1]),
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    # ------------------------------------------------------------------ #
    #  HLS lowering
    # ------------------------------------------------------------------ #
    def lower_to_hls(self, model: ModelWrapper, hls_tag: int):
        node = self.onnx_node

        output_quant = get_custom_tensor_datatype(model, node.output[0])
        if output_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{node.output[0]}' not found in model."
            )

        input_names = [
            f"{self.__get_stream_name(node.input[0])}_{i}_"
            for i in range(self.get_nodeattr("in_stream_array"))
        ]
        input_names.extend([
            f"{self.__get_stream_name(node.input[1])}_{i}_"
            for i in range(self.get_nodeattr("in_stream_array"))
        ])
        output_names = [
            f"{self.__get_stream_name(node.output[0])}_{i}_"
            for i in range(self.get_nodeattr("out_stream_array"))
        ]

        tensors_fifo_metadata = {}
        for output in output_names:
            tensors_fifo_metadata[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(output_quant, self.get_nodeattr('out_word_array'))}",
                n_array=self.get_nodeattr("out_stream_array"),
            )

        hls_kernel = HLSKernel.make_node(
            inputs=input_names,
            outputs=output_names,
            name=f"{node.name}_hls",
            domain="backend.custom_op",
            original_op_type="StreamingMatMul",
            hls_tag=hls_tag,
            hls_object_name=node.name,
            hls_variable_declarations=self.__get_variable_declaration(model),
            hls_run_call=self.__get_run_call(hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )

        hls_tag += 1
        return [hls_kernel], [], tensors_fifo_metadata, hls_tag

    # ------------------------------------------------------------------ #
    #  Resource / latency estimates
    # ------------------------------------------------------------------ #
    def get_latency(self, model: ModelWrapper) -> int:
        """IN_HEIGHT × OUT_WIDTH × (IN_WIDTH / W_PAR) × (IN_CH / CH_PAR).

        Computed on the logical shape so the formula is invariant to
        the absorbed Transpose.
        """
        logical_A    = self._get_logical_shape_A(model)
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if logical_A is None or output_shape is None:
            raise ValueError(
                f"Shape not found for latency estimate of {self.onnx_node.name}"
            )

        in_height = logical_A[2]
        in_width  = logical_A[3]
        in_ch     = logical_A[1]
        out_width = output_shape[3]
        w_par     = self.get_nodeattr("width_unroll")
        ch_par    = self.get_nodeattr("channel_unroll")

        return int(in_height * out_width * (in_width // w_par) * (in_ch // ch_par))

    def get_brams(self, model: ModelWrapper) -> int:
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        return int(self.get_nodeattr("width_unroll") * self.get_nodeattr("channel_unroll"))

    def has_linebuffer(self) -> bool:
        return False

    def can_inherit_interface(self):
        return True

    def inherit_interface(self, model: ModelWrapper, upstream: NodeInterface) -> None:
        self.set_nodeattr("in_stream_array",  upstream.out_stream_array)
        self.set_nodeattr("out_stream_array", upstream.out_stream_array)
        self.set_nodeattr("in_word_array",    upstream.out_word_array)
        self.set_nodeattr("out_word_array",   upstream.out_word_array)
        self.set_nodeattr("channel_unroll",   upstream.out_word_array)
        self.set_nodeattr("width_unroll",     upstream.out_stream_array)