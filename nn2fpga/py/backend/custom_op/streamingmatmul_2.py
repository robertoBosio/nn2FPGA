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
    """Node implementing the MatMul operation."""

    @staticmethod
    def pattern(op, a, b):
        return op.MatMul(a, b, _allow_other_attributes=True)

    @staticmethod
    def rewrite(op, a, b):
        return op.StreamingMatMul(
            a,
            b,
            _domain="backend.custom_op",
        )

    @register_rules
    def register_rules():
        return [pattern.RewriteRule(StreamingMatMul.pattern, StreamingMatMul.rewrite)]

    def get_nodeattr_types(self):
        return {
            # Parallelism unroll factors
            "channel_unroll": ("i", False, 1),  # CH_PAR
            "width_unroll":   ("i", False, 1),  # W_PAR
            # Stream array sizing
            "in_stream_array":  ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array":    ("i", False, 1),
            "out_word_array":   ("i", False, 1),
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
        node = self.onnx_node
        node_mm = helper.make_node(
            "MatMul",
            inputs=node.input,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
        )
        inpA_values = context[node.input[0]]
        inpB_values = context[node.input[1]]
        oshape = context[node.output[0]].shape

        inpA = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, inpA_values.shape)
        inpB = helper.make_tensor_value_info(node.input[1], TensorProto.FLOAT, inpB_values.shape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)

        graph_mm = helper.make_graph(
            nodes=[node_mm],
            name="single-matmul-exec",
            inputs=[inpA, inpB],
            outputs=[outp],
        )
        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        model_mm = qonnx_make_model(graph_mm, opset_imports=opset_imports)
        idict = {node.input[0]: inpA_values, node.input[1]: inpB_values}

        sess = rt.InferenceSession(model_mm.SerializeToString())
        result = np.array(sess.run(None, idict)[0])
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    # helper functions to generate HLS object declaration and run call, 
    # using node attributes and input/output quantization info
    def __get_stream_name(self, name: str) -> str: # stream name for an input/output tensor, e.g. "input0_stream"
        return f"{name}_stream"
    # Accumulator type and quantizer generation based on input/output quantization info.
    def __get_accumulator(self, input_quantA: TensorQuant, input_quantB: TensorQuant) -> str:
        """
        Accumulator type: signed, wide enough for A*B dot-product.
        acc_bits = bitwidth_A + bitwidth_B + log2(IN_WIDTH) + 1 (sign)
        """
        node = self.onnx_node
        input_shape = self._get_input_shape_A()
        in_width = int(input_shape[3]) if input_shape is not None and len(input_shape) >= 4 else 1

        acc_bits = (
            input_quantA.bitwidth
            + input_quantB.bitwidth
            + int(np.ceil(np.log2(max(in_width, 1))))
            + 1  # sign bit
        )
        signed = input_quantA.signed or input_quantB.signed
        acc_quant = TensorQuant(
            bitwidth=acc_bits,
            signed=signed,
            scale=input_quantA.scale * input_quantB.scale,
            zeropt=0,
        )
        return get_hls_quant_type(acc_quant)
    # Quantizer generation: from accumulator scale to output scale. 
    # Only power-of-two scales are supported, implemented as bit shifts in HLS.
    def __get_quantizer(
        self,
        input_quantA: TensorQuant,
        input_quantB: TensorQuant,
        output_quant: TensorQuant,
    ) -> str:
        """
        Requantizer from accumulator scale (scaleA * scaleB) to output scale.
        Only power-of-two scales are supported.
        """
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
        else:
            raise ValueError(
                "Float quantization is currently not supported for StreamingMatMul."
            )
    #  Variable declarations for the HLS object (e.g. local buffers) can be generated here if needed.
    def __get_variable_declaration(self, model) -> str:
        return ""
    # Helper to get input shape for accumulator bitwidth calculation. 
    # Assumes input A has shape [B, IN_CH, IN_HEIGHT, IN_WIDTH].
    def _get_input_shape_A(self):
        return self._model_ref.get_tensor_shape(self.onnx_node.input[0]) \
            if hasattr(self, "_model_ref") else None
    # Object declaration generation based on quantization and shape info, using cpp_object helper.
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

        input_shapeA = model.get_tensor_shape(node.input[0])
        if input_shapeA is None:
            raise ValueError(f"Input {node.input[0]} has no shape info")
        input_shapeB = model.get_tensor_shape(node.input[1])
        if input_shapeB is None:
            raise ValueError(f"Input {node.input[1]} has no shape info")
        output_shape = model.get_tensor_shape(node.output[0])
        if output_shape is None:
            raise ValueError(f"Output {node.output[0]} has no shape info")

        # Shape mapping (NCHW-like 4D):
        # A: [B, IN_CH, IN_HEIGHT, IN_WIDTH]
        # B: [B, IN_CH, IN_WIDTH,  OUT_WIDTH]  (weight matrix, transposed)
        # out: [B, IN_CH, IN_HEIGHT, OUT_WIDTH]
        in_height  = input_shapeA[2]
        in_width   = input_shapeA[3]
        in_ch      = input_shapeA[1]
        out_width  = output_shape[3]

        self._model_ref = model  # for __get_accumulator
        acc_type = self.__get_accumulator(input_quantA, input_quantB)
        quantizer = self.__get_quantizer(input_quantA, input_quantB, output_quant)

        mm_obj = cpp_object(
            "StreamingMatMul",
            f"{node.name}",
            template_args=[
                (f"{get_struct_type(input_quantA, self.get_nodeattr('in_word_array'))}", "TInputWordA"),
                (f"{get_hls_quant_type(input_quantA)}",                                  "TInputA"),
                (f"{get_struct_type(input_quantB, self.get_nodeattr('in_word_array'))}", "TInputWordB"),
                (f"{get_hls_quant_type(input_quantB)}",                                  "TInputB"),
                (f"{get_struct_type(output_quant, self.get_nodeattr('out_word_array'))}", "TOutputWord"),
                (f"{get_hls_quant_type(output_quant)}",                                  "TOutput"),
                (f"{acc_type}",                                                           "TAcc"),
                (f"{quantizer}",                                                          "Quantizer"),
                (f"{in_height}",                                                          "IN_HEIGHT"),
                (f"{in_width}",                                                           "IN_WIDTH"),
                (f"{in_ch}",                                                              "IN_CH"),
                (f"{out_width}",                                                          "OUT_WIDTH"),
                (f"{self.get_nodeattr('width_unroll')}",                                  "W_PAR"),
                (f"{self.get_nodeattr('channel_unroll')}",                                "CH_PAR"),
            ],
        )
        return mm_obj.generate_declaration()
    # Run call generation using cpp_function helper, passing stream names and HLS tag as arguments.
    def __get_run_call(self, hls_tag: int) -> str:
        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(
                (f"i_dataA", f"hls::stream"),
                (f"i_dataB", f"hls::stream"),
                (f"o_data",  f"hls::stream"),
            ),
        )
        return run.generate_call(
            [hls_tag],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.input[1]),
            self.__get_stream_name(self.onnx_node.output[0]),
        )
    # Step call generation, similar to run call but without HLS tag argument.
    def __get_step_call(self) -> str:
        run = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=(
                (f"i_dataA", f"hls::stream"),
                (f"i_dataB", f"hls::stream"),
                (f"o_data",  f"hls::stream"),
            ),
        )
        return run.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.input[1]),
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    # ------------------------------------------------------------------ #
    #  HLS lowering
    # ------------------------------------------------------------------ #
   # HLS lowering: generate HLSKernel node with appropriate template arguments, variable declarations, and run call.
    def lower_to_hls(self, model: ModelWrapper, hls_tag: int):
        node = self.onnx_node

        output_quant = get_custom_tensor_datatype(model, node.output[0])
        if output_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{node.output[0]}' not found in model."
            )

        # Input A streams (one per CH_PAR slot)
        input_names = [
            f"{self.__get_stream_name(node.input[0])}_{i}_"
            for i in range(self.get_nodeattr("in_stream_array"))
        ]
        # Input B streams
        input_names.extend([
            f"{self.__get_stream_name(node.input[1])}_{i}_"
            for i in range(self.get_nodeattr("in_stream_array"))
        ])
        # Single output stream (mat_out is scalar hls::stream&, not array)
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
    # Latency estimation based on input/output shapes and parallelism unroll factors,
    # matching ActorStatus firing count in StepState::init().
    def get_latency(self, model: ModelWrapper) -> int:
        """
        Latency = IN_HEIGHT * OUT_WIDTH * (IN_WIDTH / W_PAR) * (IN_CH / CH_PAR)
        matching ActorStatus firing count in StepState::init().
        """
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if input_shape is None or output_shape is None:
            raise ValueError(f"Shape not found for latency estimate of {self.onnx_node.name}")

        in_height  = input_shape[2]
        in_width   = input_shape[3]
        in_ch      = input_shape[1]
        out_width  = output_shape[3]
        w_par  = self.get_nodeattr("width_unroll")
        ch_par = self.get_nodeattr("channel_unroll")

        return int(in_height * out_width * (in_width // w_par) * (in_ch // ch_par))
    #  BRAM usage comes from local_A[IN_CH][IN_WIDTH] and local_B[OUT_WIDTH][IN_CH][IN_WIDTH].
    #  Approximated as 0 BRAMs (implemented in LUTRAM/registers
    def get_brams(self, model: ModelWrapper) -> int:
        """
        BRAM usage comes from local_A[IN_CH][IN_WIDTH] and local_B[OUT_WIDTH][IN_CH][IN_WIDTH].
        Approximated as 0 BRAMs (implemented in LUTRAM/registers with ARRAY_PARTITION).
        """
        return 0
    # DSP usage: one multiplier per W_PAR * CH_PAR lane.
    def get_dsps(self, model: ModelWrapper) -> int:
        """
        DSP usage: one multiplier per W_PAR * CH_PAR lane.
        """
        w_par  = self.get_nodeattr("width_unroll")
        ch_par = self.get_nodeattr("channel_unroll")
        return int(w_par * ch_par)
    # StreamingMatMul does not use line buffers.
    def has_linebuffer(self) -> bool:
        return False
    # This operator can inherit the streaming interface (stream widths and unroll factors) from an upstream node, if available.
    def can_inherit_interface(self):
        return True
    # Interface inheritance: copy stream array sizes and unroll factors from upstream node's output to this node's input/output.
    def inherit_interface(self, model: ModelWrapper, upstream: NodeInterface) -> None:
        self.set_nodeattr("in_stream_array",  upstream.out_stream_array)
        self.set_nodeattr("out_stream_array", upstream.out_stream_array)
        self.set_nodeattr("in_word_array",    upstream.out_word_array)
        self.set_nodeattr("out_word_array",   upstream.out_word_array)
        self.set_nodeattr("channel_unroll",   upstream.out_word_array)
        self.set_nodeattr("width_unroll",     upstream.out_stream_array)
