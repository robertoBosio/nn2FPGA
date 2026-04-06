import numpy as np
import onnxruntime as rt
from onnxscript.rewriter import pattern
from onnx import TensorProto, helper
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.core.tensor_quant import TensorQuant, get_custom_tensor_datatype
from nn2fpga.compiler.core.tensor_fifo import TensorFifo
from nn2fpga.compiler.custom_op.hlskernel import HLSKernel
from nn2fpga.compiler.custom_op.op_base import NN2FPGAOp, NodeInterface, DSECapable
from nn2fpga.compiler.custom_op.register_rewrite_rule import register_rules
from nn2fpga.compiler.utils.codegen_utils import (
    cpp_function,
    cpp_object,
    get_struct_type,
    get_hls_quant_type,
)
from dataclasses import dataclass
from onnxscript import ir
from onnx_ir import convenience as ir_convenience
from onnxscript.rewriter import pattern
from nn2fpga.compiler.utils.board_util import packing_feature

class StreamingMul(NN2FPGAOp, DSECapable):
    """Node implementing the Mul operation."""

    @dataclass(frozen=True)
    class DSEPoint:
        channel_unroll: int
        width_unroll: int

        # optional helpers to interop with old code / ONNX storage
        def to_dict(self) -> dict:
            return {
                "channel_unroll": self.channel_unroll,
                "width_unroll": self.width_unroll,
            }

        @staticmethod
        def from_dict(d: dict) -> "StreamingMul.DSEPoint":
            return StreamingMul.DSEPoint(
                channel_unroll=d["channel_unroll"],
                width_unroll=d["width_unroll"],
            )

    @staticmethod
    def pattern(op, a, b):
        return op.Mul(a, b, _allow_other_attributes=True)

    @staticmethod
    def _condition(context, a, b, **_):
        # Only rewrite if A and B are not constants
        return ir_convenience.get_const_tensor(b) is None and ir_convenience.get_const_tensor(a) is None

    @staticmethod
    def rewrite(op, a, b):
        return op.StreamingMul(
            a,
            b,
            _domain="nn2fpga.compiler.custom_op",
        )

    @register_rules
    def register_rules():
        return [pattern.RewriteRule(StreamingMul.pattern, StreamingMul.rewrite, StreamingMul._condition)]

    def get_nodeattr_types(self):
        return {
            # Custom attributes for unroll factors
            "channel_unroll": ("i", False, 1),
            "width_unroll": ("i", False, 1),
            # Custom attributes for input/output streams
            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),
            "activation": ("s", False, "NoOp"),  # Activation function
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node

        return helper.make_node(
            "Mul",
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
            "Mul",
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
        """ Get the internal cpp variables of the StreamingMul node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __is_power_of_two(self, value) -> bool:
        """Check if a value is a power of two."""
        return value > 0 and float(np.log2(value)).is_integer()

    def __get_multiplier(self, input_quantA, input_quantB):
        """ Returns the multiplier type for the Mul operation. """

        # Determine signedness and bitwidth
        signed = input_quantA.signed or input_quantB.signed
        mul_bits = input_quantA.bitwidth + input_quantB.bitwidth

        # In case the input signedness is different, we need an extra bit, as the accumulator will be signed.
        if (input_quantA.signed != input_quantB.signed):
            mul_bits += 1

        mul_quant = TensorQuant(
            bitwidth=mul_bits,
            signed=signed,
            scale=input_quantA.scale,
            zeropt=input_quantA.zeropt
        )
        return f"{get_hls_quant_type(mul_quant)}"

    def __get_quantizer(self, input_quantA, input_quantB, output_quant) -> str:
        """ Returns the quantizer type for the Mul operation. """

        if (
            self.__is_power_of_two(input_quantA.scale)
            and self.__is_power_of_two(input_quantB.scale)
            and self.__is_power_of_two(output_quant.scale)
        ):
            shift = -1 * (
                int(np.log2(input_quantA.scale))
                + int(np.log2(input_quantB.scale))
                - int(np.log2(output_quant.scale))
            )
            return f"DequantQuantPo2<{shift}, {self.__get_multiplier(input_quantA, input_quantB)}, {get_hls_quant_type(output_quant)}>"
        else:
            raise ValueError(
                "Float quantization is currently not supported for StreamingMul."
            )

    def __get_activation(self, input_quantA, input_quantB) -> str:
        """ Returns the activation functor for the StreamingMul operation. """

        activation = self.get_nodeattr("activation")
        if activation == "NoOp":
            return f"DequantQuantEqual<{self.__get_multiplier(input_quantA, input_quantB)}>"
        elif activation == "ReLU":
            return f"ReLU<{self.__get_multiplier(input_quantA, input_quantB)}>"
        else:
            raise ValueError(
                f"Unsupported activation function '{activation}' for StreamingMul."
            )

    def __get_object_declaration(self, model) -> cpp_object:

        input_quantA = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if (input_quantA is None):
            raise ValueError(f"Input {self.onnx_node.input[0]} has no quantization info")
        input_quantB = get_custom_tensor_datatype(model, self.onnx_node.input[1])
        if (input_quantB is None):
            raise ValueError(f"Input {self.onnx_node.input[1]} has no quantization info")
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if (output_quant is None):
            raise ValueError(f"Output {self.onnx_node.output[0]} has no quantization info")

        input_shapeA = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shapeA is None:
            raise ValueError(f"Input {self.onnx_node.input[0]} has no shape info")
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Output {self.onnx_node.output[0]} has no shape info")

        StreamingMul = cpp_object(
            "StreamingMul",
            f"{self.onnx_node.name}",
            template_args=[
                (
                    f"{get_struct_type(input_quantA, self.get_nodeattr('in_word_array'))}",
                    f"TInputWordA",
                ),
                (
                    f"{get_hls_quant_type(input_quantA)}",
                    f"TInputA",
                ),
                (
                    f"{get_struct_type(input_quantB, self.get_nodeattr('in_word_array'))}",
                    f"TInputWordB",
                ),
                (
                    f"{get_hls_quant_type(input_quantB)}",
                    f"TInputB",
                ),
                (
                    f"{get_struct_type(output_quant, self.get_nodeattr('out_word_array'))}",
                    f"TOutputWord",
                ),
                (
                    f"{get_hls_quant_type(output_quant)}",
                    f"TOutput",
                ),
                (
                    f"{self.__get_multiplier(input_quantA, input_quantB)}",
                    f"TMul",
                ),
                (self.__get_activation(input_quantA, input_quantB), "Activation"),
                (
                    f"{self.__get_quantizer(input_quantA, input_quantB, output_quant)}",
                    f"Quantizer",
                ),
                (f"{input_shapeA[2]}", "IN_HEIGHT"),
                (f"{input_shapeA[3]}", "IN_WIDTH"),
                (f"{input_shapeA[1]}", "IN_CH"),
                (f"{self.get_nodeattr('width_unroll')}", "W_PAR"),
                (f"{self.get_nodeattr('channel_unroll')}", "CH_PAR"),
            ]
        )

        return StreamingMul.generate_declaration()

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

    def __current_dse_point(self) -> "StreamingMul.DSEPoint":
        """ Returns the current DSE point of the StreamingMul operation. """
        return StreamingMul.DSEPoint(
            channel_unroll=self.get_nodeattr("channel_unroll"),
            width_unroll=self.get_nodeattr("width_unroll"),
        )

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int) -> None:
        """Lower the node to HLS code."""

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model."
            )

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
                hls_type=f"{get_struct_type(output_quant, self.get_nodeattr('out_word_array'))}",
                n_array=self.get_nodeattr("out_stream_array"),
            )

        hls_kernel = HLSKernel.make_node(
            inputs=input_names,
            outputs=output_names,
            name=f"{self.onnx_node.name}_hls",
            domain="nn2fpga.compiler.custom_op",
            original_op_type="StreamingMul",
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
        """ Estimate the latency of the StreamingMul operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated latency in clock cycles.
        """
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model.")

        unroll_factor = self.get_nodeattr("channel_unroll") * self.get_nodeattr("width_unroll")
        return np.prod(input_shape) // unroll_factor

    def get_brams(self, model: ModelWrapper) -> int:
        """ Estimate the BRAM usage of the StreamingMul operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated BRAM usage.
        """
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        """Estimate the DSP usage of the StreamingMul operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated DSP usage.
        """
        silvia_packing = model.get_metadata_prop("silvia_packing") == "true"

        inpA_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if inpA_quant is None:
            raise ValueError(
                f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model."
            )
        inpB_quant = get_custom_tensor_datatype(model, self.onnx_node.input[1])
        if inpB_quant is None:
            raise ValueError(
                f"Tensor quantization for input '{self.onnx_node.input[1]}' not found in model."
            )

        # Retrieve current parallelization attributes if not provided.
        point = self.__current_dse_point()

        mac_per_dsp, _ = packing_feature(
            (inpA_quant.bitwidth, inpB_quant.bitwidth),
            [point.width_unroll, point.channel_unroll],
            silvia_packing,
        )
        MACs = point.channel_unroll * point.width_unroll

        return MACs // mac_per_dsp

    def has_linebuffer(self) -> bool:
        """ Check if the StreamingMul operation requires a line buffer.
        Returns:
            bool: True if Line Buffering is required, False otherwise.
        """
        return False

    def get_dse_points(self, model: ModelWrapper) -> list["StreamingMul.DSEPoint"]:
        """Generate the list of valid DSE points for the StreamingMul operation."""

        def divisors(n, clip):
            return [i for i in range(1, n + 1) if (n % i == 0 and i <= clip)]

        inputA_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if inputA_quant is None:
            raise ValueError(
                f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model."
            )

        inputB_quant = get_custom_tensor_datatype(model, self.onnx_node.input[1])
        if inputB_quant is None:
            raise ValueError(
                f"Tensor quantization for input '{self.onnx_node.input[1]}' not found in model."
            )

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model."
            )

        inputA_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if inputA_shape is None:
            raise ValueError(
                f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model."
            )
        inputA_shape = inputA_shape + [1] * (4 - len(inputA_shape))  # Ensure 4D shape.
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(
                f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model."
            )
        output_shape = output_shape + [1] * (4 - len(output_shape))  # Ensure 4D shape.

        # As of now, kernel height and width are completely unrolled.
        DSE_points = []
        for channel_unroll in divisors(inputA_shape[1], inputA_shape[1]):
            for width_unroll in divisors(output_shape[3], output_shape[3]):
                # Check dimension of input streams
                if (inputA_quant.bitwidth * channel_unroll) > 4096:
                    continue

                if (inputB_quant.bitwidth * channel_unroll) > 4096:
                    continue

                # Check dimension of output streams
                if (output_quant.bitwidth * channel_unroll) > 4096:
                    continue

                # Heuristic to spread unrolling across dimensions
                # if (width_unroll > 4 or channel_unroll > 5):
                #     continue

                DSE_points.append(self.DSEPoint(channel_unroll, width_unroll))

        return DSE_points

    def apply_point(self, model: ModelWrapper, point: "StreamingMul.DSEPoint"):
        """ Set the parallelization attributes for the StreamingMul operation.
        Args:
            point (StreamingMul.DSEPoint): The DSE point containing the unrolling parameters.
        """
        self.set_nodeattr("channel_unroll", point.channel_unroll)
        self.set_nodeattr("width_unroll", point.width_unroll)

        self.set_nodeattr("in_stream_array", point.width_unroll)
        self.set_nodeattr("out_stream_array", point.width_unroll)
        self.set_nodeattr("in_word_array", point.channel_unroll)
        self.set_nodeattr("out_word_array", point.channel_unroll)
