from attr import dataclass
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

class StreamingConcat(NN2FPGAOp, DSECapable):
    """Node implementing the Concat operation."""

    @dataclass(frozen=True)
    class DSEPoint:
        channel_unroll: int
        width_unroll: int

        def to_dict(self) -> dict:
            return {
                "channel_unroll": self.channel_unroll,
                "width_unroll": self.width_unroll,
            }

        @staticmethod
        def from_dict(d: dict) -> "StreamingConcat.DSEPoint":
            return StreamingConcat.DSEPoint(
                channel_unroll=d["channel_unroll"],
                width_unroll=d["width_unroll"],
            )

    @staticmethod
    def pattern(op, a, b, axis):
        return op.Concat(a, b, axis=axis, _allow_other_attributes=True)

    @staticmethod
    def rewrite(op, a, b, axis):
        return op.StreamingConcat(
            a,
            b,
            axis=axis,
            _domain="nn2fpga.compiler.custom_op",
        )

    @register_rules
    def register_rules():
        return [pattern.RewriteRule(StreamingConcat.pattern, StreamingConcat.rewrite)]

    def get_nodeattr_types(self):
        return {
            "axis": ("i", False, 0),
            # Custom attributes for unroll factors
            "channel_unroll": ("i", False, 1),
            "width_unroll": ("i", False, 1),
            # Custom attributes for input/output streams
            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node

        return helper.make_node(
            "Concat",
            inputs=node.input,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
            axis=self.get_nodeattr("axis"),
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        # create a standard add node to compute the result
        node = self.onnx_node
        node_add = helper.make_node(
            "Concat",
            inputs=node.input,
            outputs=node.output,
            axis=self.get_nodeattr("axis"),
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
        """ Get the internal cpp variables of the StreamingConcat node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __is_power_of_two(self, value) -> bool:
        """Check if a value is a power of two."""
        return value > 0 and float(np.log2(value)).is_integer()

    def __get_quantizer(self, input_quantA, input_quantB, output_quant) -> str:
        """ Returns the quantizer type for the Add operation. """

        if (
            self.__is_power_of_two(input_quantA.scale)
            and self.__is_power_of_two(input_quantB.scale)
            and self.__is_power_of_two(output_quant.scale)
        ):
            shift = -1 * (
                int(np.log2(input_quantA.scale))
                - int(np.log2(output_quant.scale))
            )
            return f"DequantQuantPo2<{shift}, {get_hls_quant_type(input_quantA)}, {get_hls_quant_type(output_quant)}>"
        else:
            raise ValueError(
                "Float quantization is currently not supported for StreamingConcat."
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
        input_shapeA = input_shapeA + [1] * (4 - len(input_shapeA))  # Ensure 4D shape.

        input_shapeB = model.get_tensor_shape(self.onnx_node.input[1])
        if input_shapeB is None:
            raise ValueError(f"Input {self.onnx_node.input[1]} has no shape info")
        input_shapeB = input_shapeB + [1] * (4 - len(input_shapeB))  # Ensure 4D shape.

        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Output {self.onnx_node.output[0]} has no shape info")
        output_shape = output_shape + [1] * (4 - len(output_shape))  # Ensure 4D shape.

        axis = self.get_nodeattr("axis")
        axis_strings = ["Channel", "Height", "Width"] 
        class_name = f"StreamingConcat{axis_strings[axis - 1]}"
        if axis == 1:
            template_args = [
                (f"{input_shapeA[2]}", "IN_HEIGHT"),
                (f"{input_shapeA[3]}", "IN_WIDTH"),
                (f"{input_shapeA[1]}", "IN_CH_A"),
                (f"{input_shapeB[1]}", "IN_CH_B"),
            ]
        elif axis == 2:
            template_args = [
                (f"{input_shapeA[2]}", "IN_HEIGHT_A"),
                (f"{input_shapeB[2]}", "IN_HEIGHT_B"),
                (f"{input_shapeA[3]}", "IN_WIDTH"),
                (f"{input_shapeA[1]}", "IN_CH"),
            ]
        elif axis == 3:
            template_args = [
                (f"{input_shapeA[2]}", "IN_HEIGHT"),
                (f"{input_shapeA[3]}", "IN_WIDTH_A"),
                (f"{input_shapeB[3]}", "IN_WIDTH_B"),
                (f"{input_shapeA[1]}", "IN_CH"),
            ]
        else:
            raise ValueError(f"Unsupported concat axis: {axis}")

        StreamingConcat = cpp_object(
            class_name,
            f"{self.onnx_node.name}",
            template_args=[
                (
                    f"{get_struct_type(input_quantA, self.get_nodeattr('in_word_array'))}",
                    f"TInputWord",
                ),
                (
                    f"{get_hls_quant_type(input_quantA)}",
                    f"TInput",
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
                    f"{self.__get_quantizer(input_quantA, input_quantB, output_quant)}",
                    f"Quantizer",
                ),
                template_args[0],
                template_args[1],
                template_args[2],
                template_args[3],
                (f"{self.get_nodeattr('width_unroll')}", "W_PAR"),
                (f"{self.get_nodeattr('channel_unroll')}", "CH_PAR"),
            ]
        )

        return StreamingConcat.generate_declaration()

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
            original_op_type="StreamingConcat",
            hls_tag=hls_tag,
            hls_object_name=self.onnx_node.name,
            hls_variable_declarations=self.__get_variable_declaration(model),
            hls_run_call=self.__get_run_call(hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )
        hls_tag += 1

        return [hls_kernel], [], tensors_fifo_metadata, hls_tag
    
    def get_dse_points(self, model: ModelWrapper) -> list["StreamingConcat.DSEPoint"]:
        """ Generate the DSE points for the StreamingConcat operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            list[StreamingConcat.DSEPoint]: List of DSE points.
        """

        def divisors(n: list[int], clip: int) -> list[int]:
            return [
                i
                for i in range(1, min(n) + 1)
                if (all(x % i == 0 for x in n) and i <= clip)
            ]

        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(
                f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model."
            )
        input_bits = input_quant.bitwidth

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model."
            )
        output_bits = output_quant.bitwidth

        input_shape0 = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape0 is None:
            raise ValueError(
                f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model."
            )
        input_shape0 = input_shape0 + [1] * (4 - len(input_shape0))  # Ensure 4D shape.

        input_shape1 = model.get_tensor_shape(self.onnx_node.input[1])
        if input_shape1 is None:
            raise ValueError(
                f"Tensor shape for input '{self.onnx_node.input[1]}' not found in model."
            )
        input_shape1 = input_shape1 + [1] * (4 - len(input_shape1))  # Ensure 4D shape.

        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(
                f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model."
            )
        output_shape = output_shape + [1] * (4 - len(output_shape))  # Ensure 4D shape.

        # As of now, kernel height and width are completely unrolled.
        DSE_points = []
        for channel_unroll in divisors(
            [input_shape0[1], input_shape1[1], output_shape[1]],
            min(input_shape0[1], input_shape1[1], output_shape[1]),
        ):
            for width_unroll in divisors(
                [input_shape0[3], input_shape1[3], output_shape[3]],
                min(input_shape0[3], input_shape1[3], output_shape[3]),
            ):
                # Check dimension of input streams
                if (input_bits * channel_unroll) > 4096:
                    continue
                # Check dimension of output streams
                if (output_bits * channel_unroll) > 4096:
                    continue

                DSE_points.append(
                    self.DSEPoint(
                        channel_unroll, width_unroll
                    )
                )

        return DSE_points

    def get_latency(self, model: ModelWrapper) -> int:
        """ Estimate the latency of the StreamingConcat operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated latency in clock cycles.
        """
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Output {self.onnx_node.output[0]} has no shape info")

        unroll_factor = self.get_nodeattr("channel_unroll") * self.get_nodeattr(
            "width_unroll"
        )
        return np.prod(output_shape) // unroll_factor

    def get_brams(self, model: ModelWrapper) -> int:
        """ Estimate the BRAM usage of the StreamingConcat operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated BRAM usage.
        """
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        """ Estimate the DSP usage of the StreamingConcat operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated DSP usage.
        """
        return 0

    def has_linebuffer(self) -> bool:
        """ Check if the StreamingConcat operation requires a line buffer.
        Returns:
            bool: True if Line Buffering is required, False otherwise.
        """
        return False

    def apply_point(self, model: ModelWrapper, point: "StreamingConcat.DSEPoint"):
        """ Set the parallelization attributes for the StreamingConcat operation.
        Args:
            point (StreamingConcat.DSEPoint): The DSE point containing the unrolling parameters.
        """
        self.set_nodeattr("channel_unroll", point.channel_unroll)
        self.set_nodeattr("width_unroll", point.width_unroll)

        self.set_nodeattr("in_stream_array", point.width_unroll)
        self.set_nodeattr("out_stream_array", point.width_unroll)
        self.set_nodeattr("in_word_array", point.channel_unroll)
        self.set_nodeattr("out_word_array", point.channel_unroll)
