import numpy as np
from dataclasses import dataclass
from onnx import helper
from onnxscript.rewriter import pattern
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.core.tensor_quant import get_custom_tensor_datatype
from nn2fpga.compiler.core.tensor_fifo import TensorFifo
from nn2fpga.compiler.custom_op.hlskernel import HLSKernel
from nn2fpga.compiler.custom_op.op_base import NN2FPGAOp, DSECapable
from nn2fpga.compiler.custom_op.register_rewrite_rule import register_rules
from nn2fpga.compiler.utils.codegen_utils import (
    cpp_function,
    cpp_object,
    get_struct_type,
    get_hls_quant_type,
)
from onnxscript import ir
from onnx_ir import convenience as ir_convenience

def _get_const_tensor(split: ir.Value) -> list[int]:
    t = ir_convenience.get_const_tensor(split)  # handles initializer OR Constant node
    if t is None:
        raise ValueError("split is not a compile-time constant")
    arr = t.numpy().reshape(-1).tolist()
    return arr

class StreamingSplit(NN2FPGAOp, DSECapable):
    """ Node implementing the output-stationary Split operation. """

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
        def from_dict(d: dict) -> "StreamingSplit.DSEPoint":
            return StreamingSplit.DSEPoint(
                channel_unroll=d["channel_unroll"],
                width_unroll=d["width_unroll"],
            )

    @staticmethod
    def pattern_v13(
        op,
        input,
        split,
        axis,
    ):
        return op.Split(
            input,
            split,
            axis=axis,
            _allow_other_attributes=True,
            _outputs=2,
        )

    @staticmethod
    def rewrite_v13(
        op,
        input,
        split,
        axis,
    ):
        split_attr = _get_const_tensor(split)
        return op.StreamingSplit(
            input,
            split=split_attr,
            axis=axis,
            _outputs=2,
            _domain="nn2fpga.compiler.custom_op",
        )

    @register_rules
    def _rewriter_rules():
        return [
            pattern.RewriteRule(
                StreamingSplit.pattern_v13,
                StreamingSplit.rewrite_v13,
            )
        ]

    def get_nodeattr_types(self):
        return {
            # Custom attributes for parallelization of StreamingSplit
            "channel_unroll": ("i", False, 1),
            "width_unroll": ("i", False, 1),

            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),

            # Standard ONNX Split attributes
            "split": ("ints", True, []),
            "axis": ("i", False, 0),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node

        return helper.make_node(
            "Split",
            inputs=node.input,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
            split=self.get_nodeattr("split"),
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

    def __is_power_of_two(self, value) -> bool:
        """Check if a value is a power of two."""
        return value > 0 and float(np.log2(value)).is_integer()

    def __get_quantizer(self, input_quant, output_quant) -> str:
        """ Returns the quantizer type for the StreamingConv operation. """

        if (
            self.__is_power_of_two(input_quant.scale)
            and self.__is_power_of_two(output_quant.scale)
        ):
            shift = -1 * (
                int(np.log2(input_quant.scale))
                - int(np.log2(output_quant.scale))
            )
            return f"DequantQuantPo2<{shift}, {get_hls_quant_type(input_quant)}, {get_hls_quant_type(output_quant)}>"
        else:
            raise ValueError(
                "Float quantization is currently not supported for StreamingConv.  "
            )

    def __get_object_declaration(self, model) -> cpp_object:
        """ Generate the cpp_object for the StreamingSplit operation. """

        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model.")

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        # Retrieve parallelization attributes.
        point = self.__current_dse_point()

        # Retrieve tensor shape.
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model.")
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model.")
        output_shape = output_shape + [1] * (4 - len(output_shape))  # Ensure 4D shape.

        split_point = self.get_nodeattr("split")[0]
        axis = self.get_nodeattr("axis")

        if axis == 1:
            operator_name = "StreamingSplitChannels"
        elif axis == 3:
            operator_name = "StreamingSplitWidths"
        elif axis == 2:
            operator_name = "StreamingSplitHeights"
        else:
            raise ValueError(f"StreamingSplit does not support splitting along axis {axis}.")

        # Create the StreamingSplit object.
        StreamingSplit = cpp_object(
            operator_name,
            f"{self.onnx_node.name}",
            template_args=[
                (
                    f"{get_struct_type(input_quant, self.get_nodeattr('in_word_array'))}",
                    "TInputWord",
                ),
                (f"{get_hls_quant_type(input_quant)}", "TInput"),
                (
                    f"{get_struct_type(output_quant, self.get_nodeattr('out_word_array'))}",
                    "TOutputWord",
                ),
                (f"{get_hls_quant_type(output_quant)}", "TOutput"),
                (f"{self.__get_quantizer(input_quant, output_quant)}", "Quantizer"),
                (split_point, "SPLIT"),
                (input_shape[2], "IN_HEIGHT"),
                (input_shape[3], "IN_WIDTH"),
                (input_shape[1], "IN_CH"),
                (point.channel_unroll, "CH_PAR"),
                (point.width_unroll, "W_PAR"),
            ],
        )

        return StreamingSplit.generate_declaration()

    def __get_variable_declaration(self, model) -> str:
        """ Get the internal cpp variables of the StreamingSplit node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __get_run_call(self, hls_tag: int) -> str:
        """ Generates the C++ code necessary to run the StreamingSplit node. """

        # Generate the call to the StreamingSplit run method.
        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(
                (
                    f"i_data",
                    f"hls::stream<TInputStruct>",
                ),
                (
                    f"o_data",
                    f"hls::stream<TOutputStruct>",
                ),
            ),
        )

        return run.generate_call(
            [hls_tag],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.output[0]),
            self.__get_stream_name(self.onnx_node.output[1]),
        )

    def __get_step_call(self) -> str:
        """ Generates the C++ code necessary to step the StreamingSplit node. """

        # Generate the call to the StreamingSplit step method.
        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=(
                (
                    f"i_data",
                    f"hls::stream<TInputStruct>",
                ),
                (
                    f"o_data",
                    f"hls::stream<TOutputStruct>",
                ),
            ),
        )

        return step.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.output[0]),
            self.__get_stream_name(self.onnx_node.output[1]),
        )

    def __current_dse_point(self) -> "StreamingSplit.DSEPoint":
        """ Retrieve the current DSE point from the ONNX attributes. """
        return StreamingSplit.DSEPoint(
            channel_unroll=self.get_nodeattr("channel_unroll"),
            width_unroll=self.get_nodeattr("width_unroll"),
        )

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int):
        """
        Returns:
          nodes: List[onnx.NodeProto]
          initializers: List[onnx.TensorProto]
          fifo: Dict[str, TensorFifo]
        """

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(self.get_nodeattr('in_stream_array'))
        ]

        output_names = [
            f"{self.__get_stream_name(self.onnx_node.output[0])}_{i}_"
            for i in range(self.get_nodeattr('out_stream_array'))
        ]
        output_names.extend(
            [
                f"{self.__get_stream_name(self.onnx_node.output[1])}_{i}_"
                for i in range(self.get_nodeattr("out_stream_array"))
            ]
        )

        tensors_fifo_metadata = {}
        for output in output_names:
            tensors_fifo_metadata[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(output_quant, self.get_nodeattr('out_word_array'))}",
                n_array=self.get_nodeattr('out_stream_array'),
            )

        hls_kernel = HLSKernel.make_node(
            inputs=input_names,
            outputs=output_names,
            name=f"{self.onnx_node.name}_hls",
            domain="nn2fpga.compiler.custom_op",
            original_op_type="StreamingSplit",
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
        """Estimate the latency of the StreamingSplit operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated latency in clock cycles.
        """
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(
                f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model."
            )

        # Retrieve current parallelization attributes if not provided.
        point = self.__current_dse_point()

        return np.prod(input_shape) // (point.channel_unroll * point.width_unroll)

    def get_brams(self, model: ModelWrapper) -> int:
        """ Estimate the BRAM usage of the StreamingSplit operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated BRAM usage.
        """
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        """ Estimate the DSP usage of the StreamingSplit operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated DSP usage.
        """
        return 0

    def get_dse_points(self, model: ModelWrapper) -> list["StreamingSplit.DSEPoint"]:
        """ Generate the DSE points for the StreamingSplit operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            list[StreamingSplit.DSEPoint]: List of DSE points.
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

        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(
                f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model."
            )
        input_shape = input_shape + [1] * (4 - len(input_shape))  # Ensure 4D shape.
        output_shape0 = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape0 is None:
            raise ValueError(
                f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model."
            )
        output_shape0 = output_shape0 + [1] * (4 - len(output_shape0))  # Ensure 4D shape.
        output_shape1 = model.get_tensor_shape(self.onnx_node.output[1])
        if output_shape1 is None:
            raise ValueError(
                f"Tensor shape for output '{self.onnx_node.output[1]}' not found in model."
            )
        output_shape1 = output_shape1 + [1] * (4 - len(output_shape1))  # Ensure 4D shape.

        # As of now, kernel height and width are completely unrolled.
        DSE_points = []
        for channel_unroll in divisors(
            [output_shape0[1], output_shape1[1]],
            min(output_shape0[1], output_shape1[1]),
        ):
            for width_unroll in divisors(
                [output_shape0[3], output_shape1[3]],
                min(output_shape0[3], output_shape1[3]),
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

    def has_linebuffer(self) -> bool:
        """ Check if the StreamingSplit operation requires a line buffer.
        Returns:
            bool: True if Line Buffering is required, False otherwise.
        """
        return False

    def apply_point(self, model: ModelWrapper, point: "StreamingSplit.DSEPoint"):
        """ Set the parallelization attributes for the StreamingSplit operation.
        Args:
            point (StreamingSplit.DSEPoint): The DSE point containing the unrolling parameters.
        """
        self.set_nodeattr("channel_unroll", point.channel_unroll)
        self.set_nodeattr("width_unroll", point.width_unroll)

        self.set_nodeattr("in_stream_array", point.width_unroll)
        self.set_nodeattr("out_stream_array", point.width_unroll)
        self.set_nodeattr("in_word_array", point.channel_unroll)
        self.set_nodeattr("out_word_array", point.channel_unroll)
