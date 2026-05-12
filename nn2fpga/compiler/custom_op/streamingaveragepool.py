import numpy as np
from dataclasses import dataclass
from onnx import helper
from onnxscript.rewriter import pattern
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.core.tensor_type import QuantizedTensorType, require_tensor_type
from nn2fpga.compiler.core.tensor_layout import require_tensor_layout
from nn2fpga.compiler.core.tensor_fifo import TensorFifo
from nn2fpga.compiler.custom_op.hlskernel import HLSKernel
from nn2fpga.compiler.custom_op.op_base import NN2FPGAOp, DSECapable
from nn2fpga.compiler.custom_op.register_rewrite_rule import register_rules
from nn2fpga.compiler.utils.codegen_utils import (
    cpp_function,
    cpp_object,
    get_word_type,
)


class StreamingAveragePool(NN2FPGAOp, DSECapable):
    """Node implementing the output-stationary AveragePool operation."""

    @dataclass(frozen=True)
    class DSEPoint:
        dim2_unroll: int
        dim1_unroll: int
        filter_width_unroll: int
        filter_height_unroll: int

        def to_dict(self) -> dict:
            return {
                "dim2_unroll": self.dim2_unroll,
                "dim1_unroll": self.dim1_unroll,
                "filter_width_unroll": self.filter_width_unroll,
                "filter_height_unroll": self.filter_height_unroll,
            }

        @staticmethod
        def from_dict(d: dict) -> "StreamingAveragePool.DSEPoint":
            return StreamingAveragePool.DSEPoint(
                dim2_unroll=d["dim2_unroll"],
                dim1_unroll=d["dim1_unroll"],
                filter_width_unroll=d["filter_width_unroll"],
                filter_height_unroll=d["filter_height_unroll"],
            )

    @staticmethod
    def pattern(
        op,
        x,
        kernel_shape,
        pads,
        strides,
        ceil_mode,
        count_include_pad,
    ):
        y = op.AveragePool(
            x,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            _allow_other_attributes=True,
        )
        return y

    @staticmethod
    def rewrite(
        op,
        x,
        kernel_shape,
        pads,
        strides,
        ceil_mode,
        count_include_pad,
    ):
        return op.StreamingAveragePool(
            x,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            _domain="nn2fpga.compiler.custom_op",
        )

    @register_rules
    def _rewriter_rules():
        return [
            pattern.RewriteRule(
                StreamingAveragePool.pattern, StreamingAveragePool.rewrite
            )
        ]

    def get_nodeattr_types(self):
        return {
            # Standard ONNX attributes for Conv
            "kernel_shape": ("ints", True, [1, 1]),
            "pads": ("ints", True, [0, 0]),
            "strides": ("ints", True, [1, 1]),
            "ceil_mode": ("i", False, 0),
            "count_include_pad": ("i", False, 0),
            # Custom attributes for parallelization of StreamingAveragePool
            "dim2_unroll": ("i", False, 1),
            "dim1_unroll": ("i", False, 1),
            "filter_height_unroll": ("i", False, 1),
            "filter_width_unroll": ("i", False, 1),
            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node

        return helper.make_node(
            "AveragePool",
            inputs=node.input,
            outputs=node.output,
            name=f"{node.name}_shape_compatible",
            kernel_shape=self.get_nodeattr("kernel_shape"),
            pads=self.get_nodeattr("pads"),
            strides=self.get_nodeattr("strides"),
            ceil_mode=self.get_nodeattr("ceil_mode"),
            count_include_pad=self.get_nodeattr("count_include_pad"),
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
        """Returns the quantizer type for the StreamingConv operation."""

        if self.__is_power_of_two(input_quant.scale) and self.__is_power_of_two(
            output_quant.scale
        ):
            shift = -1 * (
                int(np.log2(input_quant.scale)) - int(np.log2(output_quant.scale))
            )
            if (
                shift == 0
                and input_quant.bitwidth == output_quant.bitwidth
                and input_quant.signed == output_quant.signed
            ):
                return f"DequantQuantEqual<{input_quant.get_hls_data_type()}>"
            return f"DequantQuantPo2<{shift}, {input_quant.get_hls_data_type()}, {output_quant.get_hls_data_type()}>"
        else:
            raise ValueError(
                "Float quantization is currently not supported for StreamingAveragePool.  "
            )

    def __get_accumulator(self, fh, fw, input_quant) -> str:
        """Returns the accumulator type for the StreamingAveragePool operation."""

        add_ops = fh * fw
        acc_bitwidth = input_quant.bitwidth + int(np.ceil(np.log2(add_ops)))
        acc_quant = QuantizedTensorType(
            bitwidth=acc_bitwidth,
            signed=input_quant.signed,
            scale=input_quant.scale,
            zeropt=input_quant.zeropt,
        )

        return f"{acc_quant.get_hls_data_type()}"

    def __get_divisor(self, fh, fw) -> str:
        """Returns the divisor type for the StreamingAveragePool operation."""

        divisor = fh * fw
        divisor_quant = QuantizedTensorType(
            bitwidth=int(np.ceil(np.log2(divisor + 1))),
            signed=False,
            scale=1.0,
            zeropt=0,
        )
        return f"{divisor_quant.get_hls_data_type()}"

    def __get_object_declaration(self, model) -> cpp_object:
        """Generate the cpp_object for the StreamingAveragePool operation."""

        input_type = require_tensor_type(model, self.onnx_node.input[0])
        output_type = require_tensor_type(model, self.onnx_node.output[0])

        # Retrieve parallelization attributes.
        point = self.__current_dse_point()

        # Retrieve tensor shape.
        output_shape = self.require_output_shape(model, 0)
        output_layout = require_tensor_layout(model, self.onnx_node.output[0])
        output_shape_permuted = [output_shape[dim] for dim in output_layout.perm]

        # Create the StreamingAveragePool object.
        StreamingAveragePool = cpp_object(
            "StreamingAveragePool",
            f"{self.onnx_node.name}",
            template_args=[
                (
                    f"{get_word_type(input_type, self.get_nodeattr('in_word_array'))}",
                    "TInputWord",
                ),
                (f"{input_type.get_hls_data_type()}", "TInput"),
                (
                    f"{get_word_type(output_type, self.get_nodeattr('out_word_array'))}",
                    "TOutputWord",
                ),
                (f"{output_type.get_hls_data_type()}", "TOutput"),
                (f"{self.__get_quantizer(input_type, output_type)}", "Quantizer"),
                (
                    f"{self.__get_accumulator(self.get_nodeattr('kernel_shape')[0], self.get_nodeattr('kernel_shape')[1], input_type)}",
                    "TAcc",
                ),
                (
                    f"{self.__get_divisor(self.get_nodeattr('kernel_shape')[0], self.get_nodeattr('kernel_shape')[1])}",
                    "TDiv",
                ),
                (output_shape_permuted[1], "OUT_DIM0"),
                (output_shape_permuted[2], "OUT_DIM1"),
                (output_shape_permuted[3], "OUT_DIM2"),
                (self.get_nodeattr("kernel_shape")[0], "FH"),
                (self.get_nodeattr("kernel_shape")[1], "FW"),
                (self.get_nodeattr("strides")[0], "STRIDE_DIM0"),
                (self.get_nodeattr("strides")[1], "STRIDE_DIM1"),
                (point.dim1_unroll, "DIM1_UNROLL"),
                (point.dim2_unroll, "DIM2_UNROLL"),
            ],
        )

        return StreamingAveragePool.generate_declaration()

    def __get_variable_declaration(self, model) -> str:
        """Get the internal cpp variables of the StreamingAveragePool node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __get_run_call(self, hls_tag: int) -> str:
        """Generates the C++ code necessary to run the StreamingAveragePool node."""

        # Generate the call to the StreamingAveragePool run method.
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
        """Generates the C++ code necessary to step the StreamingAveragePool node."""

        # Generate the call to the StreamingAveragePool step method.
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

    def __current_dse_point(self) -> "StreamingAveragePool.DSEPoint":
        """Retrieve the current DSE point from the ONNX attributes."""
        return StreamingAveragePool.DSEPoint(
            dim2_unroll=self.get_nodeattr("dim2_unroll"),
            dim1_unroll=self.get_nodeattr("dim1_unroll"),
            filter_height_unroll=self.get_nodeattr("filter_height_unroll"),
            filter_width_unroll=self.get_nodeattr("filter_width_unroll"),
        )

    def accepted_input_layout(self) -> tuple | None:
        """ StreamingAveragePool accepts only NHWC layout. """
        return (0, 2, 3, 1)

    def produced_output_layout(self, input_layout: tuple | None) -> tuple | None:
        """ StreamingAveragePool produces only NHWC layout. """
        return (0, 2, 3, 1)

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int):
        """
        Returns:
          nodes: List[onnx.NodeProto]
          initializers: List[onnx.TensorProto]
          fifo: Dict[str, TensorFifo]
        """

        point = self.__current_dse_point()
        FH = self.get_nodeattr("kernel_shape")[0]
        FW = self.get_nodeattr("kernel_shape")[1]
        STRIDE_W = self.get_nodeattr("strides")[1]
        FW_EXTENDED = FW + (point.dim1_unroll - 1) * STRIDE_W
        output_type = require_tensor_type(model, self.onnx_node.output[0])

        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(FH * FW_EXTENDED)
        ]

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
            original_op_type="StreamingAveragePool",
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
        """Estimate the latency of the StreamingAveragePool operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated latency in clock cycles.
        """
        output_shape = self.require_output_shape(model, 0)

        # Retrieve current parallelization attributes if not provided.
        point = self.__current_dse_point()

        return np.prod(output_shape) // (point.dim2_unroll * point.dim1_unroll)

    def get_brams(self, model: ModelWrapper) -> int:
        """Estimate the BRAM usage of the StreamingAveragePool operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated BRAM usage.
        """
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        """Estimate the DSP usage of the StreamingAveragePool operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated DSP usage.
        """

        # Division and modulo operations both have a cost of 1 DSP.
        point = self.__current_dse_point()
        return (point.dim2_unroll * point.dim1_unroll) * 2

    def get_dse_points(
        self, model: ModelWrapper
    ) -> list["StreamingAveragePool.DSEPoint"]:
        """Generate the DSE points for the StreamingAveragePool operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            list[StreamingAveragePool.DSEPoint]: List of DSE points.
        """

        kernel_height, kernel_width = self.get_nodeattr("kernel_shape")

        input_type = require_tensor_type(model, self.onnx_node.input[0])
        input_bits = input_type.bitwidth

        output_type = require_tensor_type(model, self.onnx_node.output[0])
        output_bits = output_type.bitwidth

        output_layout = require_tensor_layout(model, self.onnx_node.output[0])
        output_shape = self.require_4d_output_shape(model, 0, output_layout)

        # As of now, kernel height and width are completely unrolled.
        DSE_points = []
        for dim2_unroll in self.divisors([output_shape[-1]], output_shape[-1]):
            for dim1_unroll in self.divisors([output_shape[-2]], output_shape[-2]):
                # Check dimension of input streams
                if (input_bits * dim2_unroll) > 4096:
                    continue
                # Check dimension of output streams
                if (output_bits * dim2_unroll) > 4096:
                    continue

                DSE_points.append(
                    self.DSEPoint(dim2_unroll, dim1_unroll, kernel_height, kernel_width)
                )

        return DSE_points

    def has_linebuffer(self) -> bool:
        """Check if the StreamingAveragePool operation requires a line buffer.
        Returns:
            bool: True if Line Buffering is required, False otherwise.
        """

        kernel_shape = self.get_nodeattr("kernel_shape")
        stride = self.get_nodeattr("strides")
        pads = self.get_nodeattr("pads")
        point = self.__current_dse_point()

        # The only case in which a StreamingAveragePool does not need a line buffer is when the kernel is 1x1,
        # there is no stride, and the output width parallelization is 1 with no padding.
        if (
            all(k == 1 for k in kernel_shape)
            and all(s == 1 for s in stride)
            and point.dim1_unroll == 1
            and all(p == 0 for p in pads)
        ):
            return False
        return True

    def apply_point(self, model: ModelWrapper, point: "StreamingAveragePool.DSEPoint"):
        """Set the parallelization attributes for the StreamingAveragePool operation.
        Args:
            point (StreamingAveragePool.DSEPoint): The DSE point containing the unrolling parameters.
        """
        self.set_nodeattr("dim2_unroll", point.dim2_unroll)
        self.set_nodeattr("dim1_unroll", point.dim1_unroll)
        self.set_nodeattr("filter_width_unroll", point.filter_width_unroll)
        self.set_nodeattr("filter_height_unroll", point.filter_height_unroll)

        self.set_nodeattr("in_stream_array", point.dim1_unroll)
        self.set_nodeattr("out_stream_array", point.dim1_unroll)
        self.set_nodeattr("in_word_array", point.dim2_unroll)
        self.set_nodeattr("out_word_array", point.dim2_unroll)
