import math
import numpy as np
from dataclasses import dataclass
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.core.tensor_quant import require_tensor_quant
from nn2fpga.compiler.core.tensor_layout import require_tensor_layout
from nn2fpga.compiler.core.tensor_fifo import TensorFifo
from nn2fpga.compiler.custom_op.hlskernel import HLSKernel
from nn2fpga.compiler.custom_op.op_base import DSECapable, NN2FPGAOp
from nn2fpga.compiler.utils.codegen_utils import (
    cpp_function,
    cpp_object,
    get_struct_type,
    get_hls_quant_type,
)

class AXIToStream(DSECapable, NN2FPGAOp):
    """ Node producing a streaming tensor starting from an axi lite interface. """

    @dataclass(frozen=True)
    class DSEPoint:
        """DSE point for AXIToStream operator."""
        dim2_unroll: int
        dim1_unroll: int

        @staticmethod
        def from_dict(d: dict) -> "AXIToStream.DSEPoint":
            return AXIToStream.DSEPoint(
                dim2_unroll=d["dim2_unroll"],
                dim1_unroll=d["dim1_unroll"],
            )

        def to_dict(self) -> dict:
            return {
                "dim2_unroll": self.dim2_unroll,
                "dim1_unroll": self.dim1_unroll,
            }

    def get_nodeattr_types(self):
        return {
            "axi_bitwidth": ("i", False, 128),  # Bitwidth of the AXI interface
            # Custom attributes for unroll factors
            "dim2_unroll": ("i", False, 1),
            "dim1_unroll": ("i", False, 1),
            # Custom attributes for input/output streams
            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        return helper.make_node(
            "Identity",
            [node.input[0]],
            [node.output[0]],
            name=f"{node.name}_shape_compatible",
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_name = node.input[0]
        out_name = node.output[0]
        inp = context[inp_name]
        context[out_name] = inp

    def verify_node(self):
        pass

    def __get_stream_name(self, name: str) -> str:
        """
        Returns the name of the stream for the tensor.
        """
        return f"{name}_stream"

    def __get_data_per_word(self, model: ModelWrapper) -> int:
        """
        Returns the number of data elements that can be stored in a single word.
        This is calculated as the maximum number of pixels that can be stored in a single AXI word,
        as long as all the channels of it are fitting in the AXI word.
        """
        axi_bitwidth = self.get_nodeattr("axi_bitwidth")
        output_quant = require_tensor_quant(model, self.onnx_node.output[0])
        return int(math.floor(axi_bitwidth / output_quant.bitwidth))

    def __get_variable_cpp(self, model) -> str:
        """ Get the internal cpp variables of the AXIToStream node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ''

    def __get_object_declaration(self, model: ModelWrapper) -> str:
        """ Generates the cpp AXIToStream object. 
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: The AXIToStream as cpp_object.
        """
        # The input has to be an AXI Lite interface, the bitwidth is defined by the board used.
        input_bitwidth = self.get_nodeattr("axi_bitwidth")

        # Input and output quantization are the same, since the AXIToStream node
        # does not change the data type of the input tensor.
        output_quant = require_tensor_quant(model, self.onnx_node.output[0])

        # Retrieve parallelization attributes.
        point = self.__current_dse_point()

        # Retrieve tensor shape.
        input_shape = self.require_input_shape(model, 0)
        input_layout = require_tensor_layout(model, self.onnx_node.input[0])
        input_shape_permuted = [input_shape[i] for i in input_layout.perm]

        AXIToStream = cpp_object(
            "AXIToStream",
            f"{self.onnx_node.name}",
            [
                (f"ap_axiu<{input_bitwidth}, 0, 0, 0>", "TInputWord"),
                (f"ap_uint<{input_bitwidth}>", "TInput"),
                (
                    f"{get_struct_type(output_quant, self.get_nodeattr('out_word_array'))}",
                    "TOutputWord",
                ),
                (f"{get_hls_quant_type(output_quant)}", "TOutput"),
                (
                    f"DequantQuantEqual<{get_hls_quant_type(output_quant)}>",
                    "Quantizer",
                ),
                (self.__get_data_per_word(model), "DATA_PER_WORD"),
                (input_shape_permuted[1], "DIM0"),
                (input_shape_permuted[2], "DIM1"),
                (input_shape_permuted[3], "DIM2"),
                (point.dim1_unroll, "DIM1_UNROLL"),
                (point.dim2_unroll, "DIM2_UNROLL"),
            ]
        )

        return AXIToStream.generate_declaration()

    def __get_run_call(self, hls_tag: int) -> str:
        """ Generates the C++ code necessary to run the AXIToStream node. """

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
            self.onnx_node.input[0],
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def __get_step_call(self) -> str:
        """ Generates the C++ code necessary to run the AXIToStream node in step mode. """

        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=(
                (
                    f"o_data",
                    f"hls::stream<TOutputWord>",
                ),
            ),
        )

        return step.generate_call(
            [],
            self.onnx_node.input[0],
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int):
        """
        Returns:
          nodes: List[onnx.NodeProto]
          value_infos: List[onnx.ValueInfoProto]
          fifo: Dict[str, TensorFifo]
        """

        output_quant = require_tensor_quant(model, self.onnx_node.output[0])

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
            inputs=[self.onnx_node.input[0]],
            outputs=output_names,
            name=f"{self.onnx_node.name}_hls",
            domain="nn2fpga.compiler.custom_op",
            original_op_type=self.onnx_node.op_type,
            hls_object_name=self.onnx_node.name,
            hls_tag=hls_tag,
            hls_variable_declarations=self.__get_variable_cpp(model),
            hls_run_call=self.__get_run_call(hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )
        hls_tag += 1

        return [hls_kernel], [], tensors_fifo_metadata, hls_tag

    def __current_dse_point(self) -> "AXIToStream.DSEPoint":
        """ Retrieve the current DSE point from the node attributes.
        Returns:
            AXIToStream.DSEPoint: The current DSE point.
        """
        return AXIToStream.DSEPoint(
            dim2_unroll=self.get_nodeattr("dim2_unroll"),
            dim1_unroll=self.get_nodeattr("dim1_unroll"),
        )

    def get_latency(self, model: ModelWrapper) -> int:
        """ Estimate the latency of the NHWCtoStream operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated latency in clock cycles.
        """
        input_shape = self.require_input_shape(model, 0)

        # Retrieve current parallelization attributes if not provided.
        point = self.__current_dse_point()
        unroll_factor = point.dim2_unroll * point.dim1_unroll

        latency = np.prod(input_shape) // unroll_factor
        return latency

    def get_brams(self, model: ModelWrapper) -> int:
        """ Estimate the BRAM usage of the NHWCtoStream operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated BRAM usage.
        """
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        """ Estimate the DSP usage of the NHWCtoStream operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated DSP usage.
        """
        return 0

    def get_dse_points(self, model: ModelWrapper) -> list["AXIToStream.DSEPoint"]:
        """Generate all feasible DSE points for the NHWCtoStream operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            list[AXIToStream.DSEPoint]: A list of feasible DSE points.
        """

        def divisors(n, clip):
            return [i for i in range(1, n + 1) if (n % i == 0 and i <= clip)]

        axi_bitwidth = self.get_nodeattr("axi_bitwidth")
        output_quant = require_tensor_quant(model, self.onnx_node.output[0])
        input_shape = self.require_input_shape(model, 0)
        input_layout = require_tensor_layout(model, self.onnx_node.input[0])
        input_shape_permuted = [input_shape[i] for i in input_layout.perm]
        act_bits = output_quant.bitwidth

        DSE_points = []
        for dim2_unroll in divisors(input_shape_permuted[3], input_shape_permuted[3]):
            for dim1_unroll in divisors(
                input_shape_permuted[2], input_shape_permuted[2]
            ):

                # Check if the data fits in the AXI bitwidth.
                if (np.prod([dim2_unroll, dim1_unroll]) * act_bits) > axi_bitwidth:
                    continue

                # DIM1 parallelization can only be applied if the full DIM2 fits in the AXI word.
                if dim1_unroll > 1 and dim2_unroll != input_shape_permuted[3]:
                    continue

                DSE_points.append(
                    AXIToStream.DSEPoint(
                        dim2_unroll=dim2_unroll, dim1_unroll=dim1_unroll
                    )
                )

        return DSE_points

    def has_linebuffer(self, par: list = None) -> bool:
        """ Check if the NHWCtoStream operation requires Line Buffering.
        Returns:
            bool: True if a line buffer is required, False otherwise.
        """
        return False

    def apply_point(
        self, model: ModelWrapper, point: "AXIToStream.DSEPoint"
    ) -> None:
        """Set the unroll factors in the node attributes based on the given DSE point.
        Args:
            point (AXIToStream.DSEPoint): A DSE point containing the parallelization parameters.
        """
        self.set_nodeattr("dim2_unroll", point.dim2_unroll)
        self.set_nodeattr("dim1_unroll", point.dim1_unroll)

        self.set_nodeattr("in_stream_array", point.dim1_unroll)
        self.set_nodeattr("out_stream_array", point.dim1_unroll)
        self.set_nodeattr("in_word_array", point.dim2_unroll)
        self.set_nodeattr("out_word_array", point.dim2_unroll)
