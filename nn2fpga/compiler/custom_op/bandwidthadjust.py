from onnx import helper
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.core.tensor_quant import require_tensor_quant
from nn2fpga.compiler.core.tensor_layout import require_tensor_layout
from nn2fpga.compiler.core.tensor_fifo import TensorFifo
from nn2fpga.compiler.custom_op.hlskernel import HLSKernel
from nn2fpga.compiler.custom_op.op_base import NN2FPGAOp
from nn2fpga.compiler.utils.codegen_utils import (
    cpp_function,
    cpp_object,
    get_struct_type,
    get_hls_quant_type,
)

class BandwidthAdjust(NN2FPGAOp):
    """ Node adjusting a streaming tensor to match the bandwidth requirements."""

    def get_nodeattr_types(self):
        return {
            # Custom attributes for unroll factors
            "in_dim2_unroll": ("i", False, 1),
            "in_dim1_unroll": ("i", False, 1),
            "out_dim2_unroll": ("i", False, 1),
            "out_dim1_unroll": ("i", False, 1),
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

    def __get_variable_declaration(self, model) -> str:
        """ Get the internal cpp variables of the ProduceStream node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __get_object_declaration(self, model, name) -> str:
        input_quant = require_tensor_quant(model, self.onnx_node.input[0])
        output_quant = require_tensor_quant(model, self.onnx_node.output[0])
        input_layout = require_tensor_layout(model, self.onnx_node.input[0])
        input_shape = self.require_input_shape(model, 0)
        input_shape_permuted = [input_shape[i] for i in input_layout.perm]

        # Create the BandwidthAdjust object.
        BandwidthAdjust = cpp_object(
            name,
            f"{self.onnx_node.name}",
            template_args=[
                (f"{get_struct_type(input_quant, self.get_nodeattr('in_word_array'))}", "TInputWord"),
                (f"{get_hls_quant_type(input_quant)}", "TInput"),
                (f"{get_struct_type(output_quant, self.get_nodeattr('out_word_array'))}", "TOutputWord"),
                (f"{get_hls_quant_type(output_quant)}", "TOutput"),
                (
                    f"DequantQuantEqual<{get_hls_quant_type(output_quant)}>",
                    "Quantizer",
                ),
                (input_shape_permuted[1], "IN_DIM0"),
                (input_shape_permuted[2], "IN_DIM1"),
                (input_shape_permuted[3], "IN_DIM2"),
                (self.get_nodeattr("in_dim1_unroll"), "IN_DIM1_UNROLL"),
                (self.get_nodeattr("out_dim1_unroll"), "OUT_DIM1_UNROLL"),
                (self.get_nodeattr("in_dim2_unroll"), "IN_DIM2_UNROLL"),
                (self.get_nodeattr("out_dim2_unroll"), "OUT_DIM2_UNROLL"),
            ],
        )
        return BandwidthAdjust.generate_declaration()

    def __get_run_call(self, hls_tag: int) -> str:
        """ Generates the C++ code necessary to run the BandwidthAdjust node. """

        # Generate the call to the BandwidthAdjust run method.
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
        """ Generates the C++ code necessary to run the BandwidthAdjust node in step mode. """

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

    def accepted_input_layout(self) -> tuple | None:
        """ BandwidthAdjust is layout-agnostic, so it accepts any input layout. """
        return None

    def produced_output_layout(self, input_layout: tuple | None) -> tuple | None:
        """ BandwidthAdjust is layout-agnostic, so it produces the same layout as input. """
        return input_layout

    def lower_to_hls(self, model: ModelWrapper, name: str, hls_tag: int) -> tuple[list, list, dict]:
        """
        Lowers the BandwidthAdjust node to an HLSKernel node.
        Args:
          model: ModelWrapper
          name: Name of the HLS kernel
          hls_tag: Current HLS tag
        Returns:
          nodes: List[onnx.NodeProto]
          initializers: List[onnx.TensorProto]
          fifo: Dict[str, TensorFifo]
        """

        output_quant = require_tensor_quant(model, self.onnx_node.output[0])

        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(self.get_nodeattr("in_stream_array"))
        ]

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
            original_op_type=self.onnx_node.op_type,
            hls_object_name=self.onnx_node.name,
            hls_tag=hls_tag,
            hls_variable_declarations=self.__get_variable_declaration(model),
            hls_run_call=self.__get_run_call(hls_tag=hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model, name),
        )
        hls_tag += 1

        return [hls_kernel], [], tensors_fifo_metadata, hls_tag

    def get_latency(self, model: ModelWrapper) -> int:
        """ Estimate the latency of the BandwidthAdjust operation.
        Args:
            model: ModelWrapper
        Returns:
            int: Estimated latency in clock cycles.
        """

        input_shape = self.require_input_shape(model, 0)

        unroll_factor = np.prod(
            [
                min(self.get_nodeattr("in_dim2_unroll"), self.get_nodeattr("out_dim2_unroll")),
                min(self.get_nodeattr("in_dim1_unroll"), self.get_nodeattr("out_dim1_unroll")),
            ]
        )
        latency = np.prod(input_shape) // unroll_factor
        return latency

    def get_brams(self, model: ModelWrapper) -> int:
        """ Estimate the BRAM usage of the BandwidthAdjust operation.
        Args:
            model: ModelWrapper
        Returns:
            int: Estimated BRAM usage in number of BRAMs.
        """
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        """ Estimate the DSP usage of the BandwidthAdjust operation.
        Args:
            model: ModelWrapper
        Returns:
            int: Estimated DSP usage in number of DSPs.
        """
        return 0

    def has_linebuffer(self) -> bool:
        """ Check if the BandwidthAdjust operation requires a line buffer.
        Args:
            model: ModelWrapper
        Returns:
            bool: True if a line buffer is required, False otherwise.
        """
        return False

class BandwidthAdjustIncreaseStreams(BandwidthAdjust):
    """ Node increasing the number of streams in a tensor to match the bandwidth requirements."""

    def verify_node(self):
        return super().verify_node()

    def lower_to_hls(self, model, hls_tag: int):
        return super().lower_to_hls(model, "BandwidthAdjustIncreaseStreams", hls_tag=hls_tag)

class BandwidthAdjustDecreaseStreams(BandwidthAdjust):
    """ Node decreasing the number of streams in a tensor to match the bandwidth requirements."""

    def verify_node(self):
        return super().verify_node()
    
    def lower_to_hls(self, model, hls_tag: int):
        return super().lower_to_hls(model, "BandwidthAdjustDecreaseStreams", hls_tag=hls_tag)

class BandwidthAdjustIncreaseWord(BandwidthAdjust):
    """ Node increasing the number of data in a word to match the bandwidth requirements."""

    def verify_node(self):
        return super().verify_node()

    def lower_to_hls(self, model, hls_tag: int):
        return super().lower_to_hls(model, "BandwidthAdjustIncreaseWord", hls_tag=hls_tag)

class BandwidthAdjustDecreaseWord(BandwidthAdjust):
    """ Node decreasing the number of data in a word to match the bandwidth requirements."""

    def verify_node(self):
        return super().verify_node()

    def lower_to_hls(self, model, hls_tag: int):
        return super().lower_to_hls(model, "BandwidthAdjustDecreaseWord", hls_tag=hls_tag)
