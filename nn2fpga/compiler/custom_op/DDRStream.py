from dataclasses import dataclass
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.core.tensor_type import require_tensor_type
from nn2fpga.compiler.core.tensor_layout import require_tensor_layout
from nn2fpga.compiler.core.tensor_fifo import TensorFifo
from nn2fpga.compiler.custom_op.hlskernel import HLSKernel
from nn2fpga.compiler.custom_op.op_base import NN2FPGAOp, NodeInterface
from nn2fpga.compiler.utils.codegen_utils import (
    cpp_function,
    cpp_object,
    get_word_type,
)
import numpy as np


class DDRStream(NN2FPGAOp):
    """Node emulating a normal hls::stream, but implemented using DDR memory. """

    def get_nodeattr_types(self):
        return {
            # Custom attributes for unroll factors
            "dim2_unroll": ("i", False, 1),
            "axiword": ("i", False, 128), # Number of bits in the AXI word
            "burst_length": ("i", False, 16), # Number of AXI words in a burst
            "stream_depth": ("i", False, 1024), # Depth of the original FIFO
            "buffer_name": ("s", False, ""), # Name of the buffer in the HLS code
            # Custom attributes for input/output streams
            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),
        }

    def infer_node_datatype(self, model: ModelWrapper):
        node = self.onnx_node
        in_dtype = model.get_tensor_datatype(node.input[0])
        for out in node.output:
            model.set_tensor_datatype(out, in_dtype)

    def make_shape_compatible_op(self, model: ModelWrapper):
        node = self.onnx_node
        identity_node = helper.make_node(
            "Identity",
            inputs=[node.input[0]],
            outputs=[node.output[0]],
            name=f"{node.name}_shape_compatible_0",
        )
        return identity_node

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_name = node.input[0]
        input_val = context[input_name]
        context[node.output[0]] = input_val.copy()

    def verify_node(self):
        pass

    def __get_stream_name(self, name: str) -> str:
        """
        Returns the name of the stream for the tensor.
        """
        return f"{name}_stream"

    def __get_variable_declaration(self, model) -> str:
        """Get the internal cpp variables of the DDRStream node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __get_object_declaration(self, model) -> str:
        input_type = require_tensor_type(model, self.onnx_node.input[0])
        input_layout = require_tensor_layout(model, self.onnx_node.input[0])
        input_shape = self.require_4d_input_shape(model, 0, input_layout)
        axiword_bits = self.get_nodeattr("axiword")
        burst_length = self.get_nodeattr("burst_length")
        words_in_axiword = axiword_bits // input_type.bitwidth
        buffer_size = np.prod(input_shape) // (self.get_nodeattr("axiword") // input_type.bitwidth)

        # Create the DDRStream object.
        DDRStream = cpp_object(
            "DDRstream",
            f"{self.onnx_node.name}",
            template_args=[
                (
                    f"{get_word_type(input_type, self.get_nodeattr('in_word_array'))}",
                    "TWord",
                ),
                (
                    f"{input_type.get_hls_data_type()}",
                    "TData",
                ),
                (
                    f"ap_uint<{self.get_nodeattr('axiword')}>",
                    "TAxiWord",
                ),
                (buffer_size, "DIM"),
                (burst_length, "BURST_SIZE"),
                (words_in_axiword, "AXIWORD_PAR"),
                (self.get_nodeattr("dim2_unroll"), "DIM2_UNROLL"),
                (self.get_nodeattr("stream_depth"), "DEPTH"),
            ],
        )
        return DDRStream.generate_declaration()

    def __get_run_call(self, hls_tag: int) -> str:
        """Generates the C++ code necessary to run the DDRStream node."""

        # Generate the call to the DDRStream run method.
        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(
                (
                    f"input_stream",
                    f"hls::stream<TWord>",
                ),
                (
                    f"ddr_buffer_read",
                    f"TAXIWord*",
                ),
                (
                    f"ddr_buffer_write",
                    f"TAXIWord*",
                ),
                (
                    f"output_stream",
                    f"hls::stream<TWord>",
                ),
            ),
        )

        return run.generate_call(
            [hls_tag],
            self.__get_stream_name(self.onnx_node.input[0]),
            f"{self.get_nodeattr('buffer_name')}_read",
            f"{self.get_nodeattr('buffer_name')}_write",
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def __get_step_call(self) -> str:
        """Generates the C++ code necessary to run the DDRStream node in step mode."""

        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=(
                (
                    f"input_stream",
                    f"hls::stream<TWord>",
                ),
                (
                    f"ddr_buffer_read",
                    f"TAXIWord*",
                ),
                (
                    f"ddr_buffer_write",
                    f"TAXIWord*",
                ),
                (
                    f"output_stream",
                    f"hls::stream<TWord>",
                ),
            ),
        )

        return step.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            f"{self.get_nodeattr('buffer_name')}_read",
            f"{self.get_nodeattr('buffer_name')}_write",
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def accepted_input_layout(self) -> tuple | None:
        """ DDRStream is layout agnostic, since it just reads the input tensor as a stream of data. """
        return None

    def produced_output_layout(self, input_layout: tuple | None) -> tuple:
        """ The output layout of DDRStream is the same as the input layout. """
        return input_layout

    def lower_to_hls(
        self, model: ModelWrapper, hls_tag: int
    ) -> tuple[list, list, dict]:
        """
        Returns:
          nodes: List[onnx.NodeProto]
          initializers: List[onnx.TensorProto]
          fifo: Dict[str, TensorFifo]
        """

        output_quant = require_tensor_type(model, self.onnx_node.output[0])
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
                depth=2,
                hls_type=f"{get_word_type(output_quant, self.get_nodeattr('out_word_array'))}",
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
            hls_object_declaration=self.__get_object_declaration(model),
        )
        hls_tag += 1

        return [hls_kernel], [], tensors_fifo_metadata, hls_tag

    def has_linebuffer(self) -> bool:
        """Check if the DDRStream operation requires a linebuffer.
        Returns:
            bool: True if a linebuffer is required, False otherwise.
        """
        return False

    def get_latency(self, model: ModelWrapper) -> int:
        """Estimate the latency of the DDRStream operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: The estimated latency in cycles.
        """ 

        input_shape = self.require_4d_input_shape(model, 0)
        latency = np.prod(input_shape) // self.get_nodeattr("dim2_unroll")
        return latency

    def get_brams(self, model: ModelWrapper) -> int:
        """Estimate the BRAM usage of the DDRStream operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: The estimated BRAM usage.
        """ 
        return 0  # DDRStream does not use BRAMs.

    def get_dsps(self, model: ModelWrapper) -> int:
        """Estimate the DSP usage of the DDRStream operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: The estimated DSP usage.
        """ 
        return 0  # DDRStream does not use DSPs.

    def can_inherit_interface(self):
        return True

    def inherit_interface(self, model: ModelWrapper, upstream: NodeInterface) -> None:
        """ Inherit the interface from the upstream node."""
        self.set_nodeattr("in_stream_array", upstream.out_stream_array)
        self.set_nodeattr("out_stream_array", upstream.out_stream_array)
        self.set_nodeattr("in_word_array", upstream.out_word_array)
        self.set_nodeattr("out_word_array", upstream.out_word_array)

        self.set_nodeattr("dim2_unroll", upstream.out_word_array)
