from dataclasses import dataclass
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.core.tensor_quant import get_custom_tensor_datatype
from nn2fpga.compiler.core.tensor_fifo import TensorFifo
from nn2fpga.compiler.custom_op.hlskernel import HLSKernel
from nn2fpga.compiler.custom_op.op_base import NN2FPGAOp, NodeInterface
from nn2fpga.compiler.utils.codegen_utils import (
    cpp_function,
    cpp_object,
    get_struct_type,
)
import numpy as np


class TensorDuplicator(NN2FPGAOp):
    """Node duplicating a tensor to ensure that each consumer gets a separate copy."""

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
        }

    def infer_node_datatype(self, model: ModelWrapper):
        node = self.onnx_node
        in_dtype = model.get_tensor_datatype(node.input[0])
        for out in node.output:
            model.set_tensor_datatype(out, in_dtype)

    def make_shape_compatible_op(self, model: ModelWrapper):
        node = self.onnx_node
        shape_compatible_nodes = []
        for i in range(2):
            identity_node = helper.make_node(
                "Identity",
                inputs=[node.input[0]],
                outputs=[node.output[i]],
                name=f"{node.name}_shape_compatible_{i}",
            )
            shape_compatible_nodes.append(identity_node)
        return shape_compatible_nodes

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_name = node.input[0]
        input_val = context[input_name]
        num_copies = 2

        for i in range(num_copies):
            out_name = node.output[i]
            context[out_name] = input_val.copy()

    def verify_node(self):
        pass

    def __get_stream_name(self, name: str) -> str:
        """
        Returns the name of the stream for the tensor.
        """
        return f"{name}_stream"

    def __get_variable_declaration(self, model) -> str:
        """Get the internal cpp variables of the TensorDuplicator node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __get_object_declaration(self, model) -> str:
        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(
                f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model."
            )

        # Retrieve tensor shape.
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(
                f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model."
            )
        input_shape = input_shape + [1] * (4 - len(input_shape))  # Pad to 4D if needed.

        # Create the TensorDuplicator object.
        TensorDuplicator = cpp_object(
            "TensorDuplicator",
            f"{self.onnx_node.name}",
            template_args=[
                (
                    f"{get_struct_type(input_quant, self.get_nodeattr('in_word_array'))}",
                    "TWord",
                ),
                (input_shape[2], "IN_HEIGHT"),
                (input_shape[3], "IN_WIDTH"),
                (input_shape[1], "IN_CH"),
                (self.get_nodeattr("channel_unroll"), "CH_PAR"),
                (self.get_nodeattr("width_unroll"), "W_PAR"),
            ],
        )
        return TensorDuplicator.generate_declaration()

    def __get_run_call(self, hls_tag: int) -> str:
        """Generates the C++ code necessary to run the TensorDuplicator node."""

        # Generate the call to the TensorDuplicator run method.
        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(
                (
                    f"i_data",
                    f"hls::stream<TWord>",
                ),
                (
                    f"o_data0",
                    f"hls::stream<TWord>",
                ),
                (
                    f"o_data1",
                    f"hls::stream<TWord>",
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
        """Generates the C++ code necessary to run the TensorDuplicator node in step mode."""

        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=(
                (
                    f"i_data",
                    f"hls::stream<TWord>",
                ),
                (
                    f"o_data0",
                    f"hls::stream<TWord>",
                ),
                (
                    f"o_data1",
                    f"hls::stream<TWord>",
                ),
            ),
        )

        return step.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.__get_stream_name(self.onnx_node.output[0]),
            self.__get_stream_name(self.onnx_node.output[1]),
        )

    def lower_to_hls(
        self, model: ModelWrapper, hls_tag: int
    ) -> tuple[list, list, dict]:
        """
        Returns:
          nodes: List[onnx.NodeProto]
          initializers: List[onnx.TensorProto]
          fifo: Dict[str, TensorFifo]
        """

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model."
            )

        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(self.get_nodeattr("in_stream_array"))
        ]

        output_names = [
            f"{self.__get_stream_name(self.onnx_node.output[0])}_{i}_"
            for i in range(self.get_nodeattr("out_stream_array"))
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
        """Check if the TensorDuplicator operation requires a linebuffer.
        Returns:
            bool: True if a linebuffer is required, False otherwise.
        """
        return False

    def get_latency(self, model: ModelWrapper) -> int:
        """Estimate the latency of the TensorDuplicator operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: The estimated latency in cycles.
        """ 
        
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        input_shape = input_shape + [1] * (4 - len(input_shape))  # Pad to 4D if needed.
        latency = np.prod(input_shape) // (self.get_nodeattr("channel_unroll") * self.get_nodeattr("width_unroll"))
        return latency
    
    def get_brams(self, model: ModelWrapper) -> int:
        """Estimate the BRAM usage of the TensorDuplicator operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: The estimated BRAM usage.
        """ 
        return 0  # TensorDuplicator does not use BRAMs.
    
    def get_dsps(self, model: ModelWrapper) -> int:
        """Estimate the DSP usage of the TensorDuplicator operation.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: The estimated DSP usage.
        """ 
        return 0  # TensorDuplicator does not use DSPs.
    
    def can_inherit_interface(self):
        return True
    
    def inherit_interface(self, model: ModelWrapper, upstream: NodeInterface) -> None:
        """ Inherit the interface from the upstream node."""
        self.set_nodeattr("in_stream_array", upstream.out_stream_array)
        self.set_nodeattr("out_stream_array", upstream.out_stream_array)
        self.set_nodeattr("in_word_array", upstream.out_word_array)
        self.set_nodeattr("out_word_array", upstream.out_word_array)

        self.set_nodeattr("channel_unroll", upstream.out_word_array)
        self.set_nodeattr("width_unroll", upstream.out_stream_array)