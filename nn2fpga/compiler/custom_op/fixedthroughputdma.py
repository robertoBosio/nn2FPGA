from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
import numpy as np
from nn2fpga.compiler.core.tensor_type import require_tensor_type
from nn2fpga.compiler.core.tensor_layout import require_tensor_layout
from nn2fpga.compiler.core.tensor_fifo import TensorFifo
from nn2fpga.compiler.custom_op.hlskernel import HLSKernel
from nn2fpga.compiler.custom_op.op_base import NN2FPGAOp
from nn2fpga.compiler.utils.codegen_utils import (
    cpp_function,
    cpp_object,
)
import logging
logger = logging.getLogger(__name__)

class FixedThroughputDMA(NN2FPGAOp):
    """ Node producing a streaming window. """

    def get_nodeattr_types(self):
        return {
            "words_per_tensor": ("i", False, 1),
            "model_II": ("i", False, 1),
            "axi_bitwidth": ("i", False, 128),
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

    def __get_object_declaration(self) -> str:
        """ Get the internal cpp object declarations for the FixedThroughputDMA node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal objects.
        """
        model_II = self.get_nodeattr("model_II")
        words_per_tensor = self.get_nodeattr("words_per_tensor")
        axi_bitwidth = self.get_nodeattr("axi_bitwidth")
        FixedThroughputDMAObject = cpp_object(
            class_name="FixedThroughputDMA",
            obj_name=self.onnx_node.name,
            template_args=[f"ap_axiu<{axi_bitwidth}, 0, 0, 0>"],
            constructor_args=[words_per_tensor, model_II],
        )
        return FixedThroughputDMAObject.generate_declaration()

    def __generate_step_call(self) -> str:
        """Generate the HLS step call for the FixedThroughputDMA node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the HLS step call for the node.
        """

        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=((f"output_data_stream", f"hls::stream<TOutputWord>"),),
        )
        return step.generate_call(
            [],
            self.onnx_node.output[0],
        )

    def __generate_run_call(self, hls_tag: int) -> str:
        """Generate the HLS run call for the FixedThroughputDMA node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the HLS run call for the node.
        """
        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(("output_data_stream", f"hls::stream<TOutputWord>"),),
        )
        return run.generate_call([hls_tag], self.onnx_node.output[0])

    def accepted_input_layout(self) -> tuple | None:
        """ FixedThroughputDMA supports any input layout."""
        return None

    def produced_output_layout(self, input_layout: tuple | None) -> tuple | None:
        """ FixedThroughputDMA produces the same layout as its input (transparent)."""
        return input_layout

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int):
        """
        Lower the FixedThroughputDMA node to HLS kernels.
        Args:
          model: ModelWrapper
          hls_tag: starting HLS tag integer
        Returns:
          nodes: List[onnx.NodeProto]
          initializers: List[onnx.TensorProto]
          fifo: Dict[str, TensorFifo]
        """

        fifos = {}
        fifos[self.onnx_node.output[0]] = TensorFifo(
            depth=0,
            hls_type=f"ap_axiu<{self.get_nodeattr('axi_bitwidth')}, 0, 0, 0>",
            n_array=1,
        )

        hls_kernel = HLSKernel.make_node(
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            name=f"{self.onnx_node.name}_{self.onnx_node.output[0]}_hls",
            domain="nn2fpga.compiler.custom_op",
            original_op_type="FixedThroughputDMA",
            hls_object_name=self.onnx_node.name,
            hls_tag=hls_tag,
            hls_variable_declarations="",
            hls_run_call=self.__generate_run_call(hls_tag),
            hls_step_call=self.__generate_step_call(),
            hls_object_declaration=self.__get_object_declaration(),
        )
        hls_tag += 1

        return [hls_kernel], [], fifos, hls_tag

    def get_latency(self, model: ModelWrapper) -> int:
        """ Estimate the latency of the FixedThroughputDMA.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            int: Estimated latency in clock cycles.
        """
        return self.get_nodeattr("model_II")

    def get_brams(self, model: ModelWrapper) -> int:
        """ Estimate the BRAM usage of the FixedThroughputDMA.

        Args:
            model (ModelWrapper): The model with quantization information.

        Returns:
            int: Estimated BRAM usage.
        """
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        """ Estimate the DSP usage of the FixedThroughputDMA.

        Args:
            model (ModelWrapper): The model with quantization information.

        Returns:
            int: Estimated DSP usage.
        """
        return 0

    def has_linebuffer(self) -> bool:
        """ Check if the FixedThroughputDMA operation requires a line buffer.
        Returns:
            bool: True if a line buffer is required, False otherwise.
        """
        return False
