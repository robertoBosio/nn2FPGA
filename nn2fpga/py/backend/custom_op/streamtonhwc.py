from backend.custom_op.hlskernel import HLSKernel
from onnx import helper
from qonnx.custom_op.base import CustomOp
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import get_custom_tensor_datatype
from backend.core.tensor_fifo import TensorFifo
from backend.util.codegen_utils import (
    cpp_function,
    cpp_variable,
    cpp_object,
    NewCodeWriter,
    get_cpp_quant_type,
    get_struct_type,
    get_stream_type,
    get_hls_quant_type,
)
from backend.util.par_utils import get_par_attributes
import math

class StreamToNHWC(CustomOp):
    """ Node consuming a streaming tensor to an axi lite interface. """

    def get_nodeattr_types(self):
        return {
            "axi_bitwidth": ("i", False, 128),  # Bitwidth of the AXI interface
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
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        par_attribute = get_par_attributes(self.onnx_node)
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        fitting_data = int(math.floor(axi_bitwidth / (output_quant.bitwidth * par_attribute["out_ch_par"] * par_attribute["out_w_par"])))
        return fitting_data * par_attribute["out_ch_par"] * par_attribute["out_w_par"]

    def __get_variable_declaration(self, model) -> str:
        """ Get the internal cpp variables of the ProduceStream node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __get_object_declaration(self, model) -> str:
        """ Generates the cpp StreamToNHWC object. 
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: The StreamToNHWC as cpp_object.
        """
        # The output has to be an AXI Lite interface, the bitwidth is defined by the board used.
        output_bitwidth = self.get_nodeattr("axi_bitwidth")

        # The output quant is the same as the input quant, since the StreamToNHWC node
        # does not change the data type of the input tensor.
        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.input[0]}' not found in model.")

        # Retrieve parallelization attributes.
        par_attribute = get_par_attributes(self.onnx_node)

        # Retrieve tensor shape.
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])

        # Create the StreamToNHWC object.
        StreamToNHWC = cpp_object(
            "StreamToNHWC",
            f"{self.onnx_node.name}",
            [
                (
                    f"{get_struct_type(input_quant, par_attribute['in_ch_par'])}",
                    "TInputStruct",
                ),
                (f"{get_hls_quant_type(input_quant)}", "TInput"),
                (f"ap_axiu<{output_bitwidth}, 0, 0, 0>", "TOutputStruct"),
                (f"ap_uint<{output_bitwidth}>", "TOutput"),
                (
                    f"DequantQuantEqual<{get_hls_quant_type(input_quant)}>",
                    "Quantizer",
                ),
                (self.__get_data_per_word(model), "DATA_PER_WORD"),
                (input_shape[2], "HEIGHT"),
                (input_shape[3], "WIDTH"),
                (input_shape[1], "CH"),
                (par_attribute["in_w_par"], "IN_W_PAR"),
                (par_attribute["in_ch_par"], "IN_CH_PAR"),
            ],
        )

        return StreamToNHWC.generate_declaration()

    def __get_run_call(self) -> str:
        """ Generates the C++ code necessary to run the StreamToNHWC node. """

        # Generate the call to the StreamToNHWC run method.
        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=(
                (
                    f"input_data_stream",
                    f"hls::stream<TInputStruct>",
                ),
                (
                    f"output_data_stream",
                    f"hls::stream<TOutputStruct>",
                ),
            ),
        )

        return run.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.onnx_node.output[0],
        )

    def __get_step_call(self) -> str:
        """ Generates the C++ code necessary to step the StreamToNHWC node. """

        # Generate the call to the StreamToNHWC step method.
        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=(
                (
                    f"input_data_stream",
                    f"hls::stream<TInputStruct>",
                ),
                (
                    f"output_data_stream",
                    f"hls::stream<TOutputStruct>",
                ),
            ),
        )

        return step.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            self.onnx_node.output[0],
        )

    def lower_to_hls(self, model: ModelWrapper):
        """
        Returns:
          nodes: List[onnx.NodeProto]
          initializers: List[onnx.TensorProto]
          fifo: Dict[str, TensorFifo]
        """

        par = get_par_attributes(self.onnx_node)

        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(par["in_w_par"])
        ]

        hls_kernel = HLSKernel.make_node(
            inputs=input_names,
            outputs=[self.onnx_node.output[0]],
            name=f"{self.onnx_node.name}_hls",
            domain="backend.custom_op",
            hls_variable_declarations=self.__get_variable_declaration(model),
            hls_run_call=self.__get_run_call(),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )

        return [hls_kernel], [], {}