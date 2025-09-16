from onnx import helper
from qonnx.custom_op.base import CustomOp
from qonnx.core.modelwrapper import ModelWrapper
from backend.core.tensor_quant import get_custom_tensor_datatype
from backend.core.tensor_fifo import TensorFifo
from backend.custom_op.hlskernel import HLSKernel
from backend.util.codegen_utils import (
    cpp_function,
    cpp_variable,
    cpp_object,
    get_struct_type,
    get_stream_type,
    get_hls_quant_type,
    get_cpp_quant_type
)
from backend.util.par_utils import get_par_attributes
import math

class NHWCToStream(CustomOp):
    """ Node producing a streaming tensor starting from an axi lite interface. """

    def get_nodeattr_types(self):
        return {
            "normalize": ("i", False, 0),  # 0: no normalization, 1: normalize the input tensor
            "axi_bitwidth": ("i", False, 128),  # Bitwidth of the AXI interface
            "pipeline_depth": ("i", False, 1),  # Depth of the pipeline
            "in_ch_par": ("i", False, 1),  # Input channel parallelization
            "out_ch_par": ("i", False, 1),  # Output channel parallelization
            "in_w_par": ("i", False, 1),  # Input width parallelization
            "out_w_par": ("i", False, 1),  # Output width parallelization
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
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        return int(math.floor(axi_bitwidth / output_quant.bitwidth))

    def __get_variable_cpp(self, model) -> str:
        """ Get the internal cpp variables of the NHWCToStream node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ''

    def __get_object_declaration(self, model: ModelWrapper) -> str:
        """ Generates the cpp NHWCToStream object. 
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: The NHWCToStream as cpp_object.
        """
        # The input has to be an AXI Lite interface, the bitwidth is defined by the board used.
        input_bitwidth = self.get_nodeattr("axi_bitwidth")

        # Input and output quantization are the same, since the NHWCToStream node
        # does not change the data type of the input tensor.
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        # Retrieve parallelization attributes.
        par_attribute = get_par_attributes(self.onnx_node)

        # Retrieve tensor shape.
        input_shape = model.get_tensor_shape(self.onnx_node.input[0])
        if input_shape is None:
            raise ValueError(f"Tensor shape for input '{self.onnx_node.input[0]}' not found in model.")
        input_shape = input_shape + [1] * (4 - len(input_shape))  # Pad to 4D if needed.

        NHWCToStream = cpp_object(
            "NHWCToStream",
            f"{self.onnx_node.name}",
            [
                (f"ap_axiu<{input_bitwidth}, 0, 0, 0>", "TInputStruct"),
                (f"ap_uint<{input_bitwidth}>", "TInput"),
                (
                    f"{get_struct_type(output_quant, par_attribute['out_ch_par'])}",
                    "TOutputStruct",
                ),
                (f"{get_hls_quant_type(output_quant)}", "TOutput"),
                (
                    f"DequantQuantEqual<{get_hls_quant_type(output_quant)}>",
                    "Quantizer",
                ),
                (self.__get_data_per_word(model), "DATA_PER_WORD"),
                (input_shape[2], "HEIGHT"),
                (input_shape[3], "WIDTH"),
                (input_shape[1], "CH"),
                (par_attribute["out_w_par"], "OUT_W_PAR"),
                (par_attribute["out_ch_par"], "OUT_CH_PAR"),
            ],
            [
                (f"{self.get_nodeattr('pipeline_depth')}", "pipeline_depth"),
            ]
        )

        return NHWCToStream.generate_declaration()

    def __get_run_call(self) -> str:
        """ Generates the C++ code necessary to run the NHWCToStream node. """

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
            self.onnx_node.input[0],
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def __get_step_call(self) -> str:
        """ Generates the C++ code necessary to run the NHWCToStream node in step mode. """

        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="void",
            arguments=(
                (
                    f"output_data_stream",
                    f"hls::stream<TOutputStruct>",
                ),
            ),
        )

        return step.generate_call(
            [],
            self.onnx_node.input[0],
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def lower_to_hls(self, model: ModelWrapper):
        """
        Returns:
          nodes: List[onnx.NodeProto]
          value_infos: List[onnx.ValueInfoProto]
          fifo: Dict[str, TensorFifo]
        """

        par = get_par_attributes(self.onnx_node)
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])

        output_names = [
            f"{self.__get_stream_name(self.onnx_node.output[0])}_{i}_"
            for i in range(par["out_w_par"])
        ]

        tensors_fifo_metadata = {}
        for output in output_names:
            tensors_fifo_metadata[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(output_quant, par['out_ch_par'])}",
                n_array=par["out_w_par"],
            )

        hls_kernel = HLSKernel.make_node(
            inputs=[self.onnx_node.input[0]],
            outputs=output_names,
            name=f"{self.onnx_node.name}_hls",
            domain="backend.custom_op",
            original_op_type=self.onnx_node.op_type,
            hls_variable_declarations=self.__get_variable_cpp(model),
            hls_run_call=self.__get_run_call(),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )

        return [hls_kernel], [], tensors_fifo_metadata
