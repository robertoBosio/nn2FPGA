from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp
from qonnx.core.modelwrapper import ModelWrapper
import numpy as np
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
)
from backend.core.tensor_quant import TensorQuant
from backend.util.par_utils import get_par_attributes

class StreamingMemory(CustomOp):
    """ Node storing the parameters of a node."""

    def get_nodeattr_types(self):
        return {
            "in_ch_par": ("i", True, 1),
            "out_ch_par": ("i", True, 1),
            "in_w_par": ("i", True, 1),
            "out_w_par": ("i", True, 1),
            "times": ("i", True, 1),
            "data_per_word": ("i", True, 1),
            "mem_shape": ("ints", True, [1, 1, 1, 1]),  # N, C, H, W
            "data_to_shift": ("i", False, 0),  # Number of bits to shift data left
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
        """ Get the internal cpp variables of the StreamingMemory node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __get_run_call(self, hls_tag: int) -> str:
        """ Generates the C++ code necessary to run the StreamingMemory node. """

        if self.get_nodeattr("data_to_shift") > 0:
            run = cpp_function(
                name=f"{self.onnx_node.name}.run",
                return_type="void",
                arguments=(
                    ("i_shift_data", f"hls::stream<TInput>"),
                    ("o_shift_data", f"hls::stream<TInput>"),
                    ("o_data", f"hls::stream<TOutputWord>"),
                ),
            )

            return run.generate_call(
                [hls_tag],
                self.__get_stream_name(self.onnx_node.input[0]),
                self.__get_stream_name(self.onnx_node.output[0]),
                self.__get_stream_name(self.onnx_node.output[1]),
            )
        else:
            # No shift needed, just pass through
            run = cpp_function(
                name=f"{self.onnx_node.name}.run",
                return_type="void",
                arguments=(
                    ("i_shift_data", f"hls::stream<TInput>"),
                    ("o_data", f"hls::stream<TOutputWord>"),
                ),
            )

            return run.generate_call(
                [],
                self.__get_stream_name(self.onnx_node.input[0]),
                self.__get_stream_name(self.onnx_node.output[0]),
            )
    
    def __get_step_call(self) -> str:
        """ Generates the C++ code necessary to run the StreamingMemory node. """

        if self.get_nodeattr("data_to_shift") > 0:
            step = cpp_function(
                name=f"{self.onnx_node.name}.step",
                return_type="ActorStatus",
                arguments=(
                    ("i_shift_data", f"hls::stream<TInput>"),
                    ("o_shift_data", f"hls::stream<TInput>"),
                    ("o_data", f"hls::stream<TOutputWord>"),
                ),
            )

            return step.generate_call(
                [],
                self.__get_stream_name(self.onnx_node.input[0]),
                self.__get_stream_name(self.onnx_node.output[0]),
                self.__get_stream_name(self.onnx_node.output[1]),
            )
        else:
            step = cpp_function(
                name=f"{self.onnx_node.name}.step",
                return_type="ActorStatus",
                arguments=(
                    ("i_shift_data", f"hls::stream<TInput>"),
                    ("o_data", f"hls::stream<TOutputWord>"),
                ),
            )

            return step.generate_call(
                [],
                self.__get_stream_name(self.onnx_node.input[0]),
                self.__get_stream_name(self.onnx_node.output[0]),
            )
    
    def __get_object_declaration(self, model) -> cpp_object:
        """ Generate the cpp_object for the StreamingMemory operation. """

        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model.")

        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        # Retrieve parallelization attributes.
        par_attribute = get_par_attributes(self.onnx_node)

        # Retrieve tensor shape.
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model.")
        output_shape = output_shape + [1] * (4 - len(output_shape))  # Ensure 4D shape.

        # Create the StreamingMemory object.
        StreamingMemory = cpp_object(
            "StreamingMemory",
            f"{self.onnx_node.name}",
            template_args=[
                (f"{get_struct_type(input_quant, 1)}", "TInput"),
                (f"{get_hls_quant_type(output_quant)}", "TOutput"),
                (f"{get_struct_type(output_quant, par_attribute['in_ch_par'] * par_attribute['out_ch_par'])}", "TOutputStruct"),
                (f"{self.get_nodeattr('data_per_word')}", "DATA_PER_WORD"),
                (f"{self.get_nodeattr('data_to_shift')}", "DATA_TO_SHIFT"),
                (f"{self.get_nodeattr('times')}", "TIMES"),
                (output_shape[0], "OUT_CH"),
                (output_shape[1], "IN_CH"),
                (output_shape[2], "FH"),
                (output_shape[3], "FW"),
                (par_attribute["out_ch_par"], "OUT_CH_PAR"),
                (par_attribute["in_ch_par"], "IN_CH_PAR"),
            ])

        return StreamingMemory.generate_declaration()

    def reshape_and_pack_init_to_int32words(self,
        arr: np.ndarray, data_bitwidth: int, word_bitwidth: int
    ) -> np.ndarray:
        """
        Packs values from the input array into 32-bit words, ensuring that each value
        fits entirely within a word (no value is split between two words). Uses padding
        as needed.

        Args:
            arr (np.ndarray): Input array of unsigned integers.
            data_bitwidth (int): Number of bits per value. Must be <= 32.

        Returns:
            np.ndarray: Packed 32-bit words as a 1D array of dtype=np.uint32.
        """
        data_bitwidth = int(data_bitwidth)
        if data_bitwidth > word_bitwidth or data_bitwidth <= 0:
            raise ValueError("data_bitwidth must be between 1 and 32")
        
        # Extend the array shape to 2D if it's 1D
        in_ch_par = self.get_nodeattr("in_ch_par")
        out_ch_par = self.get_nodeattr("out_ch_par")
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        
        OC, IC, *spatial = arr.shape
        S = int(np.prod(spatial)) if spatial else 1
        X = arr.reshape(OC // out_ch_par, out_ch_par, IC // in_ch_par, in_ch_par, S)
        X = X.transpose(2, 0, 1, 3, 4)
        B = X.reshape(-1, out_ch_par * in_ch_par, S)
        arr = B.reshape(B.shape[0], B.shape[1], *spatial) if spatial else B.reshape(-1, out_ch_par * in_ch_par)

        arr = arr.flatten()  # Ensure the input is a 1D array
        values_per_word = word_bitwidth // data_bitwidth  # Max number of values per word

        # Pad the array to make its length a multiple of values_per_word
        padded_len = int(
            ((len(arr) + values_per_word - 1) // values_per_word) * values_per_word
        )
        padded_arr = np.zeros(padded_len, dtype=np.uint32)
        padded_arr[: len(arr)] = arr

        packed = []
        for i in range(0, padded_len, values_per_word):
            word = 0
            for j in range(values_per_word):
                word |= (padded_arr[i + j] & ((1 << data_bitwidth) - 1)) << (data_bitwidth * j)
            packed.append(word)

        return np.array(packed, dtype=np.uint32)

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int):
        """
        Returns:
          nodes: List[onnx.NodeProto]
          initializers: List[onnx.TensorProto]
          fifo: Dict[str, TensorFifo]
        """

        par = get_par_attributes(self.onnx_node)
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        if model.get_initializer(self.onnx_node.input[0]) is None:
            input_names = [
                f"{self.__get_stream_name(self.onnx_node.input[0])}_0_"
            ]
        else:
            input_names = [
                self.onnx_node.input[0]
            ]

        output_names = [
            f"{self.__get_stream_name(self.onnx_node.output[0])}_{i}_"
            for i in range(par["out_w_par"])
        ]

        tensors_fifo_metadata = {}
        for output in output_names:
            tensors_fifo_metadata[output] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(output_quant, par['out_ch_par'] * par['in_ch_par'])}",
                n_array=par["out_w_par"],
            )

        if len(self.onnx_node.output) > 1:
            output_names.append(f"{self.__get_stream_name(self.onnx_node.output[1])}_0_")
            tensors_fifo_metadata[output_names[-1]] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(get_custom_tensor_datatype(model, self.onnx_node.output[1]), 1)}",
                n_array=1,
            )

        hls_kernel = HLSKernel.make_node(
            inputs=input_names,
            outputs=output_names,
            name=f"{self.onnx_node.name}_hls",
            domain="backend.custom_op",
            original_op_type=self.onnx_node.op_type,
            hls_tag=hls_tag,
            hls_variable_declarations=self.__get_variable_declaration(model),
            hls_run_call=self.__get_run_call(hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model),
        )
        hls_tag += 1

        return [hls_kernel], [], tensors_fifo_metadata, hls_tag
