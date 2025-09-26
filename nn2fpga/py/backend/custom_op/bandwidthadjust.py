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
)
from backend.util.par_utils import get_par_attributes

class BandwidthAdjust(CustomOp):
    """ Node adjusting a streaming tensor to match the bandwidth requirements."""

    def get_nodeattr_types(self):
        return {
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

    def __get_variable_declaration(self, model) -> str:
        """ Get the internal cpp variables of the ProduceStream node.
        Args:
            model (ModelWrapper): The model with quantization information.
        Returns:
            str: A string representing the declaration of internal variables.
        """
        return ""

    def __get_object_declaration(self, model, name) -> str:
        input_quant = get_custom_tensor_datatype(model, self.onnx_node.input[0])
        if input_quant is None:
            raise ValueError(f"Tensor quantization for input '{self.onnx_node.input[0]}' not found in model.")

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
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model.")

        # Create the BandwidthAdjust object.
        BandwidthAdjust = cpp_object(
            name,
            f"{self.onnx_node.name}",
            template_args=[
                (f"{get_struct_type(input_quant, par_attribute['in_ch_par'])}", "TInputStruct"),
                (f"{get_hls_quant_type(input_quant)}", "TInput"),
                (f"{get_struct_type(output_quant, par_attribute['out_ch_par'])}", "TOutputStruct"),
                (f"{get_hls_quant_type(output_quant)}", "TOutput"),
                (
                    f"DequantQuantEqual<{get_hls_quant_type(output_quant)}>",
                    "Quantizer",
                ),
                (input_shape[2], "IN_HEIGHT"),
                (input_shape[3], "IN_WIDTH"),
                (input_shape[1], "IN_CH"),
                (par_attribute["in_w_par"], "IN_W_PAR"),
                (par_attribute["out_w_par"], "OUT_W_PAR"),
                (par_attribute["in_ch_par"], "IN_CH_PAR"),
                (par_attribute["out_ch_par"], "OUT_CH_PAR"),
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
            self.__get_stream_name(self.onnx_node.output[0]),
        )

    def lower_to_hls(self, model: ModelWrapper, name: str, hls_tag: int) -> tuple[list, list, dict]:
        """
        Returns:
          nodes: List[onnx.NodeProto]
          initializers: List[onnx.TensorProto]
          fifo: Dict[str, TensorFifo]
        """

        par = get_par_attributes(self.onnx_node)
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(
                f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model."
            )

        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(par["in_w_par"])
        ]

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
            inputs=input_names,
            outputs=output_names,
            name=f"{self.onnx_node.name}_hls",
            domain="backend.custom_op",
            original_op_type=self.onnx_node.op_type,
            hls_tag=hls_tag,
            hls_variable_declarations=self.__get_variable_declaration(model),
            hls_run_call=self.__get_run_call(hls_tag=hls_tag),
            hls_step_call=self.__get_step_call(),
            hls_object_declaration=self.__get_object_declaration(model, name),
        )
        hls_tag += 1

        return [hls_kernel], [], tensors_fifo_metadata, hls_tag


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

class BandwidthAdjustIncreaseChannels(BandwidthAdjust):
    """ Node increasing the number of channels in a tensor to match the bandwidth requirements."""

    def verify_node(self):
        return super().verify_node()

    def lower_to_hls(self, model, hls_tag: int):
        return super().lower_to_hls(model, "BandwidthAdjustIncreaseChannels", hls_tag=hls_tag)

class BandwidthAdjustDecreaseChannels(BandwidthAdjust):
    """ Node decreasing the number of channels in a tensor to match the bandwidth requirements."""

    def verify_node(self):
        return super().verify_node()

    def lower_to_hls(self, model, hls_tag: int):
        return super().lower_to_hls(model, "BandwidthAdjustDecreaseChannels", hls_tag=hls_tag)
