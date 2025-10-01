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

class StreamingLineBuffer(CustomOp):
    """ Node producing a streaming window. """

    def get_nodeattr_types(self):
        return {
            "in_ch_par": ("i", True, 1),
            "out_ch_par": ("i", True, 1),
            "in_w_par": ("i", True, 1),
            "out_w_par": ("i", True, 1),
            "kernel_shape": ("ints", True, [1, 1]),
            "strides": ("ints", True, [1, 1]),
            "pads": ("ints", True, [0, 0, 0, 0]),
            "dilation": ("ints", True, [1, 1]),
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

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int):
        """
        Returns:
          nodes: List[onnx.NodeProto]
          initializers: List[onnx.TensorProto]
          fifo: Dict[str, TensorFifo]
        """

        hls_kernels = []
        fifos = {}
        par = get_par_attributes(self.onnx_node)
        output_quant = get_custom_tensor_datatype(model, self.onnx_node.output[0])
        if output_quant is None:
            raise ValueError(f"Tensor quantization for output '{self.onnx_node.output[0]}' not found in model.")

        # Retrieve tensor shape.
        output_shape = model.get_tensor_shape(self.onnx_node.output[0])
        if output_shape is None:
            raise ValueError(f"Tensor shape for output '{self.onnx_node.output[0]}' not found in model.")
        output_shape = output_shape + [1] * (4 - len(output_shape))  # Ensure 4D shape.

        FH = self.get_nodeattr("kernel_shape")[0]
        FW = self.get_nodeattr("kernel_shape")[1]
        PAD_T = self.get_nodeattr("pads")[0]
        PAD_L = self.get_nodeattr("pads")[1]
        STRIDE_H = self.get_nodeattr("strides")[0]
        STRIDE_W = self.get_nodeattr("strides")[1]
        FW_EXTENDED = FW + (par["in_w_par"] - 1) * STRIDE_W

        # Create output fifo streams from the pixelWindowSelector and the
        # Pad if needed.
        output_name = self.__get_stream_name(self.onnx_node.output[0])
        if self.get_nodeattr("pads") != [0, 0, 0, 0]:
            for i in range(FH * FW_EXTENDED):
                fifos[f"{output_name}_{i}_"] = TensorFifo(
                    depth=0,
                    hls_type=f"{get_struct_type(output_quant, par['out_ch_par'])}",
                    n_array=FH * FW_EXTENDED,
                )
            output_name = f"{output_name}_prepad"

        for i in range(FH * FW_EXTENDED):
            fifos[f"{output_name}_{i}_"] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(output_quant, par['out_ch_par'])}",
                n_array=FH * FW_EXTENDED,
            )

        # Create the PixelWindowSelector internal streams.
        # The last W_PAR nodes does not streams out anything.
        for i in range(FH * FW_EXTENDED - par["in_w_par"]):
            fifos[f"{self.onnx_node.name}_buffer_stream_{i}_"] = TensorFifo(
                depth=0,
                hls_type=f"{get_struct_type(output_quant, par['in_ch_par'])}",
                n_array=FH * FW_EXTENDED - par["in_w_par"],
            )

        # Shift index pattern, based on which part of the tensor the pixel is
        # considering.
        shift_pattern = []
        for i_fh in range(FH):
            for i_fw in range(FW_EXTENDED):
                pixel_w = FW_EXTENDED - 1 - i_fw
                shift_pattern.append(
                    (pixel_w + PAD_L * (par["in_w_par"] - 1)) % par["in_w_par"]
                )

        for i_fh in range(FH):
            for i_fw in range(FW_EXTENDED):
                pixel_h = FH - 1 - i_fh
                pixel_w = FW_EXTENDED - 1 - i_fw
                pixel_index = i_fh * FW_EXTENDED + i_fw
                function_args = set()

                # Get the section of tensor considered by this pixel.
                w_stream = shift_pattern[pixel_index]

                # Determine the input stream for this pixel.
                pixel_input_name = []
                if pixel_index < par["in_w_par"]:
                    # Directly from input stream.
                    pixel_input_name.append(
                        f"{self.__get_stream_name(self.onnx_node.input[0])}_{w_stream}_"
                    )
                else:
                    # From the internal buffer.
                    buffer_index = pixel_index - par["in_w_par"]
                    pixel_input_name.append(f"{self.onnx_node.name}_buffer_stream_{buffer_index}_")
                function_args.add((
                    "i_data",
                    "hls::stream<TWord>",
                ))

                # Determine the output stream for this pixel.
                pixel_output_name = [f"{output_name}_{(FH * FW_EXTENDED) - pixel_index - 1}_"]
                function_args.add((
                    "o_data",
                    "hls::stream<TWord>",
                ))

                # Determine the output shift stream for this pixel.
                if pixel_index < FH * FW_EXTENDED - par["in_w_par"]:
                    # Search the next pixel with the same w_stream.
                    next_pixel_index = None
                    for search_pixel_index in range(pixel_index + 1, FH * FW_EXTENDED):
                        if shift_pattern[search_pixel_index] == w_stream:
                            next_pixel_index = search_pixel_index
                            break
                    next_pixel_index = next_pixel_index - par["in_w_par"]
                    pixel_output_name.append(f"{self.onnx_node.name}_buffer_stream_{next_pixel_index}_")
                    function_args.add((
                        "o_shift_data",
                        "hls::stream<TWord>",
                    ))

                # Create run call
                run = cpp_function(
                    name=f"{self.onnx_node.name}_pixel_{pixel_index}.run",
                    return_type="void",
                    arguments=function_args,
                )
                run_call = run.generate_call([hls_tag], *pixel_input_name, *pixel_output_name)

                # Create step call
                step = cpp_function(
                    name=f"{self.onnx_node.name}_pixel_{pixel_index}.step",
                    return_type="ActorStatus",
                    arguments=function_args,
                )
                step_call = step.generate_call(
                    [], *pixel_input_name, *pixel_output_name
                )

                # Create the WindowSelector object.
                WindowSelector = cpp_object(
                    f"StreamingWindowSelector",
                    f"{self.onnx_node.name}_pixel_{pixel_index}",
                    template_args=[
                        (f"{get_struct_type(output_quant, par['in_ch_par'])}", "TWord"),
                        (output_shape[2], "IN_HEIGHT"),
                        (output_shape[3], "IN_WIDTH"),
                        (output_shape[1], "IN_CH"),
                        (self.get_nodeattr("kernel_shape")[0], "FH"),
                        (self.get_nodeattr("kernel_shape")[1], "FW"),
                        (self.get_nodeattr("strides")[0], "STRIDE_H"),
                        (self.get_nodeattr("strides")[1], "STRIDE_W"),
                        (self.get_nodeattr("dilation")[0], "DILATION_H"),
                        (self.get_nodeattr("dilation")[1], "DILATION_W"),
                        (self.get_nodeattr("pads")[0], "PAD_T"),
                        (self.get_nodeattr("pads")[1], "PAD_L"),
                        (self.get_nodeattr("pads")[2], "PAD_B"),
                        (self.get_nodeattr("pads")[3], "PAD_R"),
                        (pixel_h, "POS_H"),
                        (pixel_w, "POS_W"),
                        (par["in_w_par"], "W_PAR"),
                        (par["in_ch_par"], "CH_PAR"),
                    ],
                )

                # Create the HLS kernel for this pixel.
                hls_kernels.append(
                    HLSKernel.make_node(
                        inputs=pixel_input_name,
                        outputs=pixel_output_name,
                        name=f"{self.onnx_node.name}_pixel_{pixel_index}_hls",
                        domain="backend.custom_op",
                        original_op_type=self.onnx_node.op_type,
                        hls_object_name=f"{self.onnx_node.name}_pixel_{pixel_index}",
                        hls_tag=hls_tag,
                        hls_variable_declarations=self.__get_variable_declaration(
                            model
                        ),
                        hls_run_call=run_call,
                        hls_step_call=step_call,
                        hls_object_declaration=WindowSelector.generate_declaration(),
                    )
                )

                hls_tag += 1

        if self.get_nodeattr("pads") != [0, 0, 0, 0]:
            # Create the Pad kernel.
            input_name = f"{self.__get_stream_name(self.onnx_node.output[0])}_prepad"
            input_names = [
                f"{self.__get_stream_name(self.onnx_node.output[0])}_prepad_{i}_"
                for i in range(FH * FW_EXTENDED)
            ]
            output_name = f"{self.__get_stream_name(self.onnx_node.output[0])}"
            output_names = [
                f"{self.__get_stream_name(self.onnx_node.output[0])}_{i}_"
                for i in range(FH * FW_EXTENDED)
            ]
            function_args = (
                ("i_data", "hls::stream<TWord>"),
                ("o_data", "hls::stream<TWord>"),
            )

            run = cpp_function(
                name=f"{self.onnx_node.name}_pad.run",
                return_type="void",
                arguments=function_args,
            )
            run_call = run.generate_call([hls_tag], input_name, output_name)

            step = cpp_function(
                name=f"{self.onnx_node.name}_pad.step",
                return_type="ActorStatus",
                arguments=function_args,
            )
            step_call = step.generate_call(
                [],
                input_name,
                output_name
            )

            Pad = cpp_object(
                f"StreamingPad",
                f"{self.onnx_node.name}_pad",
                template_args=[
                    (f"{get_struct_type(output_quant, par['in_ch_par'])}", "TWord"),
                    (output_shape[2], "IN_HEIGHT"),
                    (output_shape[3], "IN_WIDTH"),
                    (output_shape[1], "IN_CH"),
                    (self.get_nodeattr("kernel_shape")[0], "FH"),
                    (self.get_nodeattr("kernel_shape")[1], "FW"),
                    (self.get_nodeattr("strides")[0], "STRIDE_H"),
                    (self.get_nodeattr("strides")[1], "STRIDE_W"),
                    (self.get_nodeattr("dilation")[0], "DILATION_H"),
                    (self.get_nodeattr("dilation")[1], "DILATION_W"),
                    (self.get_nodeattr("pads")[0], "PAD_T"),
                    (self.get_nodeattr("pads")[1], "PAD_L"),
                    (self.get_nodeattr("pads")[2], "PAD_B"),
                    (self.get_nodeattr("pads")[3], "PAD_R"),
                    (par["in_w_par"], "W_PAR"),
                    (par["in_ch_par"], "CH_PAR"),
                ])

            hls_kernels.append(
                HLSKernel.make_node(
                    inputs=input_names,
                    outputs=output_names,
                    name=f"{self.onnx_node.name}_pad_hls",
                    domain="backend.custom_op",
                    original_op_type=self.onnx_node.op_type,
                    hls_object_name=f"{self.onnx_node.name}_pad",
                    hls_tag=hls_tag,
                    hls_variable_declarations=self.__get_variable_declaration(model),
                    hls_run_call=run_call,
                    hls_step_call=step_call,
                    hls_object_declaration=Pad.generate_declaration()
                )
            )
            hls_tag += 1

        return hls_kernels, [], fifos, hls_tag
