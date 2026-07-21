import logging

from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper

from nn2fpga.compiler.core.tensor_fifo import TensorFifo
from nn2fpga.compiler.core.tensor_layout import require_tensor_layout
from nn2fpga.compiler.core.tensor_type import require_tensor_type
from nn2fpga.compiler.custom_op.hlskernel import HLSKernel
from nn2fpga.compiler.custom_op.op_base import NN2FPGAOp
from nn2fpga.compiler.utils.codegen_utils import cpp_function, cpp_object, get_word_type


logger = logging.getLogger(__name__)


class StreamingCircularLineBuffer(NN2FPGAOp):
    """Single-actor circular-buffer window generator."""

    def _ceil_div(self, num: int, den: int) -> int:
        return (num + den - 1) // den

    def _normalize_pads(self, pads):
        pads = list(pads)
        if len(pads) == 2:
            return [pads[0], pads[1], pads[0], pads[1]]
        return pads

    def _get_optional_nodeattr(self, attr_name: str, default=None):
        try:
            return self.get_nodeattr(attr_name)
        except Exception:
            return default

    def _get_output_fifo_depth(self, model: ModelWrapper) -> int:
        input_layout = require_tensor_layout(model, self.onnx_node.input[0])
        input_shape = self.require_4d_input_shape(model, 0, input_layout)

        pads = self._normalize_pads(self.get_nodeattr("pads"))
        fh, fw = self.get_nodeattr("kernel_shape")
        stride_h, stride_w = self.get_nodeattr("strides")
        width_unroll = self.get_nodeattr("width_unroll")
        channel_unroll = self.get_nodeattr("channel_unroll")

        is_pointwise = fh == 1 and fw == 1
        ch_groups = input_shape[-1] // channel_unroll
        input_words_per_row = input_shape[-2] // width_unroll
        padded_words_per_row = (input_shape[-2] + pads[1] + pads[3]) // width_unroll
        fw_extended = fw + (width_unroll - 1) * stride_w
        out_height = ((input_shape[-3] + pads[0] + pads[2] - (fh - 1) - 1) // stride_h) + 1
        out_width = ((input_shape[-2] + pads[1] + pads[3] - (fw - 1) - 1) // stride_w) + 1
        output_window_groups_per_row = out_width // width_unroll
        output_window_groups_total = out_height * output_window_groups_per_row * ch_groups
        input_words_total = input_shape[-3] * input_words_per_row * ch_groups
        model_ii = int(model.get_metadata_prop("model_II"))

        # Restart and stride gaps are modeled as no-output spans in linebuffer
        # firings over the padded tensor scan, then converted through the global
        # average input/output throughput ratio.

        # Width stride skips complete input word groups between valid emissions.
        width_gap_firings = max(0, self._ceil_div(stride_w, width_unroll) - 1) * ch_groups

        # Height stride skips full rows of the padded scan.
        height_gap_firings = max(0, stride_h - 1) * padded_words_per_row * ch_groups

        restart_gap_firings = 0
        if not is_pointwise:
            restart_gap_firings = (
                (max(0, fh - 1) * padded_words_per_row)
                + (self._ceil_div(pads[1] + fw_extended, width_unroll) - 1)
            ) * ch_groups

        restart_bound = max(
            2,
            self._ceil_div(restart_gap_firings * output_window_groups_total, model_ii),
        )
        stride_gap_firings = max(width_gap_firings, height_gap_firings)
        stride_bound = max(
            2,
            self._ceil_div(stride_gap_firings * output_window_groups_total, input_words_total),
        )
        return max(restart_bound, stride_bound)

    def get_nodeattr_types(self):
        return {
            "kernel_shape": ("ints", True, [1, 1]),
            "strides": ("ints", True, [1, 1]),
            "pads": ("ints", True, [0, 0, 0, 0]),
            "dilation": ("ints", True, [1, 1]),
            "channel_unroll": ("i", False, 1),
            "width_unroll": ("i", False, 1),
            "in_stream_array": ("i", False, 1),
            "out_stream_array": ("i", False, 1),
            "in_word_array": ("i", False, 1),
            "out_word_array": ("i", False, 1),
            "pad_value": ("f", False, 0.0),
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
        context[node.output[0]] = context[node.input[0]]

    def verify_node(self):
        pass

    def accepted_input_layout(self) -> tuple | None:
        return (0, 2, 3, 1)

    def produced_output_layout(self, input_layout: tuple | None) -> tuple | None:
        return (0, 2, 3, 1)

    def __get_stream_name(self, name: str) -> str:
        return f"{name}_stream"

    def __get_pad_value(self, pad_value, output_quant):
        if output_quant.signed:
            if pad_value == float("-inf"):
                return f"{int(-(2 ** (output_quant.bitwidth - 1)))}"
            return str(int(pad_value))
        return str(int(pad_value) if pad_value >= 0 else int(0))

    def lower_to_hls(self, model: ModelWrapper, hls_tag: int):
        hls_kernels = []
        fifos = {}

        output_type = require_tensor_type(model, self.onnx_node.output[0])
        output_layout = require_tensor_layout(model, self.onnx_node.output[0])
        output_shape = self.require_4d_output_shape(model, 0, output_layout)

        fh = self.get_nodeattr("kernel_shape")[0]
        fw = self.get_nodeattr("kernel_shape")[1]
        stride_w = self.get_nodeattr("strides")[1]
        width_unroll = self.get_nodeattr("width_unroll")
        output_fifo_depth = self._get_output_fifo_depth(model)
        fw_extended = fw + (width_unroll - 1) * stride_w

        output_name = self.__get_stream_name(self.onnx_node.output[0])
        for i in range(fh * fw_extended):
            fifos[f"{output_name}_{i}_"] = TensorFifo(
                depth=output_fifo_depth,
                hls_type=get_word_type(output_type, self.get_nodeattr("out_word_array")),
                n_array=fh * fw_extended,
            )

        input_names = [
            f"{self.__get_stream_name(self.onnx_node.input[0])}_{i}_"
            for i in range(width_unroll)
        ]
        output_names = [f"{output_name}_{i}_" for i in range(fh * fw_extended)]

        function_args = (
            ("i_data", "hls::stream<TWord>"),
            ("o_data", "hls::stream<TWord>"),
        )

        run = cpp_function(
            name=f"{self.onnx_node.name}.run",
            return_type="void",
            arguments=function_args,
        )
        run_call = run.generate_call(
            [hls_tag],
            self.__get_stream_name(self.onnx_node.input[0]),
            output_name,
        )

        step = cpp_function(
            name=f"{self.onnx_node.name}.step",
            return_type="ActorStatus",
            arguments=function_args,
        )
        step_call = step.generate_call(
            [],
            self.__get_stream_name(self.onnx_node.input[0]),
            output_name,
        )

        circular_linebuffer = cpp_object(
            "StreamingCircularLineBuffer",
            self.onnx_node.name,
            template_args=[
                (get_word_type(output_type, self.get_nodeattr("in_word_array")), "TWord"),
                (output_type.get_hls_data_type(), "TData"),
                (output_shape[-3], "IN_HEIGHT"),
                (output_shape[-2], "IN_WIDTH"),
                (output_shape[-1], "IN_CH"),
                (fh, "FH"),
                (fw, "FW"),
                (self.get_nodeattr("strides")[0], "STRIDE_H"),
                (self.get_nodeattr("strides")[1], "STRIDE_W"),
                (self.get_nodeattr("dilation")[0], "DILATION_H"),
                (self.get_nodeattr("dilation")[1], "DILATION_W"),
                (self.get_nodeattr("pads")[0], "PAD_T"),
                (self.get_nodeattr("pads")[1], "PAD_L"),
                (self.get_nodeattr("pads")[2], "PAD_B"),
                (self.get_nodeattr("pads")[3], "PAD_R"),
                (width_unroll, "W_PAR"),
                (self.get_nodeattr("channel_unroll"), "CH_PAR"),
                (
                    self.__get_pad_value(self.get_nodeattr("pad_value"), output_type),
                    "PAD_VALUE",
                ),
            ],
        )

        hls_kernels.append(
            HLSKernel.make_node(
                inputs=input_names,
                outputs=output_names,
                name=f"{self.onnx_node.name}_hls",
                domain="nn2fpga.compiler.custom_op",
                original_op_type="StreamingCircularLineBuffer",
                hls_object_name=self.onnx_node.name,
                hls_tag=hls_tag,
                hls_variable_declarations="",
                hls_run_call=run_call,
                hls_step_call=step_call,
                hls_object_declaration=circular_linebuffer.generate_declaration(),
            )
        )
        hls_tag += 1

        return hls_kernels, [], fifos, hls_tag

    def get_latency(self, model: ModelWrapper) -> int:
        input_layout = require_tensor_layout(model, self.onnx_node.input[0])
        input_shape = self.require_4d_input_shape(model, 0, input_layout)

        pads = self.get_nodeattr("pads")
        padded_height = input_shape[-3] + pads[0] + pads[2]
        padded_width = input_shape[-2] + pads[1] + pads[3]
        width_unroll = self.get_nodeattr("width_unroll")
        channel_unroll = self.get_nodeattr("channel_unroll")

        return padded_height * (padded_width // width_unroll) * (
            input_shape[-1] // channel_unroll
        )

    def get_brams(self, model: ModelWrapper) -> int:
        return 0

    def get_dsps(self, model: ModelWrapper) -> int:
        return 0

    def has_linebuffer(self) -> bool:
        return False
