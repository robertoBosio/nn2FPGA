import csnake
import numpy as np

from .base_hls_test import BaseHLSTest


class TestStreamingWindowToChannels(BaseHLSTest):

    @property
    def operator_filename(self):
        return "StreamingWindowToChannels"

    @property
    def unit_filename(self):
        return "StreamingWindowToChannels"

    def generate_config_file(self, config_dict, **kwargs):
        input_dtype = self.get_numpy_dtype(
            config_dict["INPUT_DATAWIDTH"], is_unsigned=False
        )
        output_dtype = self.get_numpy_dtype(
            config_dict["OUTPUT_DATAWIDTH"], is_unsigned=False
        )
        channel_groups = (
            config_dict["OUT_DIM2"]
            // (config_dict["FH"] * config_dict["FW"])
            // config_dict["DIM2_UNROLL"]
        )
        fw_expand = config_dict["FW"] + (
            config_dict["DIM1_UNROLL"] - 1
        ) * config_dict["STRIDE_W"]
        window_streams = config_dict["FH"] * fw_expand
        input_tensor = np.random.randint(
            np.iinfo(input_dtype).min,
            np.iinfo(input_dtype).max + 1,
            size=(
                config_dict["OUT_DIM0"],
                config_dict["OUT_DIM1"] // config_dict["DIM1_UNROLL"],
                channel_groups,
                window_streams,
                config_dict["DIM2_UNROLL"],
            ),
            dtype=input_dtype,
        )
        output_tensor = np.empty(
            (
                config_dict["OUT_DIM0"],
                config_dict["OUT_DIM1"],
                config_dict["OUT_DIM2"],
            ),
            dtype=output_dtype,
        )
        for out_h in range(config_dict["OUT_DIM0"]):
            for width_group in range(input_tensor.shape[1]):
                for output_group in range(
                    config_dict["OUT_DIM2"] // config_dict["DIM2_UNROLL"]
                ):
                    window_position = output_group // channel_groups
                    channel_group = output_group % channel_groups
                    fh = window_position % config_dict["FH"]
                    fw = window_position // config_dict["FH"]
                    for width_lane in range(config_dict["DIM1_UNROLL"]):
                        stream_index = (
                            fh * fw_expand
                            + width_lane * config_dict["STRIDE_W"]
                            + fw
                        )
                        output_tensor[
                            out_h,
                            width_group * config_dict["DIM1_UNROLL"] + width_lane,
                            output_group
                            * config_dict["DIM2_UNROLL"] : (output_group + 1)
                            * config_dict["DIM2_UNROLL"],
                        ] = input_tensor[
                            out_h,
                            width_group,
                            channel_group,
                            stream_index,
                            :,
                        ]

        cwr = csnake.CodeWriter()
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.include("DequantQuant.hpp")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            cwr.add_line(f"const int {key} = {int(value)};")
        cwr.add_line(
            f"typedef ap_int<{config_dict['INPUT_DATAWIDTH']}> TInput;"
        )
        cwr.add_line(
            f"typedef ap_int<{config_dict['OUTPUT_DATAWIDTH']}> TOutput;"
        )
        cwr.add_line(
            "using TInputWord = std::array<TInput, DIM2_UNROLL>;"
        )
        cwr.add_line(
            "using TOutputWord = std::array<TOutput, DIM2_UNROLL>;"
        )
        cwr.add_line("using Quantizer = DequantQuantPo2<0, TInput, TOutput>;")
        cwr.add_lines(
            csnake.Variable(
                "input_tensor", primitive="TInput", value=input_tensor
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "output_tensor", primitive="TOutput", value=output_tensor
            ).generate_initialization()
        )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_focus_2x2_stride2_wpar2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "OUT_DIM0": 2,
            "OUT_DIM1": 4,
            "OUT_DIM2": 12,
            "FH": 2,
            "FW": 2,
            "STRIDE_W": 2,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 3,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
