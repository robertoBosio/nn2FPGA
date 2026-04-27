import csnake
from .base_hls_test import BaseHLSTest


class TestStreamingConv(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "BandwidthAdjust"

    @property
    def unit_filename(self) -> str:
        return "BandwidthAdjust"

    def generate_config_file(
        self, config_dict, class_name: str = "BandwidthAdjustIncreaseStreams"
    ):

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.include("DequantQuant.hpp")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            if key in ["X_SCALE", "W_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {value}f;")
            else:
                cwr.add_line(f"const int {key} = {value};")
        cwr.add_line(f"typedef ap_int<{config_dict['IN_DATAWIDTH']}> TInput;")
        cwr.add_line(f"typedef ap_int<{config_dict['OUT_DATAWIDTH']}> TOutput;")
        cwr.add_line(f"typedef DequantQuantPo2<0, TOutput, TOutput> Quantizer;")
        cwr.add_line(
            f"using TInputWord = std::array<TInput, {config_dict['IN_DIM2_UNROLL']}>;"
        )
        cwr.add_line(
            f"using TOutputWord = std::array<TOutput, {config_dict['OUT_DIM2_UNROLL']}>;"
        )
        cwr.add_line(
            f"using BandwidthAdjust = {class_name}<TInputWord, TInput, TOutputWord, TOutput, Quantizer, IN_DIM0, IN_DIM1, IN_DIM2, IN_DIM1_UNROLL, OUT_DIM1_UNROLL, IN_DIM2_UNROLL, OUT_DIM2_UNROLL>;"
        )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_increase_streams(self, hls_steps):
        config_dict = {
            "IN_DATAWIDTH": 8,
            "OUT_DATAWIDTH": 8,
            "IN_DIM0": 4,
            "IN_DIM1": 4,
            "IN_DIM2": 4,
            "OUT_DIM2": 4,
            "IN_DIM2_UNROLL": 2,
            "OUT_DIM2_UNROLL": 2,
            "IN_DIM1_UNROLL": 2,
            "OUT_DIM1_UNROLL": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict,
            hls_steps,
            workdir=".",
            class_name="BandwidthAdjustIncreaseStreams",
        )

    def test_decrease_streams(self, hls_steps):
        config_dict = {
            "IN_DATAWIDTH": 8,
            "OUT_DATAWIDTH": 8,
            "IN_DIM0": 4,
            "IN_DIM1": 4,
            "IN_DIM2": 4,
            "OUT_DIM2": 4,
            "IN_DIM2_UNROLL": 2,
            "OUT_DIM2_UNROLL": 2,
            "IN_DIM1_UNROLL": 4,
            "OUT_DIM1_UNROLL": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict,
            hls_steps,
            workdir=".",
            class_name="BandwidthAdjustDecreaseStreams",
        )

    def test_increase_word(self, hls_steps):
        config_dict = {
            "IN_DATAWIDTH": 8,
            "OUT_DATAWIDTH": 8,
            "IN_DIM0": 4,
            "IN_DIM1": 4,
            "IN_DIM2": 4,
            "OUT_DIM2": 4,
            "IN_DIM2_UNROLL": 2,
            "OUT_DIM2_UNROLL": 4,
            "IN_DIM1_UNROLL": 2,
            "OUT_DIM1_UNROLL": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict,
            hls_steps,
            workdir=".",
            class_name="BandwidthAdjustIncreaseWord",
        )

    def test_decrease_word(self, hls_steps):
        config_dict = {
            "IN_DATAWIDTH": 8,
            "OUT_DATAWIDTH": 8,
            "IN_DIM0": 4,
            "IN_DIM1": 4,
            "IN_DIM2": 4,
            "OUT_DIM2": 4,
            "IN_DIM2_UNROLL": 4,
            "OUT_DIM2_UNROLL": 2,
            "IN_DIM1_UNROLL": 2,
            "OUT_DIM1_UNROLL": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict,
            hls_steps,
            workdir=".",
            class_name="BandwidthAdjustDecreaseWord",
        )
