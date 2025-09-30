import csnake
from .base_hls_test import BaseHLSTest

class TestStreamingConv(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
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
        cwr.add_line(
            f"typedef DequantQuantPo2<0, TOutput, TOutput> Quantizer;"
        )
        cwr.add_line(f"using TInputWord = std::array<TInput, {config_dict['IN_CH_PAR']}>;")
        cwr.add_line(f"using TOutputWord = std::array<TOutput, {config_dict['OUT_CH_PAR']}>;")
        cwr.add_line(
            f"using BandwidthAdjust = {class_name}<TInputWord, TInput, TOutputWord, TOutput, Quantizer, IN_HEIGHT, IN_WIDTH, IN_CH, IN_W_PAR, OUT_W_PAR, IN_CH_PAR, OUT_CH_PAR>;"
        )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_increase_WPAR(self, hls_steps):
        config_dict = {
            "IN_DATAWIDTH": 8,
            "OUT_DATAWIDTH": 8,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 4,
            "IN_CH": 4,
            "OUT_CH": 4,
            "IN_CH_PAR": 2,
            "OUT_CH_PAR": 2,
            "IN_W_PAR": 2,
            "OUT_W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict,
            hls_steps,
            workdir=".",
            class_name="BandwidthAdjustIncreaseStreams",
        )

    def test_decrease_WPAR(self, hls_steps):
        config_dict = {
            "IN_DATAWIDTH": 8,
            "OUT_DATAWIDTH": 8,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 4,
            "IN_CH": 4,
            "OUT_CH": 4,
            "IN_CH_PAR": 2,
            "OUT_CH_PAR": 2,
            "IN_W_PAR": 4,
            "OUT_W_PAR": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict,
            hls_steps,
            workdir=".",
            class_name="BandwidthAdjustDecreaseStreams",
        )

    def test_increase_CHPAR(self, hls_steps):
        config_dict = {
            "IN_DATAWIDTH": 8,
            "OUT_DATAWIDTH": 8,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 4,
            "IN_CH": 4,
            "OUT_CH": 4,
            "IN_CH_PAR": 2,
            "OUT_CH_PAR": 4,
            "IN_W_PAR": 2,
            "OUT_W_PAR": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict,
            hls_steps,
            workdir=".",
            class_name="BandwidthAdjustIncreaseChannels",
        )

    def test_decrease_CHPAR(self, hls_steps):
        config_dict = {
            "IN_DATAWIDTH": 8,
            "OUT_DATAWIDTH": 8,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 4,
            "IN_CH": 4,
            "OUT_CH": 4,
            "IN_CH_PAR": 4,
            "OUT_CH_PAR": 2,
            "IN_W_PAR": 2,
            "OUT_W_PAR": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict,
            hls_steps,
            workdir=".",
            class_name="BandwidthAdjustDecreaseChannels",
        )
