import csnake
from .base_hls_test import BaseHLSTest

class TestStreamingTensorDuplicator(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "StreamingTensorDuplicator"

    @property
    def unit_filename(self) -> str:
        return "StreamingTensorDuplicator"

    def generate_config_file(self, config_dict):

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
        cwr.add_line(f"typedef ap_int<{config_dict['DATAWIDTH']}> TInput;")
        cwr.add_line(f"using TWord = std::array<TInput, {config_dict['DIM2_UNROLL']}>;")
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_2copies(self, hls_steps):
        config_dict = {
            "DATAWIDTH": 8,
            "DIM0": 4,
            "DIM1": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict,
            hls_steps,
        )
