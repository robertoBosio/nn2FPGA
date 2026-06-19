import numpy as np
import csnake
from tests.hw.library.base_hls_test import BaseHLSTest


class TestTransposeReplayPacked(BaseHLSTest):

    @property
    def operator_filename(self):
        return "YoloAttention/Transpose"

    @property
    def unit_filename(self):
        return "YoloAttention/TransposeReplayPacked"

    def generate_config_file(self, config_dict):
        data_unsigned = bool(config_dict.get("DATA_IS_UNSIGNED", False))
        data_bits = int(config_dict["DATA_DATAWIDTH"])
        np_data_type = self.get_numpy_dtype(data_bits, data_unsigned)

        in_info = np.iinfo(np_data_type)
        input_tensor = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,
            size=(1, config_dict["IN_HEIGHT"], config_dict["IN_WIDTH"]),
            dtype=np_data_type,
        )

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            if isinstance(value, bool):
                value_str = "true" if value else "false"
                cwr.add_line(f"const bool {key} = {value_str};")
            else:
                cwr.add_line(f"const int {key} = {int(value)};")

        typedef_suffix = "u" if data_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{data_bits}> TData;")
        cwr.add_line("using TInputWord = std::array<TData, 1>;")
        cwr.add_line("using TOutputWord = std::array<TData, REDUCE_PAR>;")

        cwr.add_lines(
            csnake.Variable(
                "tensor",
                primitive="TData",
                value=input_tensor,
            ).generate_initialization()
        )

        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_8bit_po2_signed(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "DATA_DATAWIDTH": 8,
            "DATA_IS_UNSIGNED": False,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 8,
            "DIM_P": 4,
            "REDUCE_PAR": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
