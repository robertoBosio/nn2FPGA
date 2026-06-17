import csnake
import numpy as np

from .base_hls_test import BaseHLSTest


class TestStreamingCircularLineBuffer(BaseHLSTest):
    @property
    def operator_filename(self):
        return "StreamingCircularLineBuffer"

    @property
    def unit_filename(self):
        return "StreamingCircularLineBuffer"

    def generate_config_file(self, config_dict, **kwargs):
        config_dict = {"PAD_VALUE": 0, "PIPELINE_DEPTH": 1, **config_dict}
        data_unsigned = bool(config_dict.get("DATA_IS_UNSIGNED", False))
        data_bits = int(config_dict["DATA_DATAWIDTH"])
        np_data_type = self.get_numpy_dtype(data_bits, data_unsigned)

        in_info = np.iinfo(np_data_type)
        input_tensor = np.random.randint(
            in_info.min,
            in_info.max + 1,
            size=(
                1,
                config_dict["IN_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ),
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
        cwr.add_line("typedef std::array<TData, CH_PAR> TWord;")
        cwr.add_lines(
            csnake.Variable(
                "input_tensor",
                primitive="TData",
                value=input_tensor,
            ).generate_initialization()
        )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_3x3_stride1_wpar1(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "DATA_DATAWIDTH": 8,
            "DATA_IS_UNSIGNED": False,
            "IN_HEIGHT": 6,
            "IN_WIDTH": 7,
            "IN_CH": 6,
            "FH": 3,
            "FW": 3,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PAD_T": 0,
            "PAD_B": 0,
            "PAD_L": 0,
            "PAD_R": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "CH_PAR": 2,
            "W_PAR": 1,
        }
        self.run(config_dict, hls_steps)

    def test_3x3_stride1_wpar2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "DATA_DATAWIDTH": 8,
            "DATA_IS_UNSIGNED": False,
            "IN_HEIGHT": 6,
            "IN_WIDTH": 8,
            "IN_CH": 8,
            "FH": 3,
            "FW": 3,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PAD_T": 0,
            "PAD_B": 0,
            "PAD_L": 0,
            "PAD_R": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "CH_PAR": 4,
            "W_PAR": 2,
        }
        self.run(config_dict, hls_steps)

    def test_3x3_stride2_wpar2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "DATA_DATAWIDTH": 8,
            "DATA_IS_UNSIGNED": False,
            "IN_HEIGHT": 8,
            "IN_WIDTH": 10,
            "IN_CH": 4,
            "FH": 3,
            "FW": 3,
            "STRIDE_H": 2,
            "STRIDE_W": 2,
            "PAD_T": 0,
            "PAD_B": 0,
            "PAD_L": 0,
            "PAD_R": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "CH_PAR": 2,
            "W_PAR": 2,
        }
        self.run(config_dict, hls_steps)

    def test_3x3_pad1_stride1_wpar1(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "DATA_DATAWIDTH": 8,
            "DATA_IS_UNSIGNED": False,
            "IN_HEIGHT": 6,
            "IN_WIDTH": 7,
            "IN_CH": 6,
            "FH": 3,
            "FW": 3,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PAD_T": 1,
            "PAD_B": 1,
            "PAD_L": 1,
            "PAD_R": 1,
            "PAD_VALUE": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "CH_PAR": 2,
            "W_PAR": 1,
        }
        self.run(config_dict, hls_steps)

    def test_3x3_pad1_stride1_wpar2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "DATA_DATAWIDTH": 8,
            "DATA_IS_UNSIGNED": False,
            "IN_HEIGHT": 6,
            "IN_WIDTH": 8,
            "IN_CH": 8,
            "FH": 3,
            "FW": 3,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PAD_T": 1,
            "PAD_B": 1,
            "PAD_L": 1,
            "PAD_R": 1,
            "PAD_VALUE": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "CH_PAR": 4,
            "W_PAR": 2,
        }
        self.run(config_dict, hls_steps)

    def test_3x3_pad1_stride2_wpar2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "DATA_DATAWIDTH": 8,
            "DATA_IS_UNSIGNED": False,
            "IN_HEIGHT": 8,
            "IN_WIDTH": 12,
            "IN_CH": 4,
            "FH": 3,
            "FW": 3,
            "STRIDE_H": 2,
            "STRIDE_W": 2,
            "PAD_T": 1,
            "PAD_B": 1,
            "PAD_L": 1,
            "PAD_R": 1,
            "PAD_VALUE": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "CH_PAR": 2,
            "W_PAR": 2,
        }
        self.run(config_dict, hls_steps)

    def test_2x1_stride1_wpar4(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "DATA_DATAWIDTH": 8,
            "DATA_IS_UNSIGNED": False,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 8,
            "IN_CH": 8,
            "FH": 2,
            "FW": 1,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PAD_T": 0,
            "PAD_B": 0,
            "PAD_L": 0,
            "PAD_R": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "CH_PAR": 2,
            "W_PAR": 4,
        }
        self.run(config_dict, hls_steps)

    def test_1x1_stride1_wpar1(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "DATA_DATAWIDTH": 8,
            "DATA_IS_UNSIGNED": False,
            "IN_HEIGHT": 5,
            "IN_WIDTH": 7,
            "IN_CH": 6,
            "FH": 1,
            "FW": 1,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PAD_T": 0,
            "PAD_B": 0,
            "PAD_L": 0,
            "PAD_R": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "CH_PAR": 2,
            "W_PAR": 1,
        }
        self.run(config_dict, hls_steps)

    def test_1x1_stride2_wpar1(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "DATA_DATAWIDTH": 8,
            "DATA_IS_UNSIGNED": False,
            "IN_HEIGHT": 8,
            "IN_WIDTH": 8,
            "IN_CH": 8,
            "FH": 1,
            "FW": 1,
            "STRIDE_H": 2,
            "STRIDE_W": 2,
            "PAD_T": 0,
            "PAD_B": 0,
            "PAD_L": 0,
            "PAD_R": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "CH_PAR": 4,
            "W_PAR": 1,
        }
        self.run(config_dict, hls_steps)
