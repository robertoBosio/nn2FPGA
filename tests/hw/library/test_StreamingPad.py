import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest

class TestStreamingPad(BaseHLSTest):

    @property
    def operator_filename(self):
        return "StreamingPad"

    @property
    def unit_filename(self):
        return "StreamingPad"

    def generate_config_file(self, config_dict, **kwargs):

        data_unsigned = bool(config_dict.get("DATA_IS_UNSIGNED", False))
        data_bits = int(config_dict["DATA_DATAWIDTH"])
        np_data_type = self.get_numpy_dtype(data_bits, data_unsigned)

        # random tensors
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
            if key in ["X_SCALE", "W_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {float(value)}f;")
            else:
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                    cwr.add_line(f"const bool {key} = {value_str};")
                else:
                    cwr.add_line(f"const int {key} = {int(value)};")
        typedef_suffix = "u" if data_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{data_bits}> TData;")
        cwr.add_line(f"typedef std::array<TData, CH_PAR> TWord;")
        cwr.add_lines(
            csnake.Variable(
                "input_tensor",
                primitive=f"TData",
                value=input_tensor,
            ).generate_initialization()
        )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_3x3_asympadding_0pad(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "DATA_DATAWIDTH": 8,
            "DATA_IS_UNSIGNED": False,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 4,
            "IN_CH": 6,
            "FH": 3,
            "FW": 3,
            "STRIDE_H": 2,
            "STRIDE_W": 2,
            "PAD_T": 1,
            "PAD_B": 0,
            "PAD_L": 1,
            "PAD_R": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "CH_PAR": 3,
            "W_PAR": 2,
            "PAD_VALUE": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_3x3_sympadding_0pad(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "DATA_DATAWIDTH": 8,
            "DATA_IS_UNSIGNED": False,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 4,
            "IN_CH": 4,
            "FH": 3,
            "FW": 3,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PAD_T": 1,
            "PAD_B": 1,
            "PAD_L": 1,
            "PAD_R": 1,
            "DIL_H": 1,
            "DIL_W": 1,
            "CH_PAR": 2,
            "W_PAR": 2,
            "PIPELINE_DEPTH": 5,
            "PAD_VALUE": 0,
        }
        self.run(config_dict, hls_steps)

    def test_3x3_sympadding_wpar4(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "DATA_DATAWIDTH": 8,
            "DATA_IS_UNSIGNED": False,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 4,
            "IN_CH": 4,
            "FH": 3,
            "FW": 3,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PAD_T": 1,
            "PAD_B": 1,
            "PAD_L": 1,
            "PAD_R": 1,
            "DIL_H": 1,
            "DIL_W": 1,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
            "PAD_VALUE": -128,
        }
        self.run(config_dict, hls_steps)

    def test_5x5_sympadding_0pad(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "DATA_DATAWIDTH": 8,
            "DATA_IS_UNSIGNED": False,
            "IN_HEIGHT": 7,
            "IN_WIDTH": 7,
            "IN_CH": 4,
            "FH": 5,
            "FW": 5,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PAD_T": 1,
            "PAD_B": 1,
            "PAD_L": 1,
            "PAD_R": 1,
            "DIL_H": 1,
            "DIL_W": 1,
            "CH_PAR": 2,
            "W_PAR": 1,
            "PIPELINE_DEPTH": 5,
            "PAD_VALUE": 0,
        }
        self.run(config_dict, hls_steps)