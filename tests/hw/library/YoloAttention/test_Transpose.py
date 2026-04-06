import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from tests.base_hls_test import BaseHLSTest

class TestTranspose(BaseHLSTest):

    @property
    def operator_filename(self):
        return "YoloAttention/Transpose"

    @property
    def unit_filename(self):
        return "YoloAttention/Transpose"

    def generate_config_file(self, config_dict):
        """
        Generate a self-contained C++ test config (input/output tensors and operator typedefs),
        supporting signed/unsigned input and signed/unsigned output independently.

        Expected config_dict keys (in addition to your existing ones):
        INPUT_IS_UNSIGNED (bool, optional; default False)
        OUTPUT_IS_UNSIGNED (bool, optional; default False)
        INPUT_DATAWIDTH / OUTPUT_DATAWIDTH (int)
        """

        data_unsigned = bool(config_dict.get("DATA_IS_UNSIGNED", False))
        data_bits = int(config_dict["DATA_DATAWIDTH"])
        np_data_type = self.get_numpy_dtype(data_bits, data_unsigned)

        # Random input tensor in correct integer domain/range
        in_info = np.iinfo(np_data_type)
        input_tensor = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,  # randint upper bound is exclusive
            size=(
                1,
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
                config_dict["IN_CH"],
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
            if key in ["V_SCALE", "P_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {float(value)}f;")
            else:
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                    cwr.add_line(f"const bool {key} = {value_str};")
                else:
                    cwr.add_line(f"const int {key} = {int(value)};")

        typedef_suffix = "u" if data_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{data_bits}> Tdata;")
        cwr.add_line(f"using TWord = std::array<Tdata, 1>;")

        cwr.add_lines(
            csnake.Variable(
                "tensor",
                primitive="Tdata",
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
            "IN_HEIGHT": 64,
            "IN_WIDTH": 400,
            "IN_CH": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
