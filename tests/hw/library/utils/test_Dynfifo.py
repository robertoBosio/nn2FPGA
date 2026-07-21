import csnake
import numpy as np
from tests.hw.library.base_hls_test import BaseHLSTest

class TestDynfifo(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "utils/dynfifo_utils"

    @property
    def unit_filename(self) -> str:
        return "Utils/UnitDynfifo"

    def generate_config_file(self, config_dict):

        in_unsigned = bool(config_dict["IN_UNSIGNED"])
        in_bits = config_dict["DATAWIDTH"]
        np_type = self.get_numpy_dtype(in_bits, in_unsigned)

        # Random input tensor in correct integer domain/range
        in_info = np.iinfo(np_type)
        input_tensor = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,  # randint upper bound is exclusive
            size=(
                1,
                config_dict["DIM2"],
                config_dict["DIM0"],
                config_dict["DIM1"],
            ),
            dtype=np_type,
        )
        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.include("DequantQuant.hpp")
        cwr.add_line("namespace test_config {")
        cwr.indent()

        config_dict["AXIWORD_PAR"] = config_dict["AXI_DATAWIDTH"] // (config_dict["DATAWIDTH"])
        config_dict["PAR"] = config_dict["DIM2_UNROLL"]
        config_dict["DIM"] = (
            config_dict["DIM0"]
            * config_dict["DIM1"]
            * config_dict["DIM2"]
            // config_dict["AXIWORD_PAR"]
        )
        for key, value in config_dict.items():
            if key in ["X_SCALE", "W_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {value}f;")
            elif isinstance(value, str):
                continue # Skip string values (like IN_TYPE) as they are not needed in the C++ config.
            elif isinstance(value, bool):
                value_str = "true" if value else "false"
                cwr.add_line(f"const bool {key} = {value_str};")
            else:   
                cwr.add_line(f"const int {key} = {value};")
        cwr.add_line(f"typedef {config_dict['DATATYPE']} TData;")
        cwr.add_line(f"typedef std::array<TData, {config_dict['DIM2_UNROLL']}> TWord;")
        cwr.add_line(f"typedef {config_dict['AXI_DATATYPE']} TAXIWord;")

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

    def test_axi128_par2(self, hls_steps):
        config_dict = {
            "AXI_DATATYPE": "ap_uint<128>",
            "AXI_DATAWIDTH": 128,
            "DATATYPE": "ap_uint<8>",
            "IN_UNSIGNED": True,
            "DATAWIDTH": 8,
            "DIM0": 4,
            "DIM1": 8,
            "DIM2": 16,
            "DIM1_UNROLL": 1,
            "DIM2_UNROLL": 2,
            "BURST_SIZE": 4,
            "DEPTH": 32,
            "PIPELINE_DEPTH": 4,
        }
        self.run(config_dict, hls_steps)
    
    def test_axi128_float(self, hls_steps):
        config_dict = {
            "AXI_DATATYPE": "ap_uint<128>",
            "AXI_DATAWIDTH": 128,
            "DATATYPE": "ap_float<32, 8>",
            "IN_UNSIGNED": False, # Ignored for float, but set to False for clarity.
            "DATAWIDTH": 32,
            "DIM0": 4,
            "DIM1": 8,
            "DIM2": 16,
            "DIM1_UNROLL": 1,
            "DIM2_UNROLL": 2,
            "BURST_SIZE": 4,
            "DEPTH": 32,
            "PIPELINE_DEPTH": 4,
        }
        self.run(config_dict, hls_steps)
