import csnake
from .base_hls_test import BaseHLSTest

class TestDequantQuant(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "DequantQuant"
    
    @property
    def unit_filename(self) -> str:
        return "DequantQuant"

    def generate_config_file(self, config_dict):

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            if key not in ["ACC_SIGNED", "OUT_SIGNED"]:
                cwr.add_line(f"const int {key} = {value};")
        cwr.add_line(
            "using TAcc = "
            + (
                "ap_int<ACC_DATAWIDTH>;"
                if config_dict.get("ACC_SIGNED", True)
                else "ap_uint<ACC_DATAWIDTH>;"
            )
        )
        cwr.add_line(
            "using TOut = "
            + (
                "ap_int<OUT_DATAWIDTH>;"
                if config_dict.get("OUT_SIGNED", True)
                else "ap_uint<OUT_DATAWIDTH>;"
            )
        )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_basic_shift(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "SHIFT": 2,
            "INPUT": -63,
            "EXPECTED": -16,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 1,
        }
        self.run(
            config_dict,
            hls_steps,
        )

    def test_zero_shift(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "SHIFT": 0,
            "INPUT": 64,
            "EXPECTED": 64,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 1,
        }
        self.run(
            config_dict,
            hls_steps,
        )

    def test_negative_shift(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "SHIFT": -2,
            "INPUT": 16,
            "EXPECTED": 64,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 1,
        }
        self.run(
            config_dict,
            hls_steps,
        )

    def test_rounddown_to_even(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "SHIFT": 1,
            "INPUT": 5,
            "EXPECTED": 2,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 1,
        }
        self.run(
            config_dict,
            hls_steps,
        )

    def test_roundup_to_even(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "SHIFT": 1,
            "INPUT": 7,
            "EXPECTED": 4,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 1,
        }
        self.run(
            config_dict,
            hls_steps,
        )

    def test_rounddown_to_even_negative(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "SHIFT": 1,
            "INPUT": -5,
            "EXPECTED": -2,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 1,
        }
        self.run(
            config_dict,
            hls_steps,
        )

    def test_roundup_to_even_negative(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "SHIFT": 1,
            "INPUT": -7,
            "EXPECTED": -4,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 1,
        }
        self.run(
            config_dict,
            hls_steps,
        )

    def test_clamp_positive(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "SHIFT": 0,
            "INPUT": 300,
            "EXPECTED": 127,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 1,
        }
        self.run(
            config_dict,
            hls_steps,
        )

    def test_clamp_negative(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "SHIFT": 0,
            "INPUT": -300,
            "EXPECTED": -128,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 1,
        }
        self.run(
            config_dict,
            hls_steps,
        )

    def test_clamp_unsigned_positive_input(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "SHIFT": 0,
            "INPUT": 300,
            "EXPECTED": 255,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 0,
        }
        self.run(
            config_dict,
            hls_steps,
        )
    
    def test_clamp_unsigned_negative_input(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "SHIFT": 0,
            "INPUT": -300,
            "EXPECTED": 0,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 0,
        }
        self.run(
            config_dict,
            hls_steps,
        )