import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest

class TestStreamingWindowBuffer(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return ["StreamingWindowSelector", "StreamingPad"]

    @property
    def unit_filename(self) -> str:
        return "StreamingWindowBuffer"

    def generate_config_file(self, config_dict, **kwargs):

        # random tensors
        input_tensor = np.random.randint(
            -128,
            127,
            size=(1, config_dict["IN_CH"], config_dict["IN_HEIGHT"], config_dict["IN_WIDTH"]),
            dtype=np.int8,
        )
        
        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            cwr.add_line(f"const int {key} = {value};")
        cwr.add_line(f"typedef ap_int<{config_dict['INPUT_DATAWIDTH']}> TData;")
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

    def test_3x3_window_wpar4(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "IN_HEIGHT": 112,
            "IN_WIDTH": 112,
            "IN_CH": 32,
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
        }
        self.run(
            config_dict,
            hls_steps,
        )
