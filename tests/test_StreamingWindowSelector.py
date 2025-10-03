import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest

class TestStreamingWindowSelector(BaseHLSTest):

    @property
    def operator_filename(self):
        return "StreamingWindowSelector"

    @property
    def unit_filename(self):
        return "StreamingWindowSelector"
    
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
        cwr.add_line(f"typedef std::array<ap_int<{config_dict['INPUT_DATAWIDTH']}>, CH_PAR> TWord;")
        cwr.add_lines(
            csnake.Variable(
                "input_tensor",
                primitive=f"ap_int<{config_dict['INPUT_DATAWIDTH']}>",
                value=input_tensor,
            ).generate_initialization()
        )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code
    
    def test_3x3_bottomright_corner(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 4,
            "IN_CH": 6,
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
            "POS_H": 2,
            "POS_W": 3,
            "CH_PAR": 3,
            "W_PAR": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_topleft_corner(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 4,
            "IN_CH": 6,
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
            "POS_H": 0,
            "POS_W": 0,
            "CH_PAR": 3,
            "W_PAR": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_topleft_corner_stride2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 8,
            "IN_CH": 6,
            "FH": 3,
            "FW": 3,
            "STRIDE_H": 2,
            "STRIDE_W": 2,
            "PAD_T": 1,
            "PAD_B": 1,
            "PAD_L": 1,
            "PAD_R": 1,
            "DIL_H": 1,
            "DIL_W": 1,
            "POS_H": 2,
            "POS_W": 4,
            "CH_PAR": 3,
            "W_PAR": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel0_wpar4(self, hls_steps):
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
            "POS_H": 0,
            "POS_W": 0,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel1_wpar4(self, hls_steps):
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
            "POS_H": 0,
            "POS_W": 1,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel2_wpar4(self, hls_steps):
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
            "POS_H": 0,
            "POS_W": 2,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel3_wpar4(self, hls_steps):
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
            "POS_H": 0,
            "POS_W": 3,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel4_wpar4(self, hls_steps):
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
            "POS_H": 0,
            "POS_W": 4,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel5_wpar4(self, hls_steps):
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
            "POS_H": 0,
            "POS_W": 5,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel6_wpar4(self, hls_steps):
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
            "POS_H": 1,
            "POS_W": 0,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel7_wpar4(self, hls_steps):
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
            "POS_H": 1,
            "POS_W": 1,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel8_wpar4(self, hls_steps):
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
            "POS_H": 1,
            "POS_W": 2,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel9_wpar4(self, hls_steps):
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
            "POS_H": 1,
            "POS_W": 3,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel10_wpar4(self, hls_steps):
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
            "POS_H": 1,
            "POS_W": 4,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel11_wpar4(self, hls_steps):
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
            "POS_H": 1,
            "POS_W": 5,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel12_wpar4(self, hls_steps):
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
            "POS_H": 2,
            "POS_W": 0,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel13_wpar4(self, hls_steps):
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
            "POS_H": 2,
            "POS_W": 1,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel14_wpar4(self, hls_steps):
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
            "POS_H": 2,
            "POS_W": 2,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel15_wpar4(self, hls_steps):
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
            "POS_H": 2,
            "POS_W": 3,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel16_wpar4(self, hls_steps):
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
            "POS_H": 2,
            "POS_W": 4,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_3x3_pixel17_wpar4(self, hls_steps):
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
            "POS_H": 2,
            "POS_W": 5,
            "CH_PAR": 2,
            "W_PAR": 4,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)