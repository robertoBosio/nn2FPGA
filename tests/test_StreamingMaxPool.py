import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest

class TestStreamingMaxPool(BaseHLSTest):

    @property
    def operator_filename(self):
        return "StreamingMaxPool"

    @property
    def unit_filename(self):
        return "StreamingMaxPool"

    def generate_config_file(self, config_dict):
        
        # random tensors
        input_tensor = np.random.randint(
            -128,
            127,
            size=(1, config_dict["OUT_CH"], config_dict["IN_HEIGHT"], config_dict["IN_WIDTH"]),
            dtype=np.int8,
        )

        X = helper.make_tensor_value_info(
            "X", TensorProto.INT8, [1, config_dict["OUT_CH"], config_dict["IN_HEIGHT"], config_dict["IN_WIDTH"]]
        )
        Y = helper.make_tensor_value_info(
            "Y",
            TensorProto.INT8,
            [
                1,
                config_dict["OUT_CH"],
                1,
                1,
            ],
        )

        X_scale = helper.make_tensor(
            "X_scale", TensorProto.FLOAT, [], [config_dict["X_SCALE"]]
        )
        X_zp = helper.make_tensor("X_zp", TensorProto.INT8, [], [config_dict["X_ZP"]])
        Y_scale = helper.make_tensor(
            "Y_scale", TensorProto.FLOAT, [], [config_dict["X_SCALE"]]
        )
        Y_zp = helper.make_tensor("Y_zp", TensorProto.INT8, [], [config_dict["X_ZP"]])

        dqlinear = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "X_scale", "X_zp"],
            outputs=["X_dq"],
        )

        max_pool = helper.make_node(
            "MaxPool",
            inputs=["X_dq"],
            outputs=["Y_dq"],
            kernel_shape=[config_dict["FH"], config_dict["FW"]],
            strides=[config_dict["STRIDE_H"], config_dict["STRIDE_W"]],
            pads=[config_dict["PAD_T"], config_dict["PAD_L"], config_dict["PAD_B"], config_dict["PAD_R"]],
            dilations=[config_dict["DIL_H"], config_dict["DIL_W"]],
            ceil_mode=config_dict["CEIL_MODE"],
        )

        qlinear = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dq", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [dqlinear, max_pool, qlinear],
            "qconv_test",
            [X],
            [Y],
            initializer=[X_scale, X_zp, Y_scale, Y_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        y = sess.run(None, {"X": input_tensor})[0]

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            if key in ["X_SCALE", "W_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {value}f;")
            else:
                cwr.add_line(f"const int {key} = {value};")
        cwr.add_line(f"typedef ap_int<{config_dict['INPUT_DATAWIDTH']}> TInput;")
        cwr.add_line(f"typedef DequantQuantPo2<0, TInput, TInput> Quantizer;")
        cwr.add_lines(
            csnake.Variable(
                "input_tensor",
                primitive=f"ap_int<{config_dict['INPUT_DATAWIDTH']}>",
                value=input_tensor,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "output_tensor",
                primitive=f"ap_int<{config_dict['INPUT_DATAWIDTH']}>",
                value=y,
            ).generate_initialization()
        )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code
    
    def test_7x7_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "IN_HEIGHT": 20,
            "IN_WIDTH": 20,
            "OUT_CH": 20,
            "CH_PAR": 5,
            "W_PAR": 5,
            "FH": 3,
            "FW": 3,
            "STRIDE_H": 2,
            "STRIDE_W": 2,
            "DIL_H": 1,
            "DIL_W": 1,
            "PAD_T": 1,
            "PAD_B": 1,
            "PAD_L": 1,
            "PAD_R": 1,
            "CEIL_MODE": 0,
            "X_SCALE": 2**-5,
            "X_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)