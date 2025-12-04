import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest

class TestStreamingReLU(BaseHLSTest):

    @property
    def operator_filename(self):
        return "StreamingReLU"

    @property
    def unit_filename(self):
        return "StreamingReLU"

    def generate_config_file(self, config_dict):

        # random tensors
        input_tensor = np.random.randint(
            -128,
            127,
            size=(
                1,
                config_dict["IN_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ),
            dtype=np.int8,
        )

        X = helper.make_tensor_value_info(
            "X",
            TensorProto.INT8,
            [
                1,
                config_dict["IN_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            TensorProto.INT8,
            [
                1,
                config_dict["IN_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ],
        )

        X_scale = helper.make_tensor(
            "X_scale", TensorProto.FLOAT, [], [config_dict["X_SCALE"]]
        )
        X_zp = helper.make_tensor("X_zp", TensorProto.INT8, [], [config_dict["X_ZP"]])
        Y_scale = helper.make_tensor(
            "Y_scale", TensorProto.FLOAT, [], [config_dict["Y_SCALE"]]
        )
        Y_zp = helper.make_tensor("Y_zp", TensorProto.INT8, [], [config_dict["Y_ZP"]])

        dqlinear = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "X_scale", "X_zp"],
            outputs=["X_dq"],
        )

        global_avg_pool = helper.make_node(
            "Relu",
            inputs=["X_dq"],
            outputs=["Y_dq"],
        )

        qlinear = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dq", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [dqlinear, global_avg_pool, qlinear],
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
        cwr.add_line(f"typedef ap_uint<{config_dict['OUTPUT_DATAWIDTH']}> TOutput;")
        cwr.add_line(f"typedef DequantQuantPo2<0, TInput, TOutput> Quantizer;")
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
                primitive=f"ap_int<{config_dict['OUTPUT_DATAWIDTH']}>",
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
            "OUTPUT_DATAWIDTH": 8,
            "IN_HEIGHT": 2,
            "IN_WIDTH": 2,
            "IN_CH": 2,
            "CH_PAR": 2,
            "W_PAR": 2,
            "X_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "X_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
