import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_catapult_test import BaseCatapultTest
from .base_vitis_test import BaseVitisTest


def generate_relu_test_data(config_dict):
    in_unsigned = bool(config_dict.get("INPUT_IS_UNSIGNED", False))
    out_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False))
    in_bits = int(config_dict["INPUT_DATAWIDTH"])
    out_bits = int(config_dict["OUTPUT_DATAWIDTH"])

    onnx_in_type = BaseVitisTest.get_tensorproto_dtype(in_bits, in_unsigned)
    onnx_out_type = BaseVitisTest.get_tensorproto_dtype(out_bits, out_unsigned)
    np_in_type = BaseVitisTest.get_numpy_dtype(in_bits, in_unsigned)

    in_info = np.iinfo(np_in_type)
    input_tensor = np.random.randint(
        int(in_info.min),
        int(in_info.max) + 1,
        size=(1, config_dict["DIM2"], config_dict["DIM0"], config_dict["DIM1"]),
        dtype=np_in_type,
    )

    X = helper.make_tensor_value_info(
        "X", onnx_in_type, [1, config_dict["DIM2"], config_dict["DIM0"], config_dict["DIM1"]]
    )
    Y = helper.make_tensor_value_info(
        "Y", onnx_out_type, [1, config_dict["DIM2"], config_dict["DIM0"], config_dict["DIM1"]]
    )
    X_scale = helper.make_tensor("X_scale", TensorProto.FLOAT, [], [config_dict["X_SCALE"]])
    X_zp = helper.make_tensor("X_zp", onnx_in_type, [], [config_dict["X_ZP"]])
    Y_scale = helper.make_tensor("Y_scale", TensorProto.FLOAT, [], [config_dict["Y_SCALE"]])
    Y_zp = helper.make_tensor("Y_zp", onnx_out_type, [], [config_dict["Y_ZP"]])
    graph = helper.make_graph(
        [
            helper.make_node("DequantizeLinear", inputs=["X", "X_scale", "X_zp"], outputs=["X_dq"]),
            helper.make_node("Relu", inputs=["X_dq"], outputs=["Y_dq"]),
            helper.make_node("QuantizeLinear", inputs=["Y_dq", "Y_scale", "Y_zp"], outputs=["Y"]),
        ],
        "relu_test",
        [X],
        [Y],
        initializer=[X_scale, X_zp, Y_scale, Y_zp],
    )
    model = helper.make_model(graph, producer_name="qonnx")
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return input_tensor, session.run(None, {"X": input_tensor})[0]


class StreamingReLUTestData:
    @staticmethod
    def simple_po2_config():
        return {
            "INPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": 0,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": 0,
            "DIM0": 2,
            "DIM1": 2,
            "DIM2": 2,
            "DIM2_UNROLL": 2,
            "DIM1_UNROLL": 2,
            "X_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "X_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }


class TestStreamingReLU(StreamingReLUTestData, BaseVitisTest):

    @property
    def operator_filename(self):
        return "StreamingReLU"

    @property
    def unit_filename(self):
        return "StreamingReLU"

    def generate_config_file(self, config_dict):
        in_unsigned = bool(config_dict.get("INPUT_IS_UNSIGNED", False))
        out_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False))

        in_bits = int(config_dict["INPUT_DATAWIDTH"])
        out_bits = int(config_dict["OUTPUT_DATAWIDTH"])

        np_out_type = self.get_numpy_dtype(out_bits, out_unsigned)
        input_tensor, y = generate_relu_test_data(config_dict)

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
        typedef_suffix = "u" if in_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['INPUT_DATAWIDTH']}> TInput;")
        typedef_suffix = "u" if out_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['OUTPUT_DATAWIDTH']}> TOutput;")
        cwr.add_line(f"typedef DequantQuantPo2<0, TInput, TOutput> Quantizer;")
        cwr.add_lines(
            csnake.Variable(
                "input_tensor",
                primitive=f"TInput",
                value=input_tensor,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "output_tensor",
                primitive=f"TOutput",
                value=y,
            ).generate_initialization()
        )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_simple_po2(self, hls_steps):
        np.random.seed(42)
        self.run(self.simple_po2_config(), hls_steps)


class TestCatapultStreamingReLU(StreamingReLUTestData, BaseCatapultTest):
    @property
    def unit_filename(self):
        return "StreamingReLU"

    def generate_config_file(self, config_dict):
        input_tensor, output_tensor = generate_relu_test_data(config_dict)
        cwr = csnake.CodeWriter()
        cwr.include("<array>")
        cwr.include("<ac_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            if key in ["X_SCALE", "W_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {value}f;")
            else:
                cwr.add_line(f"const int {key} = {value};")
        cwr.add_line(f"using TInput = ac_int<{config_dict['INPUT_DATAWIDTH']}, true>;")
        cwr.add_line(f"using TOutput = ac_int<{config_dict['OUTPUT_DATAWIDTH']}, true>;")
        cwr.add_line("struct Quantizer {")
        cwr.indent()
        cwr.add_line("TOutput operator()(TInput value) const { return value; }")
        cwr.dedent()
        cwr.add_line("};")
        cwr.add_lines(csnake.Variable("input_tensor", primitive="TInput", value=input_tensor).generate_initialization())
        cwr.add_lines(csnake.Variable("output_tensor", primitive="TOutput", value=output_tensor).generate_initialization())
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_simple_po2(self, hls_steps, tmp_path):
        np.random.seed(42)
        self.run(self.simple_po2_config(), hls_steps, str(tmp_path))
