import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper

from .base_hls_test import BaseHLSTest
from .utils.quantizer_codegen import emit_requantizer_typedef


def _is_power_of_two_ratio(num: float, den: float, atol=1e-12) -> bool:
    ratio = float(num) / float(den)
    log2_ratio = np.log2(ratio)
    return np.isclose(log2_ratio, np.round(log2_ratio), atol=atol)


class TestStreamingMaxPool(BaseHLSTest):

    @property
    def operator_filename(self):
        return "StreamingMaxPool"

    @property
    def unit_filename(self):
        return "StreamingMaxPool"

    def _output_dim(self, input_dim, pad_before, pad_after, kernel, dilation, stride):
        numerator = input_dim + pad_before + pad_after - dilation * (kernel - 1) - 1
        return 1 + (numerator // stride)

    def generate_config_file(self, config_dict):
        input_unsigned = bool(config_dict.get("INPUT_IS_UNSIGNED", False))
        output_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False))

        input_bits = int(config_dict["INPUT_DATAWIDTH"])
        output_bits = int(config_dict["OUTPUT_DATAWIDTH"])

        onnx_input_type = self.get_tensorproto_dtype(input_bits, input_unsigned)
        onnx_output_type = self.get_tensorproto_dtype(output_bits, output_unsigned)
        np_input_type = self.get_numpy_dtype(input_bits, input_unsigned)
        np_output_type = self.get_numpy_dtype(output_bits, output_unsigned)

        input_info = np.iinfo(np_input_type)
        input_tensor = np.random.randint(
            int(input_info.min),
            int(input_info.max) + 1,
            size=(
                1,
                config_dict["OUT_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ),
            dtype=np_input_type,
        )

        out_height = self._output_dim(
            config_dict["IN_HEIGHT"],
            config_dict["PAD_T"],
            config_dict["PAD_B"],
            config_dict["FH"],
            config_dict["DIL_H"],
            config_dict["STRIDE_H"],
        )
        out_width = self._output_dim(
            config_dict["IN_WIDTH"],
            config_dict["PAD_L"],
            config_dict["PAD_R"],
            config_dict["FW"],
            config_dict["DIL_W"],
            config_dict["STRIDE_W"],
        )

        X = helper.make_tensor_value_info(
            "X",
            onnx_input_type,
            [
                1,
                config_dict["OUT_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_output_type,
            [1, config_dict["OUT_CH"], out_height, out_width],
        )

        X_scale = helper.make_tensor(
            "X_scale", TensorProto.FLOAT, [], [float(config_dict["X_SCALE"])]
        )
        X_zp = helper.make_tensor(
            "X_zp", onnx_input_type, [], [int(config_dict["X_ZP"])]
        )
        Y_scale = helper.make_tensor(
            "Y_scale", TensorProto.FLOAT, [], [float(config_dict["Y_SCALE"])]
        )
        Y_zp = helper.make_tensor(
            "Y_zp", onnx_output_type, [], [int(config_dict["Y_ZP"])]
        )

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
            pads=[
                config_dict["PAD_T"],
                config_dict["PAD_L"],
                config_dict["PAD_B"],
                config_dict["PAD_R"],
            ],
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
            "qmaxpool_test",
            [X],
            [Y],
            initializer=[X_scale, X_zp, Y_scale, Y_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        y = sess.run(None, {"X": input_tensor})[0]
        y = y.astype(np_output_type, copy=False)
        config_dict["VALUE_TOLERANCE_LSB"] = 0 if _is_power_of_two_ratio(
            config_dict["X_SCALE"], config_dict["Y_SCALE"]
        ) else 1

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            if key in ["X_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {float(value)}f;")
            elif isinstance(value, bool):
                cwr.add_line(f"const bool {key} = {'true' if value else 'false'};")
            else:
                cwr.add_line(f"const int {key} = {int(value)};")

        input_typedef_suffix = "u" if input_unsigned else ""
        output_typedef_suffix = "u" if output_unsigned else ""
        cwr.add_line(f"typedef ap_{input_typedef_suffix}int<{input_bits}> TInput;")
        cwr.add_line(f"typedef ap_{output_typedef_suffix}int<{output_bits}> TOutput;")
        emit_requantizer_typedef(
            cwr,
            "Quantizer",
            "TInput",
            "TOutput",
            config_dict["X_SCALE"],
            config_dict["Y_SCALE"],
            input_zero_point=config_dict["X_ZP"],
            output_zero_point=config_dict["Y_ZP"],
        )
        cwr.add_lines(
            csnake.Variable(
                "input_tensor",
                primitive=f"ap_{input_typedef_suffix}int<{input_bits}>",
                value=input_tensor,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "output_tensor",
                primitive=f"ap_{output_typedef_suffix}int<{output_bits}>",
                value=y,
            ).generate_initialization()
        )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_equal_quant(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
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
            "Y_SCALE": 2**-5,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_asymmetric_po2(self, hls_steps):
        np.random.seed(43)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "IN_HEIGHT": 18,
            "IN_WIDTH": 20,
            "OUT_CH": 12,
            "CH_PAR": 4,
            "W_PAR": 2,
            "FH": 2,
            "FW": 2,
            "STRIDE_H": 2,
            "STRIDE_W": 2,
            "DIL_H": 1,
            "DIL_W": 1,
            "PAD_T": 0,
            "PAD_B": 0,
            "PAD_L": 0,
            "PAD_R": 0,
            "CEIL_MODE": 0,
            "X_SCALE": 2**-4,
            "X_ZP": -7,
            "Y_SCALE": 2**-5,
            "Y_ZP": 11,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_asymmetric_float(self, hls_steps):
        np.random.seed(44)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "IN_HEIGHT": 17,
            "IN_WIDTH": 19,
            "OUT_CH": 16,
            "CH_PAR": 4,
            "W_PAR": 1,
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
            "X_SCALE": 0.3,
            "X_ZP": -12,
            "Y_SCALE": 0.8,
            "Y_ZP": 9,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
