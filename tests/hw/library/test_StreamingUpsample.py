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


class TestStreamingUpsample(BaseHLSTest):

    @property
    def operator_filename(self):
        return "StreamingUpsample"

    @property
    def unit_filename(self):
        return "StreamingUpsample"

    def generate_config_file(self, config_dict):
        in_unsigned = bool(config_dict.get("INPUT_IS_UNSIGNED", False))
        out_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False))
        in_bits = int(config_dict["INPUT_DATAWIDTH"])
        out_bits = int(config_dict["OUTPUT_DATAWIDTH"])
        onnx_in_type = self.get_tensorproto_dtype(in_bits, in_unsigned)
        onnx_out_type = self.get_tensorproto_dtype(out_bits, out_unsigned)
        np_in_type = self.get_numpy_dtype(in_bits, in_unsigned)
        np_out_type = self.get_numpy_dtype(out_bits, out_unsigned)

        in_info = np.iinfo(np_in_type)
        input_tensor = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,
            size=(
                1,
                config_dict["IN_DIM2"],
                config_dict["IN_DIM0"],
                config_dict["IN_DIM1"],
            ),
            dtype=np_in_type,
        )

        X = helper.make_tensor_value_info(
            "X",
            onnx_in_type,
            [
                1,
                config_dict["IN_DIM2"],
                config_dict["IN_DIM0"],
                config_dict["IN_DIM1"],
            ],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_out_type,
            [
                1,
                config_dict["OUT_DIM2"],
                config_dict["OUT_DIM0"],
                config_dict["OUT_DIM1"],
            ],
        )

        X_scale = helper.make_tensor(
            "X_scale", TensorProto.FLOAT, [], [float(config_dict["X_SCALE"])]
        )
        X_zp = helper.make_tensor("X_zp", onnx_in_type, [], [int(config_dict["X_ZP"])])
        Y_scale = helper.make_tensor(
            "Y_scale", TensorProto.FLOAT, [], [float(config_dict["Y_SCALE"])]
        )
        Y_zp = helper.make_tensor("Y_zp", onnx_out_type, [], [int(config_dict["Y_ZP"])])

        dqlinear = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "X_scale", "X_zp"],
            outputs=["X_dq"],
        )

        upsample_scales = helper.make_tensor(
            "scales",
            TensorProto.FLOAT,
            [4],
            [1.0, 1.0, config_dict["SCALE_FACTOR"], config_dict["SCALE_FACTOR"]],
        )

        upsample_roi = helper.make_tensor("roi", TensorProto.FLOAT, [0], [])

        upsample = helper.make_node(
            "Resize",
            inputs=["X_dq", "roi", "scales"],
            outputs=["Y_dq"],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            cubic_coeff_a=-0.75,
            nearest_mode="floor",
        )

        qlinear = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dq", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [dqlinear, upsample, qlinear],
            "qupsample_test",
            [X],
            [Y],
            initializer=[X_scale, X_zp, Y_scale, Y_zp, upsample_roi, upsample_scales],
        )
        model = helper.make_model(graph, producer_name="qonnx")
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        y = sess.run(None, {"X": input_tensor})[0]
        y = y.astype(np_out_type, copy=False)
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
            else:
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                    cwr.add_line(f"const bool {key} = {value_str};")
                else:
                    cwr.add_line(f"const int {key} = {int(value)};")
        typedef_suffix = "u" if in_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{in_bits}> TInput;")

        typedef_suffix = "u" if out_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{out_bits}> TOutput;")
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
                primitive="TInput",
                value=input_tensor,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "output_tensor",
                primitive="TOutput",
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
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_DIM0": 20,
            "IN_DIM1": 20,
            "IN_DIM2": 20,
            "OUT_DIM0": 40,
            "OUT_DIM1": 40,
            "OUT_DIM2": 20,
            "DIM2_UNROLL": 5,
            "IN_DIM1_UNROLL": 5,
            "OUT_DIM1_UNROLL": 10,
            "X_SCALE": 2**-5,
            "X_ZP": 0,
            "Y_SCALE": 2**-5,
            "Y_ZP": 0,
            "SCALE_FACTOR": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_asymmetric_po2(self, hls_steps):
        np.random.seed(43)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_DIM0": 16,
            "IN_DIM1": 20,
            "IN_DIM2": 12,
            "OUT_DIM0": 32,
            "OUT_DIM1": 40,
            "OUT_DIM2": 12,
            "DIM2_UNROLL": 4,
            "IN_DIM1_UNROLL": 5,
            "OUT_DIM1_UNROLL": 10,
            "X_SCALE": 2**-4,
            "X_ZP": -7,
            "Y_SCALE": 2**-5,
            "Y_ZP": 11,
            "SCALE_FACTOR": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_asymmetric_float(self, hls_steps):
        np.random.seed(44)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_DIM0": 18,
            "IN_DIM1": 18,
            "IN_DIM2": 16,
            "OUT_DIM0": 36,
            "OUT_DIM1": 36,
            "OUT_DIM2": 16,
            "DIM2_UNROLL": 4,
            "IN_DIM1_UNROLL": 3,
            "OUT_DIM1_UNROLL": 6,
            "X_SCALE": 0.3,
            "X_ZP": -12,
            "Y_SCALE": 0.8,
            "Y_ZP": 9,
            "SCALE_FACTOR": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
