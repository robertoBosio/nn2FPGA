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

class TestStreamingConcat(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "StreamingConcat"

    @property
    def unit_filename(self) -> str:
        return "StreamingConcat"

    def generate_config_file(self, config_dict, class_name: str = "StreamingConcatHeight"):
        ina_unsigned = bool(config_dict.get("INPUTA_IS_UNSIGNED", False))
        inb_unsigned = bool(config_dict.get("INPUTB_IS_UNSIGNED", False))
        out_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False))

        ina_bits = int(config_dict["INPUTA_DATAWIDTH"])
        inb_bits = int(config_dict["INPUTB_DATAWIDTH"])
        out_bits = int(config_dict["OUTPUT_DATAWIDTH"])

        onnx_ina_type = self.get_tensorproto_dtype(ina_bits, ina_unsigned)
        onnx_inb_type = self.get_tensorproto_dtype(inb_bits, inb_unsigned)
        onnx_out_type = self.get_tensorproto_dtype(out_bits, out_unsigned)
        np_ina_type = self.get_numpy_dtype(ina_bits, ina_unsigned)
        np_inb_type = self.get_numpy_dtype(inb_bits, inb_unsigned)

        # random tensors
        in_info = np.iinfo(np_ina_type)
        input_tensor0 = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,
            size=(
                1,
                config_dict["IN_DIM2_A"],
                config_dict["IN_DIM0_A"],
                config_dict["IN_DIM1_A"],
            ),
            dtype=np_ina_type,
        )
        in_info = np.iinfo(np_inb_type)
        input_tensor1 = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,
            size=(
                1,
                config_dict["IN_DIM2_B"],
                config_dict["IN_DIM0_B"],
                config_dict["IN_DIM1_B"],
            ),
            dtype=np_inb_type,
        )

        output_shape = [
            1,
            (
                config_dict["IN_DIM2_A"] + config_dict["IN_DIM2_B"]
                if config_dict["CONCAT_AXIS"] == 1
                else config_dict["IN_DIM2_A"]
            ),
            (
                config_dict["IN_DIM0_A"] + config_dict["IN_DIM0_B"]
                if config_dict["CONCAT_AXIS"] == 2
                else config_dict["IN_DIM0_A"]
            ),
            (
                config_dict["IN_DIM1_A"] + config_dict["IN_DIM1_B"]
                if config_dict["CONCAT_AXIS"] == 3
                else config_dict["IN_DIM1_A"]
            ),
        ]

        # I/O
        X0 = helper.make_tensor_value_info(
            "X0",
            onnx_ina_type,
            [
                1,
                config_dict["IN_DIM2_A"],
                config_dict["IN_DIM0_A"],
                config_dict["IN_DIM1_A"],
            ],
        )
        X1 = helper.make_tensor_value_info(
            "X1",
            onnx_inb_type,
            [
                1,
                config_dict["IN_DIM2_B"],
                config_dict["IN_DIM0_B"],
                config_dict["IN_DIM1_B"],
            ],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_out_type,
            output_shape,
        )

        A_scale = helper.make_tensor(
            "A_scale", TensorProto.FLOAT, [], [config_dict["A_SCALE"]]
        )
        A_zp = helper.make_tensor("A_zp", onnx_ina_type, [], [config_dict["A_ZP"]])
        B_scale = helper.make_tensor(
            "B_scale", TensorProto.FLOAT, [], [config_dict["B_SCALE"]]
        )
        B_zp = helper.make_tensor("B_zp", onnx_inb_type, [], [config_dict["B_ZP"]])
        Y_scale = helper.make_tensor(
            "Y_scale", TensorProto.FLOAT, [], [config_dict["Y_SCALE"]]
        )
        Y_zp = helper.make_tensor("Y_zp", onnx_out_type, [], [config_dict["Y_ZP"]])

        dequant0 = helper.make_node(
            "DequantizeLinear",
            inputs=["X0", "A_scale", "A_zp"],
            outputs=["X0_dequant"],
        )
        dequant1 = helper.make_node(
            "DequantizeLinear",
            inputs=["X1", "B_scale", "B_zp"],
            outputs=["X1_dequant"],
        )
        quant = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dequant", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        concat = helper.make_node(
            "Concat",
            inputs=[
                "X0_dequant",
                "X1_dequant",
            ],
            outputs=["Y_dequant"],
            axis=config_dict["CONCAT_AXIS"],
        )

        graph = helper.make_graph(
            [dequant0, dequant1, concat, quant],
            "concat_test",
            [X0, X1],
            [Y],
            initializer=[A_scale, A_zp, B_scale, B_zp, Y_scale, Y_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        y = sess.run(None, {"X0": input_tensor0, "X1": input_tensor1})[0]
        y = y.astype(self.get_numpy_dtype(out_bits, out_unsigned), copy=False)
        config_dict["VALUE_TOLERANCE_LSB"] = 0 if (
            _is_power_of_two_ratio(config_dict["A_SCALE"], config_dict["Y_SCALE"])
            and _is_power_of_two_ratio(config_dict["B_SCALE"], config_dict["Y_SCALE"])
        ) else 1

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            if key in ["A_SCALE", "B_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {value}f;")
            else:
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                    cwr.add_line(f"const bool {key} = {value_str};")
                else:
                    cwr.add_line(f"const int {key} = {int(value)};")
        cwr.add_line(f"const size_t OUT_DIM0 = {output_shape[2]};")
        cwr.add_line(f"const size_t OUT_DIM1 = {output_shape[3]};")
        cwr.add_line(f"const size_t OUT_DIM2 = {output_shape[1]};")
        typedef_suffix = "u" if ina_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['INPUTA_DATAWIDTH']}> TInputA;")
        typedef_suffix = "u" if inb_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['INPUTB_DATAWIDTH']}> TInputB;")
        typedef_suffix = "u" if out_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['OUTPUT_DATAWIDTH']}> TOutput;")
        cwr.add_line(
            f"using TInputWordA = std::array<TInputA, {config_dict['DIM2_UNROLL']}>;"
        )
        cwr.add_line(
            f"using TInputWordB = std::array<TInputB, {config_dict['DIM2_UNROLL']}>;"
        )
        cwr.add_line(
            f"using TOutputWord = std::array<TOutput, {config_dict['DIM2_UNROLL']}>;"
        )
        emit_requantizer_typedef(
            cwr,
            "TransformA",
            "TInputA",
            "TOutput",
            config_dict["A_SCALE"],
            config_dict["Y_SCALE"],
            input_zero_point=config_dict["A_ZP"],
            output_zero_point=config_dict["Y_ZP"],
        )
        emit_requantizer_typedef(
            cwr,
            "TransformB",
            "TInputB",
            "TOutput",
            config_dict["B_SCALE"],
            config_dict["Y_SCALE"],
            input_zero_point=config_dict["B_ZP"],
            output_zero_point=config_dict["Y_ZP"],
        )
        if config_dict["CONCAT_AXIS"] == 1:
            cwr.add_line(
                f"using StreamingConcat = {class_name}<TInputWordA, TInputA, TInputWordB, TInputB, TOutputWord, TOutput, TransformA, TransformB, IN_DIM0_A, IN_DIM1_A, IN_DIM2_A, IN_DIM2_B, DIM1_UNROLL, DIM2_UNROLL>;"
            )
        elif config_dict["CONCAT_AXIS"] == 2:
            cwr.add_line(
                f"using StreamingConcat = {class_name}<TInputWordA, TInputA, TInputWordB, TInputB, TOutputWord, TOutput, TransformA, TransformB, IN_DIM0_A, IN_DIM0_B, IN_DIM1_A, IN_DIM2_A, DIM1_UNROLL, DIM2_UNROLL>;"
            )
        elif config_dict["CONCAT_AXIS"] == 3:
            cwr.add_line(
                f"using StreamingConcat = {class_name}<TInputWordA, TInputA, TInputWordB, TInputB, TOutputWord, TOutput, TransformA, TransformB, IN_DIM0_A, IN_DIM1_A, IN_DIM1_B, IN_DIM2_A, DIM1_UNROLL, DIM2_UNROLL>;"
            )
        cwr.add_lines(
            csnake.Variable(
                "input_tensor0",
                primitive="TInputA",
                value=input_tensor0,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "input_tensor1",
                primitive="TInputB",
                value=input_tensor1,
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

    def test_channelconcat_pertensor_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTA_DATAWIDTH": 8,
            "INPUTA_IS_UNSIGNED": False,
            "INPUTB_DATAWIDTH": 8,
            "INPUTB_IS_UNSIGNED": False,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_DIM0_A": 4,
            "IN_DIM0_B": 4,
            "IN_DIM1_A": 4,
            "IN_DIM1_B": 4,
            "IN_DIM2_A": 4,
            "IN_DIM2_B": 8,
            "DIM2_UNROLL": 2,
            "DIM1_UNROLL": 1,
            "CONCAT_AXIS": 1,
            "A_SCALE": 2**-5,
            "B_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict, hls_steps, workdir=".", class_name="StreamingConcatDim2"
        )

    def test_heightconcat_pertensor_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTA_DATAWIDTH": 8,
            "INPUTA_IS_UNSIGNED": False,
            "INPUTB_DATAWIDTH": 8,
            "INPUTB_IS_UNSIGNED": False,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_DIM0_A": 4,
            "IN_DIM0_B": 8,
            "IN_DIM1_A": 4,
            "IN_DIM1_B": 4,
            "IN_DIM2_A": 4,
            "IN_DIM2_B": 4,
            "DIM2_UNROLL": 2,
            "DIM1_UNROLL": 1,
            "CONCAT_AXIS": 2,
            "A_SCALE": 2**-5,
            "B_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict, hls_steps, workdir=".", class_name="StreamingConcatDim0"
        )

    def test_widthconcat_pertensor_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTA_DATAWIDTH": 8,
            "INPUTA_IS_UNSIGNED": False,
            "INPUTB_DATAWIDTH": 8,
            "INPUTB_IS_UNSIGNED": False,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_DIM0_A": 4,
            "IN_DIM0_B": 4,
            "IN_DIM1_A": 4,
            "IN_DIM1_B": 8,
            "IN_DIM2_A": 4,
            "IN_DIM2_B": 4,
            "DIM2_UNROLL": 2,
            "DIM1_UNROLL": 1,
            "CONCAT_AXIS": 3,
            "A_SCALE": 2**-5,
            "B_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict, hls_steps, workdir=".", class_name="StreamingConcatDim1"
        )

    def test_channelconcat_pertensor_float_asymmetric(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTA_DATAWIDTH": 8,
            "INPUTA_IS_UNSIGNED": True,
            "INPUTB_DATAWIDTH": 8,
            "INPUTB_IS_UNSIGNED": False,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": True,
            "IN_DIM0_A": 4,
            "IN_DIM0_B": 4,
            "IN_DIM1_A": 4,
            "IN_DIM1_B": 4,
            "IN_DIM2_A": 4,
            "IN_DIM2_B": 8,
            "DIM2_UNROLL": 2,
            "DIM1_UNROLL": 1,
            "CONCAT_AXIS": 1,
            "A_SCALE": 0.375,
            "B_SCALE": 0.2,
            "Y_SCALE": 0.3,
            "A_ZP": 91,
            "B_ZP": -17,
            "Y_ZP": 109,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict, hls_steps, workdir=".", class_name="StreamingConcatDim2"
        )
