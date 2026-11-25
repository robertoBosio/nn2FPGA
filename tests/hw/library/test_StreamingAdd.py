from fractions import Fraction

import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest
from .utils.quantizer_codegen import emit_linear_quantizer_typedef


def _signed_range_bits(bound: int) -> int:
    bound = max(1, int(bound))
    return int(np.ceil(np.log2(bound + 1))) + 1


def _quant_range(bitwidth: int, unsigned: bool) -> tuple[int, int]:
    if unsigned:
        return 0, (1 << bitwidth) - 1
    return -(1 << (bitwidth - 1)), (1 << (bitwidth - 1)) - 1


def _max_centered_magnitude(bitwidth: int, unsigned: bool, zeropt: int) -> int:
    qmin, qmax = _quant_range(bitwidth, unsigned)
    return max(abs(qmin - zeropt), abs(qmax - zeropt))


def _aligned_bound(scale: float, acc_scale: float, magnitude: int) -> int:
    ratio = float(scale) / float(acc_scale)
    return int(np.ceil(abs(ratio) * magnitude + 0.5))


def _is_power_of_two_ratio(num: float, den: float, atol=1e-12) -> bool:
    ratio = float(num) / float(den)
    log2_ratio = np.log2(ratio)
    return np.isclose(log2_ratio, np.round(log2_ratio), atol=atol)


def _emit_input_aligner_typedef(cwr, alias, in_type, acc_type, in_scale, acc_scale, zero_point, max_den=1 << 20, atol=1e-12):
    ratio = float(in_scale) / float(acc_scale)
    if np.isclose(ratio, 1.0, atol=atol):
        if int(zero_point) == 0:
            cwr.add_line(f"typedef DequantQuantEqual<{in_type}> {alias};")
        else:
            cwr.add_line(
                f"typedef DequantAlignEqual<{int(zero_point)}, {in_type}, {acc_type}> {alias};"
            )
        return

    log2_ratio = np.log2(ratio)
    rounded = np.round(log2_ratio)
    if np.isclose(log2_ratio, rounded, atol=atol):
        shift = int(-rounded)
        cwr.add_line(f"const int {alias}_SHIFT = {shift};")
        if int(zero_point) == 0:
            cwr.add_line(
                f"typedef DequantQuantPo2<{alias}_SHIFT, {in_type}, {acc_type}> {alias};"
            )
        else:
            cwr.add_line(
                f"typedef DequantAlignPo2<{alias}_SHIFT, {int(zero_point)}, {in_type}, {acc_type}> {alias};"
            )
        return

    frac = Fraction(ratio).limit_denominator(max_den)
    cwr.add_line(f"const int {alias}_NUM = {int(frac.numerator)};")
    cwr.add_line(f"const int {alias}_DEN = {int(frac.denominator)};")
    if int(zero_point) == 0:
        cwr.add_line(
            f"typedef DequantQuantFloat<{alias}_NUM, {alias}_DEN, {in_type}, {acc_type}> {alias};"
        )
    else:
        cwr.add_line(
            f"typedef DequantAlignFloat<{alias}_NUM, {alias}_DEN, {int(zero_point)}, {in_type}, {acc_type}> {alias};"
        )

class TestStreamingAdd(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "StreamingAdd"

    @property
    def unit_filename(self) -> str:
        return "StreamingAdd"

    def generate_config_file(self, config_dict):
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
        np_out_type = self.get_numpy_dtype(out_bits, out_unsigned)

        acc_scale = float(config_dict.get("ACC_SCALE", min(config_dict["A_SCALE"], config_dict["B_SCALE"])))

        # Random input tensor in correct integer domain/range
        in_info = np.iinfo(np_ina_type)
        inputa_tensor = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,  # randint upper bound is exclusive
            size=(
                1,
                config_dict["DIM2"],
                config_dict["DIM0"],
                config_dict["DIM1"],
            ),
            dtype=np_ina_type,
        )

        in_info = np.iinfo(np_inb_type)
        inputb_tensor = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,  # randint upper bound is exclusive
            size=(
                1,
                config_dict["DIM2"],
                config_dict["DIM0"],
                config_dict["DIM1"],
            ),
            dtype=np_inb_type,
        )

        # I/O
        A = helper.make_tensor_value_info(
            "A",
            onnx_ina_type,
            [
                1,
                config_dict["DIM2"],
                config_dict["DIM0"],
                config_dict["DIM1"],
            ],
        )
        B = helper.make_tensor_value_info(
            "B",
            onnx_inb_type,
            [
                1,
                config_dict["DIM2"],
                config_dict["DIM0"],
                config_dict["DIM1"],
            ],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_out_type,
            [
                1,
                config_dict["DIM2"],
                config_dict["DIM0"],
                config_dict["DIM1"],
            ],
        )

        A_scale = helper.make_tensor(
            "A_scale", TensorProto.FLOAT, [], [config_dict["A_SCALE"]]
        )
        B_scale = helper.make_tensor(
            "B_scale", TensorProto.FLOAT, [], [config_dict["B_SCALE"]]
        )
        A_zp = helper.make_tensor(
            "A_zp",
            onnx_ina_type,
            [],
            [config_dict["A_ZP"]],
        )
        B_zp = helper.make_tensor(
            "B_zp",
            onnx_inb_type,  
            [],
            [config_dict["B_ZP"]],
        )
        Y_scale = helper.make_tensor(
            "Y_scale", TensorProto.FLOAT, [], [config_dict["Y_SCALE"]]
        )
        Y_zp = helper.make_tensor(
            "Y_zp",
            onnx_out_type,
            [],
            [config_dict["Y_ZP"]],
        )

        dequant0 = helper.make_node(
            "DequantizeLinear",
            inputs=["A", "A_scale", "A_zp"],
            outputs=["A_dequant"],
        )
        dequant1 = helper.make_node(
            "DequantizeLinear",
            inputs=["B", "B_scale", "B_zp"],
            outputs=["B_dequant"],
        )
        quant = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dequant", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        add = helper.make_node(
            "Add",
            inputs=[
                "A_dequant",
                "B_dequant",
            ],
            outputs=["Y_dequant"],
        )

        graph = helper.make_graph(
            [dequant0, dequant1, add, quant],
            "add_test",
            [A, B],
            [Y],
            initializer=[A_scale, B_scale, A_zp, B_zp, Y_scale, Y_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(
            model.SerializeToString(), session_options, providers=["CPUExecutionProvider"]
        )
        y = sess.run(None, {"A": inputa_tensor, "B": inputb_tensor})[0]
        y = y.astype(np_out_type, copy=False)

        max_a = _max_centered_magnitude(ina_bits, ina_unsigned, int(config_dict["A_ZP"]))
        max_b = _max_centered_magnitude(inb_bits, inb_unsigned, int(config_dict["B_ZP"]))
        bound_a = _aligned_bound(config_dict["A_SCALE"], acc_scale, max_a)
        bound_b = _aligned_bound(config_dict["B_SCALE"], acc_scale, max_b)
        tacc_bits = _signed_range_bits(bound_a + bound_b + 1)
        config_dict["ACC_SCALE"] = acc_scale
        config_dict["ACC_DATAWIDTH"] = tacc_bits
        config_dict["VALUE_TOLERANCE_LSB"] = 0 if (
            _is_power_of_two_ratio(config_dict["A_SCALE"], acc_scale)
            and _is_power_of_two_ratio(config_dict["B_SCALE"], acc_scale)
            and _is_power_of_two_ratio(acc_scale, config_dict["Y_SCALE"])
        ) else 1

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            if key in ["A_SCALE", "B_SCALE", "Y_SCALE", "ACC_SCALE"]:
                cwr.add_line(f"const float {key} = {value}f;")
            else:
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                    cwr.add_line(f"const bool {key} = {value_str};")
                else:
                    cwr.add_line(f"const int {key} = {value};")
        typedef_suffix = "u" if ina_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{ina_bits}> TInputA;")
        typedef_suffix = "u" if inb_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{inb_bits}> TInputB;")
        cwr.add_line(f"typedef ap_int<{tacc_bits}> TAcc;")
        typedef_suffix = "u" if out_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{out_bits}> TOutput;")

        _emit_input_aligner_typedef(
            cwr,
            "AlignA",
            "TInputA",
            "TAcc",
            config_dict["A_SCALE"],
            acc_scale,
            config_dict["A_ZP"],
        )
        _emit_input_aligner_typedef(
            cwr,
            "AlignB",
            "TInputB",
            "TAcc",
            config_dict["B_SCALE"],
            acc_scale,
            config_dict["B_ZP"],
        )
        emit_linear_quantizer_typedef(
            cwr,
            "OutputTransform",
            "TAcc",
            "TOutput",
            acc_scale,
            config_dict["Y_SCALE"],
            False,
            zero_point=config_dict["Y_ZP"],
        )

        cwr.add_lines(
            csnake.Variable(
                "input_tensor0",
                primitive=f"TInputA",
                value=inputa_tensor,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "input_tensor1",
                primitive=f"TInputB",
                value=inputb_tensor,
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

    def test_pertensor_po2_signed(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTA_IS_UNSIGNED": False,
            "INPUTB_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "INPUTA_DATAWIDTH": 8,
            "INPUTB_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "DIM0": 4,
            "DIM1": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 1,
            "A_SCALE": 2**-5,
            "B_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_pertensor_po2_unsigned(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTA_IS_UNSIGNED": True,
            "INPUTB_IS_UNSIGNED": True,
            "OUTPUT_IS_UNSIGNED": True,
            "INPUTA_DATAWIDTH": 8,
            "INPUTB_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "DIM0": 4,
            "DIM1": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 1,
            "A_SCALE": 2**-5,
            "B_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_pertensor_po2_hybrid(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTA_IS_UNSIGNED": False,
            "INPUTB_IS_UNSIGNED": True,
            "OUTPUT_IS_UNSIGNED": True,
            "INPUTA_DATAWIDTH": 8,
            "INPUTB_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "DIM0": 4,
            "DIM1": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 1,
            "A_SCALE": 2**-5,
            "B_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_pertensor_po2_mixed_scales(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTA_IS_UNSIGNED": False,
            "INPUTB_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "INPUTA_DATAWIDTH": 8,
            "INPUTB_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "DIM0": 4,
            "DIM1": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 1,
            "A_SCALE": 2**-4,
            "B_SCALE": 2**-6,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_pertensor_po2_mixed_scales_asymmetric(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTA_IS_UNSIGNED": True,
            "INPUTB_IS_UNSIGNED": True,
            "OUTPUT_IS_UNSIGNED": True,
            "INPUTA_DATAWIDTH": 8,
            "INPUTB_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "DIM0": 4,
            "DIM1": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 1,
            "A_SCALE": 2**-4,
            "B_SCALE": 2**-6,
            "Y_SCALE": 2**-5,
            "A_ZP": 23,
            "B_ZP": 101,
            "Y_ZP": 77,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_pertensor_float_mixed_scales_symmetric(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTA_IS_UNSIGNED": False,
            "INPUTB_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "INPUTA_DATAWIDTH": 8,
            "INPUTB_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "DIM0": 4,
            "DIM1": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 1,
            "A_SCALE": 0.375,
            "B_SCALE": 0.2,
            "Y_SCALE": 0.125,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_pertensor_float_mixed_scales_asymmetric(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTA_IS_UNSIGNED": True,
            "INPUTB_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": True,
            "INPUTA_DATAWIDTH": 8,
            "INPUTB_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "DIM0": 4,
            "DIM1": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 1,
            "A_SCALE": 0.375,
            "B_SCALE": 0.2,
            "Y_SCALE": 0.3,
            "A_ZP": 91,
            "B_ZP": -17,
            "Y_ZP": 109,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps, clean = False)
