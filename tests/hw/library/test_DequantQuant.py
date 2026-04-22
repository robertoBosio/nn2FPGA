import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest
from fractions import Fraction

class TestDequantQuant(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "DequantQuant"
    
    @property
    def unit_filename(self) -> str:
        return "DequantQuant"

    def _classify_quantizer(self, acc_scale, out_scale, same_type, max_den=1 << 20, atol=1e-12):
        ratio = float(acc_scale) / float(out_scale)

        if np.isclose(ratio, 1.0, atol=atol) and same_type:
            return "equal", None

        log2_ratio = np.log2(ratio)
        rounded = np.round(log2_ratio)
        if np.isclose(log2_ratio, rounded, atol=atol):
            shift = int(-rounded)
            return "po2", shift

        frac = Fraction(ratio).limit_denominator(max_den)
        return "float", (int(frac.numerator), int(frac.denominator))

    def generate_config_file(self, config_dict):
        acc_unsigned = not bool(config_dict.get("ACC_SIGNED", True))
        out_unsigned = not bool(config_dict.get("OUT_SIGNED", True))

        acc_bits = int(config_dict["ACC_DATAWIDTH"])
        out_bits = int(config_dict["OUT_DATAWIDTH"])

        onnx_in_type = self.get_tensorproto_dtype(acc_bits, acc_unsigned)
        onnx_out_type = self.get_tensorproto_dtype(out_bits, out_unsigned)
        np_in_type = self.get_numpy_dtype(acc_bits, acc_unsigned)
        np_out_type = self.get_numpy_dtype(out_bits, out_unsigned)

        input_tensor = np.array([int(config_dict["INPUT"])], dtype=np_in_type)

        X = helper.make_tensor_value_info("X", onnx_in_type, [1])
        Y = helper.make_tensor_value_info("Y", onnx_out_type, [1])

        acc_scale = helper.make_tensor(
            "ACC_SCALE", TensorProto.FLOAT, [], [float(config_dict["ACC_SCALE"])]
        )
        out_scale = helper.make_tensor(
            "OUT_SCALE", TensorProto.FLOAT, [], [float(config_dict["OUT_SCALE"])]
        )
        acc_zp = helper.make_tensor(
            "ACC_ZP", onnx_in_type, [], [int(config_dict.get("ACC_ZP", 0))]
        )
        out_zp = helper.make_tensor(
            "OUT_ZP", onnx_out_type, [], [int(config_dict.get("OUT_ZP", 0))]
        )

        dqlinear = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "ACC_SCALE", "ACC_ZP"],
            outputs=["X_dq"],
        )
        qlinear = helper.make_node(
            "QuantizeLinear",
            inputs=["X_dq", "OUT_SCALE", "OUT_ZP"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [dqlinear, qlinear],
            "dequantquant_test",
            [X],
            [Y],
            initializer=[acc_scale, acc_zp, out_scale, out_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")

        sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        y = sess.run(None, {"X": input_tensor})[0]
        y = y.astype(np_out_type, copy=False)

        same_type = (acc_bits == out_bits) and (acc_unsigned == out_unsigned)
        mode, params = self._classify_quantizer(
            config_dict["ACC_SCALE"],
            config_dict["OUT_SCALE"],
            same_type,
        )

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<iostream>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()

        for key, value in config_dict.items():
            if key in ["ACC_SCALE", "OUT_SCALE"]:
                cwr.add_line(f"const float {key} = {float(value)}f;")
            else:
                if isinstance(value, bool):
                    cwr.add_line(f"const bool {key} = {'true' if value else 'false'};")
                else:
                    cwr.add_line(f"const int {key} = {int(value)};")

        typedef_suffix = "u" if acc_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{acc_bits}> TAcc;")

        typedef_suffix = "u" if out_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{out_bits}> TOut;")

        if mode == "equal":
            cwr.add_line("typedef DequantQuantEqual<TOut> Quantizer;")
        elif mode == "po2":
            cwr.add_line(f"const int SHIFT = {int(params)};")
            cwr.add_line("typedef DequantQuantPo2<SHIFT, TAcc, TOut> Quantizer;")
        else:
            num, den = params
            cwr.add_line(f"const int NUM = {num};")
            cwr.add_line(f"const int DEN = {den};")
            cwr.add_line("typedef DequantQuantFloat<NUM, DEN, TAcc, TOut> Quantizer;")

        cwr.add_lines(
            csnake.Variable(
                "input_tensor",
                primitive="TAcc",
                value=input_tensor,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "output_tensor",
                primitive="TOut",
                value=y,
            ).generate_initialization()
        )

        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_basic_shift(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "INPUT": -63,
            "ACC_SCALE": 0.25,
            "OUT_SCALE": 1.0,
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
            "INPUT": 64,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 1.0,
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
            "INPUT": 16,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 0.25,
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
            "INPUT": 5,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 0.5,
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
            "INPUT": 7,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 0.5,
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
            "INPUT": -5,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 0.5,
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
            "INPUT": -7,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 0.5,
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
            "INPUT": 300,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 1.0,
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
            "INPUT": -300,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 1.0,
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
            "INPUT": 300,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 0,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 1.0,
        }
        self.run(
            config_dict,
            hls_steps,
        )
    
    def test_clamp_unsigned_negative_input(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "INPUT": -300,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 0,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 1.0,
        }
        self.run(
            config_dict,
            hls_steps,
        )
    
    def test_clip_signed_negative_shift(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 8,
            "OUT_DATAWIDTH": 8,
            "INPUT": -100,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 1,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 0.25,
        }
        self.run(
            config_dict,
            hls_steps,
        )

    def test_clip_signed_negative_shift_min_limit_bug_extreme(self, hls_steps):
        # Shift left by 7 -> should saturate to -128 for int8
        # If LimitsImpl<TOut>::min() is wrong (+128), the code will NOT clip to -128.
        config_dict = {
            "ACC_DATAWIDTH": 8,
            "OUT_DATAWIDTH": 8,
            "INPUT": -1,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 1,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 0.0078125,  # 1/128
        }
        self.run(
            config_dict,
            hls_steps,
        )

    def test_clip_signed_negative_shift_min_limit_bug_near_boundary(self, hls_steps):
        # -2 << 6 = -128 exactly. This should produce -128 (no wrap/clip ambiguity).
        # With an incorrect positive min limit (+128), comparisons against "min" are broken,
        # and implementations sometimes produce 0 or +128 depending on downstream casts.
        config_dict = {
            "ACC_DATAWIDTH": 8,
            "OUT_DATAWIDTH": 8,
            "INPUT": -2,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 1,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 0.015625,  # 1/64
        }
        self.run(
            config_dict,
            hls_steps,
        )

    def test_unsigned_to_signed_no_shift(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 16,
            "OUT_DATAWIDTH": 8,
            "INPUT": 3,
            "ACC_SIGNED": 0,
            "OUT_SIGNED": 1,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 1.0,
        }
        self.run(
            config_dict,
            hls_steps,
        )
    
    def test_mixed_1(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "INPUT": 4194304,
            "ACC_SIGNED": 0,
            "OUT_SIGNED": 1,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 65536.0,  # 2^16
        }
        self.run(
            config_dict,
            hls_steps,
        )
    
    def test_mixed_2(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "INPUT": 4194304,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 0,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 65536.0,  # 2^16
        }
        self.run(
            config_dict,
            hls_steps,
        )
    
    def test_simple_float_ratio(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "INPUT": 10,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 1,
            "ACC_SCALE": 1.0,
            "OUT_SCALE": 0.1,
        }
        self.run(
            config_dict,
            hls_steps,
        )
    
    def test_nontrivial_float_ratio(self, hls_steps):
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "OUT_DATAWIDTH": 8,
            "INPUT": 10,
            "ACC_SIGNED": 1,
            "OUT_SIGNED": 1,
            "ACC_SCALE": 0.8,
            "OUT_SCALE": 0.3,
        }
        self.run(
            config_dict,
            hls_steps,
        )