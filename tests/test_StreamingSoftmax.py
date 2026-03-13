import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest

class TestStreamingSoftmax(BaseHLSTest):

    @property
    def operator_filename(self):
        return "StreamingSoftmax"

    @property
    def unit_filename(self):
        return "StreamingSoftmax"

    def generate_lut_memory(self, config_dict):
        """
        Generate LUT contents for a softmax-style exponential using NumPy only:

            LUT[d] = Quantize_RNE_TIES_TO_EVEN( exp(-d * X_SCALE) )   in Q0.F

        where:
        - d is an unsigned integer index in [0 .. 2^INPUT_DATAWIDTH - 1]
        - X_SCALE is the real step per integer diff (e.g. 0.125)
        - F = EXP_PRECISION is the number of fractional bits (e.g. 12, 16, 24)

        This matches the common softmax kernel usage:
            diff = max - x    (>= 0)
            E    = LUT[diff]  ~= exp(x - max)

        Expected config_dict keys:
        INPUT_DATAWIDTH (int)
        EXP_PRECISION   (int)   # fractional bits F
        X_SCALE         (float)

        Optional keys:
        OUTPUT_IS_UNSIGNED (bool, default True)  # LUT is normally unsigned
        SATURATE           (bool, default True)  # saturate to the chosen bitwidth
        OUTPUT_TOTAL_BITS  (int, optional)       # if omitted, uses EXP_PRECISION+1
                                                # (enough to represent values up to ~1.0)

        Notes:
        - Output is an *integer table* representing Q0.F fixed point.
        - We use round-to-nearest, ties-to-even (banker's rounding), like ONNX QuantizeLinear.
        - For exp(0)=1.0: ideal value is 2^F, but if OUTPUT_TOTAL_BITS == F
            you can't represent it. Default OUTPUT_TOTAL_BITS is F+1 so 2^F fits.
        """
        # ---- read config ----
        nbits = int(config_dict["INPUT_DATAWIDTH"])
        F = int(config_dict["EXP_PRECISION"])
        lut_entries = 1 << nbits

        x_scale = float(config_dict["X_SCALE"])

        # Total output bits for the LUT integer values.
        # Default to F+1 so exp(0)=2^F is representable.
        out_total_bits = int(config_dict.get("OUTPUT_TOTAL_BITS", F))

        if out_total_bits <= 0:
            raise ValueError("OUTPUT_TOTAL_BITS must be positive")
        if F < 0:
            raise ValueError("EXP_PRECISION must be >= 0")

        # ---- build d = 0..2^nbits-1 ----
        d = np.arange(lut_entries, dtype=np.float64)

        # ---- compute real exp(-d * x_scale) ----
        y_real = np.exp(-d * x_scale)

        # ---- quantize to Q0.F with RNE ties-to-even ----
        # y_q = round(y_real * 2^F) with banker's rounding
        scale = float(2 ** F)
        y_q = np.rint(y_real * scale).astype(np.int64)  # np.rint = ties-to-even
        qmin, qmax = 0, (1 << out_total_bits) - 1
        y_q = np.clip(y_q, qmin, qmax)

        # ---- emit C++ variable ----
        # Choose a reasonable C++ primitive based on out_total_bits + signedness
        # (you can override outside if you prefer a fixed type)
        primitive = f"ap_uint<{out_total_bits}>"

        lut_values = [int(v) for v in y_q.tolist()]

        lut_variable = csnake.Variable(
            name="LUTmem",
            primitive=primitive,
            value=lut_values,
        )
        return lut_variable.generate_initialization()

    def generate_config_file(self, config_dict):
        """
        Generate a self-contained C++ test config (input/output tensors + LUT),
        supporting signed/unsigned input and signed/unsigned output independently.

        Expected config_dict keys (in addition to your existing ones):
        INPUT_IS_UNSIGNED (bool, optional; default False)
        OUTPUT_IS_UNSIGNED (bool, optional; default False)
        INPUT_DATAWIDTH / OUTPUT_DATAWIDTH (int)
        """
        in_unsigned = bool(config_dict.get("INPUT_IS_UNSIGNED", False))
        out_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False))
        config_dict["EXP_PRECISION"] = 12  # Number of bits for LUT output (Q0.16 format for max precision)
        config_dict["DIV_PRECISION"] = 32

        in_bits = int(config_dict["INPUT_DATAWIDTH"])
        out_bits = int(config_dict["OUTPUT_DATAWIDTH"])
        exp_bits = config_dict["EXP_PRECISION"]
        div_bits = config_dict["DIV_PRECISION"]
        config_dict["LUT_SIZE"] = 1 << in_bits  # LUT size must match input index domain

        onnx_in_type = self.get_tensorproto_dtype(in_bits, in_unsigned)
        onnx_out_type = self.get_tensorproto_dtype(out_bits, out_unsigned)
        np_in_type = self.get_numpy_dtype(in_bits, in_unsigned)
        np_out_type = self.get_numpy_dtype(out_bits, out_unsigned)

        # Random input tensor in correct integer domain/range
        in_info = np.iinfo(np_in_type)
        input_tensor = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,  # randint upper bound is exclusive
            size=(
                1,
                config_dict["IN_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ),
            dtype=np_in_type,
        )

        X = helper.make_tensor_value_info(
            "X",
            onnx_in_type,
            [1, config_dict["IN_CH"], config_dict["IN_HEIGHT"], config_dict["IN_WIDTH"]],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_out_type,
            [1, config_dict["IN_CH"], config_dict["IN_HEIGHT"], config_dict["IN_WIDTH"]],
        )

        X_scale = helper.make_tensor("X_scale", TensorProto.FLOAT, [], [float(config_dict["X_SCALE"])])
        Y_scale = helper.make_tensor("Y_scale", TensorProto.FLOAT, [], [float(config_dict["Y_SCALE"])])

        # ZPs must match the tensor element types
        X_zp = helper.make_tensor("X_zp", onnx_in_type, [], [int(config_dict["X_ZP"])])
        Y_zp = helper.make_tensor("Y_zp", onnx_out_type, [], [int(config_dict["Y_ZP"])])

        dqlinear = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "X_scale", "X_zp"],
            outputs=["X_dq"],
        )

        SoftMax = helper.make_node(
            "Softmax",
            inputs=["X_dq"],
            outputs=["Y_dq"],
            axis=1,  # Channel dimension
        )

        qlinear = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dq", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [dqlinear, SoftMax, qlinear],
            "qsoftmax_test",
            [X],
            [Y],
            initializer=[X_scale, X_zp, Y_scale, Y_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        y = sess.run(None, {"X": input_tensor})[0]

        # Optional: cast ORT output to expected numpy dtype (ORT should already produce correct dtype)
        y = y.astype(np_out_type, copy=False)

        # shift based on Po2 scales (assumes ratio is power-of-two)
        shift = int(
            np.log2(float(config_dict["Y_SCALE"]) / 2 ** -(div_bits - exp_bits))
        )  # Output is in Q0.31 format for max precision
        acc_bits = (
            int(np.floor(np.log2(config_dict["IN_CH"]) + 1)) + exp_bits
        )  # Bits needed to accumulate IN_CH exponentials without overflow

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()

        for key, value in config_dict.items():
            if key in ["X_SCALE", "W_SCALE", "Y_SCALE"]:
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

        cwr.add_line(f"typedef ap_uint<{exp_bits}> TLUT;")
        cwr.add_line(f"typedef ap_uint<{acc_bits}> TAcc;")
        cwr.add_line(f"typedef ap_uint<{div_bits}> TDiv;")
        cwr.add_line(f"typedef DequantQuantPo2<{shift}, TDiv, TOutput> Quantizer;")

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

        cwr.add_lines(self.generate_lut_memory(config_dict).code)

        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_8bit_po2_signed(self, hls_steps):
        np.random.seed(0)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": True,
            "IN_HEIGHT": 12,
            "IN_WIDTH": 12,
            "IN_CH": 20,
            "CH_PAR": 1,
            "W_PAR": 1,
            "X_SCALE": 2**-3,
            "Y_SCALE": 2**-6,
            "X_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
