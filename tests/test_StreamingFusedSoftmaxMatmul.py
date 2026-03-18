import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
import onnx
from .base_hls_test import BaseHLSTest

class TestStreamingFusedSoftmaxMatmul(BaseHLSTest):

    @property
    def operator_filename(self):
        return "StreamingFusedSoftmaxMatmul"

    @property
    def unit_filename(self):
        return "StreamingFusedSoftmaxMatmul"

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

        x_scale = float(config_dict["QK_SCALE"])

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

        # There is actually a quantization node between the softmax and matmul in the model, 
        # so the LUT output is clipped to the quantized range of the softmax output tensor scaled.
        # i.e., if the output is 8 bits and the exponential precision is 12 bits, we need to clip the LUT
        # output to [0, 127 * 2^(12-8)] to avoid overflow in the quantized softmax output.
        # if config_dict.get("SATURATE", True):
        #     qmax = min(qmax, (int(1 << (nbits - 1)) - 1) * (1 << (F - nbits + 1)))  # max representable value in quantized softmax output 
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
        in_qk_unsigned = bool(config_dict.get("INPUT_QK_IS_UNSIGNED", False))
        in_v_unsigned = bool(config_dict.get("INPUT_V_IS_UNSIGNED", False))
        out_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False))
        config_dict["EXP_PRECISION"] = 10  # Number of bits for LUT output (Q0.16 format for max precision)
        config_dict["DIV_PRECISION"] = 32

        in_bits = int(config_dict["INPUT_DATAWIDTH"])
        out_bits = int(config_dict["OUTPUT_DATAWIDTH"])
        exp_bits = config_dict["EXP_PRECISION"]
        div_bits = config_dict["DIV_PRECISION"]
        config_dict["LUT_SIZE"] = 1 << in_bits  # LUT size must match input index domain

        onnx_qk_type = self.get_tensorproto_dtype(in_bits, in_qk_unsigned)
        onnx_v_type = self.get_tensorproto_dtype(in_bits, in_v_unsigned)
        onnx_out_type = self.get_tensorproto_dtype(out_bits, out_unsigned)
        np_qk_type = self.get_numpy_dtype(in_bits, in_qk_unsigned)
        np_v_type = self.get_numpy_dtype(in_bits, in_v_unsigned)
        np_out_type = self.get_numpy_dtype(out_bits, out_unsigned)

        # Random input tensor in correct integer domain/range
        in_info = np.iinfo(np_qk_type)
        qk_tensor = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,  # randint upper bound is exclusive
            size=(
                1,
                config_dict["DIM_HEADS"],
                config_dict["DIM_SEQ"],
                config_dict["DIM_SEQ"],
            ),
            dtype=np_qk_type,
        )

        in_info = np.iinfo(np_v_type)
        v_tensor = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,  # randint upper bound is exclusive
            size=(
                1,
                config_dict["DIM_HEADS"],
                config_dict["DIM_V"],
                config_dict["DIM_SEQ"],
            ),
            dtype=np_v_type,
        )

        QK = helper.make_tensor_value_info(
            "QK",
            onnx_qk_type,
            [1, config_dict["DIM_HEADS"], config_dict["DIM_SEQ"], config_dict["DIM_SEQ"]],
        )
        V = helper.make_tensor_value_info(
            "V",
            onnx_v_type,
            [1, config_dict["DIM_HEADS"], config_dict["DIM_V"], config_dict["DIM_SEQ"]],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_out_type,
            [1, config_dict["DIM_HEADS"], config_dict["DIM_V"], config_dict["DIM_SEQ"]],
        )

        QK_scale = helper.make_tensor("QK_scale", TensorProto.FLOAT, [], [float(config_dict["QK_SCALE"])])
        V_scale = helper.make_tensor("V_scale", TensorProto.FLOAT, [], [float(config_dict["V_SCALE"])])
        Y_scale = helper.make_tensor("Y_scale", TensorProto.FLOAT, [], [float(config_dict["Y_SCALE"])])

        # ZPs must match the tensor element types
        QK_zp = helper.make_tensor("QK_zp", onnx_qk_type, [], [int(config_dict["QK_ZP"])])
        V_zp = helper.make_tensor("V_zp", onnx_v_type, [], [int(config_dict["V_ZP"])])
        Y_zp = helper.make_tensor("Y_zp", onnx_out_type, [], [int(config_dict["Y_ZP"])])

        dqlinear_qk = helper.make_node(
            "DequantizeLinear",
            inputs=["QK", "QK_scale", "QK_zp"],
            outputs=["QK_dq"],
        )

        dqlinear_v = helper.make_node(
            "DequantizeLinear",
            inputs=["V", "V_scale", "V_zp"],
            outputs=["V_dq"],
        )

        SoftMax = helper.make_node(
            "Softmax",
            inputs=["QK_dq"],
            outputs=["A_dq"],
            axis=-1,
        )

        transpose_softmax = helper.make_node(
            "Transpose",
            inputs=["A_dq"],
            outputs=["A_transposed"],
            perm=[0, 1, 3, 2],
        )

        matmul = helper.make_node(
            "MatMul",
            inputs=["V_dq", "A_transposed"],
            outputs=["Y_dq"],
        )

        qlinear_output = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dq", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [
                dqlinear_qk,
                dqlinear_v,
                SoftMax,
                transpose_softmax,
                matmul,
                qlinear_output,
            ],
            "qfusedsoftmaxmatmul_test",
            [QK, V],
            [Y],
            initializer=[QK_scale, QK_zp, V_scale, V_zp, Y_scale, Y_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        y = sess.run(None, {"QK": qk_tensor, "V": v_tensor})[0]

        # Optional: cast ORT output to expected numpy dtype (ORT should already produce correct dtype)
        y = y.astype(np_out_type, copy=False)
        # a = a.astype(np_out_type, copy=False)

        # shift based on Po2 scales (assumes ratio is power-of-two)
        den_bits = (
            int(np.floor(np.log2(config_dict["DIM_SEQ"]) + 1)) + exp_bits
        )  # Bits needed to accumulate DIM_SEQ exponentials without overflow
        num_bits = (
            int(np.floor(np.log2(config_dict["DIM_SEQ"]) + 1))
            + in_bits
            + exp_bits
        )  # Bits needed to accumulate DIM_SEQ products of input * exponential without overflow
        shift_num = div_bits - num_bits
        shift = int(
            np.log2(float(config_dict["Y_SCALE"]) / float(config_dict["V_SCALE"]))
        )
        shift += shift_num # Additional shift to increase precision for the division step, since we have extra bits in TDiv

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()

        for key, value in config_dict.items():
            if key in ["QK_SCALE", "V_SCALE", "A_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {float(value)}f;")
            else:
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                    cwr.add_line(f"const bool {key} = {value_str};")
                else:
                    cwr.add_line(f"const int {key} = {int(value)};")

        typedef_suffix = "u" if in_qk_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{in_bits}> TQKInput;")
        typedef_suffix = "u" if in_v_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{in_bits}> TVInput;")
        typedef_suffix = "u" if out_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{out_bits}> TOutput;")

        cwr.add_line(f"typedef ap_uint<{exp_bits}> TLUT;")
        typedef_suffix = "u" if in_v_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{num_bits}> TNum;")
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{div_bits}> TDiv;")
        cwr.add_line(f"typedef ap_uint<{den_bits}> TDen;")
        cwr.add_line(f"typedef DequantQuantPo2<{shift}, TDiv, TOutput> Quantizer;")

        cwr.add_lines(
            csnake.Variable(
                "qk_tensor",
                primitive="TQKInput",
                value=qk_tensor,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "v_tensor",
                primitive="TVInput",
                value=v_tensor,
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
            "INPUT_QK_IS_UNSIGNED": False,
            "INPUT_V_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "DIM_HEADS": 2,
            "DIM_V": 40,
            "DIM_SEQ": 40,
            "REDUCE_PAR": 1,
            "QK_SCALE": 2**-3,
            "V_SCALE": 2**-4,
            "Y_SCALE": 2**-4,
            "QK_ZP": 0,
            "V_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
