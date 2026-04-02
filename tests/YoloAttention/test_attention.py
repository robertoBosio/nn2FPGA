import numpy as np
import onnxruntime as ort
import csnake
import onnx
from onnx import TensorProto, helper
from tests.base_hls_test import BaseHLSTest

class TestQKMatMul(BaseHLSTest):

    @property
    def operator_filename(self):
        return ["YoloAttention/QKMatMul", "YoloAttention/SplitReshape", "StreamingSoftmax", "YoloAttention/VPMatMul", "StreamingConstMul"]

    @property
    def unit_filename(self):
        return "YoloAttention/Attention"

    def generate_lut_memory(self, softmax_input_bits, exp_precision, softmax_scale, lut_bits):
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
        nbits = int(softmax_input_bits)
        F = int(exp_precision)
        lut_entries = 1 << nbits

        x_scale = float(softmax_scale)

        # Total output bits for the LUT integer values.
        # Default to F+1 so exp(0)=2^F is representable.
        out_total_bits = int(lut_bits)

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
        Generate a self-contained C++ test config (input/output tensors and operator typedefs),
        supporting signed/unsigned input and signed/unsigned output independently.

        Expected config_dict keys (in addition to your existing ones):
        INPUT_IS_UNSIGNED (bool, optional; default False)
        OUTPUT_IS_UNSIGNED (bool, optional; default False)
        INPUT_DATAWIDTH / OUTPUT_DATAWIDTH (int)
        """

        in_unsigned = bool(config_dict.get("INPUT_IS_UNSIGNED", False))
        out_matmul_unsigned = bool(config_dict.get("OUTPUT_MATMUL_IS_UNSIGNED", False))
        out_v_unsigned = bool(config_dict.get("OUTPUT_V_IS_UNSIGNED", False))
        const_unsigned = bool(config_dict.get("CONST_IS_UNSIGNED", False))

        in_bits = int(config_dict["INPUT_DATAWIDTH"])
        out_matmul_bits = int(config_dict["OUTPUT_MATMUL_DATAWIDTH"])
        out_v_bits = int(config_dict["OUTPUT_V_DATAWIDTH"])
        exp_precision = config_dict["EXP_PRECISION"]
        const_bits = int(config_dict.get("CONST_DATAWIDTH", 9))

        onnx_in_type = self.get_tensorproto_dtype(in_bits, in_unsigned)
        onnx_out_matmul_type = self.get_tensorproto_dtype(out_matmul_bits, out_matmul_unsigned)
        onnx_out_v_type = self.get_tensorproto_dtype(out_v_bits, out_v_unsigned)
        mul_const_type = self.get_tensorproto_dtype(
            const_bits,
            const_unsigned,
        )
        np_in_type = self.get_numpy_dtype(in_bits, in_unsigned)
        np_out_matmul_type = self.get_numpy_dtype(out_matmul_bits, out_matmul_unsigned)
        np_out_v_type = self.get_numpy_dtype(out_v_bits, out_v_unsigned)

        # Random input tensor in correct integer domain/range
        in_info = np.iinfo(np_in_type)
        input_tensor = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,
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

        Y_out = helper.make_tensor_value_info(
            "Y_reshaped_q",
            onnx_out_matmul_type,
            [1, config_dict["OUT_CH"], config_dict["OUT_HEIGHT"], config_dict["OUT_WIDTH"]],
        )

        V_out = helper.make_tensor_value_info(
            "V_q_reshaped",
            onnx_out_v_type,
            [1, config_dict["OUT_CH"], config_dict["OUT_HEIGHT"], config_dict["OUT_WIDTH"]],
        )

        # Debug tensor for the intermediate P_q output of the softmax
        Y_q = helper.make_tensor_value_info(
            "Y_q",
            onnx_out_matmul_type,
            [
                1,
                2,
                64,
                config_dict["IN_HEIGHT"] * config_dict["IN_WIDTH"],
            ],
        )

        def make_const(name, vals, data_type=TensorProto.INT64, dims=None):
            arr = np.array(vals)
            if dims is None:
                dims = list(arr.shape) if arr.shape else []
            flat_vals = arr.flatten().tolist() if arr.shape else [arr.item()]
            return helper.make_node(
                "Constant",
                inputs=[],
                outputs=[name],
                value=helper.make_tensor(
                    name=f"{name}_value",
                    data_type=data_type,
                    dims=dims,
                    vals=flat_vals,
                ),
            )

        mul_value = helper.make_tensor(
            "MUL_value", mul_const_type, [], [int(config_dict["MUL_CONST"])]
        )
        mul_scale = helper.make_tensor(
            "MUL_scale", TensorProto.FLOAT, [], [float(config_dict["CONST_SCALE"])]
        )
        mul_zp = helper.make_tensor(
            "MUL_zp", mul_const_type, [], [int(config_dict["CONST_ZP"])]
        )

        X_scale = helper.make_tensor("X_scale", TensorProto.FLOAT, [], [float(config_dict["X_SCALE"])])
        Y_scale = helper.make_tensor("Y_scale", TensorProto.FLOAT, [], [float(config_dict["Y_SCALE"])])
        V_scale = helper.make_tensor("V_scale", TensorProto.FLOAT, [], [float(config_dict["V_SCALE"])])
        Q_scale = helper.make_tensor("Q_scale", TensorProto.FLOAT, [], [float(config_dict["Q_SCALE"])])
        K_scale = helper.make_tensor("K_scale", TensorProto.FLOAT, [], [float(config_dict["K_SCALE"])])
        QK_scale = helper.make_tensor("QK_scale", TensorProto.FLOAT, [], [float(config_dict["QK_SCALE"])])
        QK_scaled_scale = helper.make_tensor(
            "QK_scaled_scale", TensorProto.FLOAT, [], [float(config_dict["QK_SCALED_SCALE"])]
        )
        P_scale = helper.make_tensor("P_scale", TensorProto.FLOAT, [], [float(config_dict["P_SCALE"])])

        X_zp = helper.make_tensor("X_zp", onnx_in_type, [], [int(config_dict["X_ZP"])])
        Y_zp = helper.make_tensor("Y_zp", onnx_out_matmul_type, [], [int(config_dict["Y_ZP"])])
        V_zp = helper.make_tensor("V_zp", onnx_out_v_type, [], [int(config_dict["V_ZP"])])
        Q_zp = helper.make_tensor("Q_zp", onnx_in_type, [], [int(config_dict["Q_ZP"])])
        K_zp = helper.make_tensor("K_zp", onnx_in_type, [], [int(config_dict["K_ZP"])])
        QK_zp = helper.make_tensor("QK_zp", onnx_out_matmul_type, [], [int(config_dict["QK_ZP"])])
        QK_scaled_zp = helper.make_tensor(
            "QK_scaled_zp", onnx_out_matmul_type, [], [int(config_dict["QK_SCALED_ZP"])]
        )
        P_zp = helper.make_tensor("P_zp", onnx_out_matmul_type, [], [int(config_dict["P_ZP"])])

        dqlinearx = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "X_scale", "X_zp"],
            outputs=["X_dq"],
        )

        shapex = make_const(
            "shapeX",
            [
                1,
                2,
                128,
                config_dict["IN_HEIGHT"] * config_dict["IN_WIDTH"],
            ],
        )

        reshapex = helper.make_node(
            "Reshape",
            inputs=["X_dq", "shapeX"],
            outputs=["X_reshaped"],
        )

        axes_slices = make_const("axes_slices", [2])
        steps_slices = make_const("steps_slices", [1])

        starts_sliceq = make_const("starts_sliceq", [0])
        ends_sliceq = make_const("ends_sliceq", [32])
        sliceq = helper.make_node(
            "Slice",
            inputs=["X_reshaped", "starts_sliceq", "ends_sliceq", "axes_slices", "steps_slices"],
            outputs=["Q"],
        )

        starts_slicek = make_const("starts_slicek", [32])
        ends_slicek = make_const("ends_slicek", [64])
        slicek = helper.make_node(
            "Slice",
            inputs=["X_reshaped", "starts_slicek", "ends_slicek", "axes_slices", "steps_slices"],
            outputs=["K"],
        )

        starts_slicev = make_const("starts_slicev", [64])
        ends_slicev = make_const("ends_slicev", [128])
        slicev = helper.make_node(
            "Slice",
            inputs=["X_reshaped", "starts_slicev", "ends_slicev", "axes_slices", "steps_slices"],
            outputs=["V"],
        )

        qlinearq = helper.make_node(
            "QuantizeLinear",
            inputs=["Q", "Q_scale", "Q_zp"],
            outputs=["Q_q"],
        )

        qlineark = helper.make_node(
            "QuantizeLinear",
            inputs=["K", "K_scale", "K_zp"],
            outputs=["K_q"],
        )

        dqlinearq = helper.make_node(
            "DequantizeLinear",
            inputs=["Q_q", "Q_scale", "Q_zp"],
            outputs=["Q_dq"],
        )

        dqlineark = helper.make_node(
            "DequantizeLinear",
            inputs=["K_q", "K_scale", "K_zp"],
            outputs=["K_dq"],
        )

        transposeq = helper.make_node(
            "Transpose",
            inputs=["Q_dq"],
            outputs=["Q_dq_transposed"],
            perm=[0, 1, 3, 2],
        )

        matmulqk = helper.make_node(
            "MatMul",
            inputs=["Q_dq_transposed", "K_dq"],
            outputs=["QK"],
        )

        qlinearqk = helper.make_node(
            "QuantizeLinear",
            inputs=["QK", "QK_scale", "QK_zp"],
            outputs=["QK_q"],
        )

        dqlinearqk = helper.make_node(
            "DequantizeLinear",
            inputs=["QK_q", "QK_scale", "QK_zp"],
            outputs=["QK_dq"],
        )

        dequantconst = helper.make_node(
            "DequantizeLinear",
            inputs=["MUL_value", "MUL_scale", "MUL_zp"],
            outputs=["MUL_dequant"],
        )

        mul = helper.make_node(
            "Mul",
            inputs=["QK_dq", "MUL_dequant"],
            outputs=["QK_scaled"],
        )

        qlinearqk_scaled = helper.make_node(
            "QuantizeLinear",
            inputs=["QK_scaled", "QK_scaled_scale", "QK_scaled_zp"],
            outputs=["QK_scaled_q"],
        )

        dqlinearqk_scaled = helper.make_node(
            "DequantizeLinear",
            inputs=["QK_scaled_q", "QK_scaled_scale", "QK_scaled_zp"],
            outputs=["QK_scaled_dq"],
        )

        softmax = helper.make_node(
            "Softmax",
            inputs=["QK_scaled_dq"],
            outputs=["P"],
            axis=-1,
        )

        qlinearp = helper.make_node(
            "QuantizeLinear",
            inputs=["P", "P_scale", "P_zp"],
            outputs=["P_q"],
        )

        dqlinearp = helper.make_node(
            "DequantizeLinear",
            inputs=["P_q", "P_scale", "P_zp"],
            outputs=["P_dq"],
        )

        transposep = helper.make_node(
            "Transpose",
            inputs=["P_dq"],
            outputs=["P_dq_transposed"],
            perm=[0, 1, 3, 2],
        )

        shapeout = make_const(
            "shapeout",
            [
                1,
                config_dict["OUT_CH"],
                config_dict["OUT_HEIGHT"],
                config_dict["OUT_WIDTH"],
            ],
        )

        qlinearv = helper.make_node(
            "QuantizeLinear",
            inputs=["V", "V_scale", "V_zp"],
            outputs=["V_q"],
        )

        dqlinearv = helper.make_node(
            "DequantizeLinear",
            inputs=["V_q", "V_scale", "V_zp"],
            outputs=["V_dq"],
        )

        reshapev = helper.make_node(
            "Reshape",
            inputs=["V_dq", "shapeout"],
            outputs=["V_reshaped"],
        )

        qlinearv2 = helper.make_node(
            "QuantizeLinear",
            inputs=["V_reshaped", "V_scale", "V_zp"],
            outputs=["V_q_reshaped"],
        )

        matmulvp = helper.make_node(
            "MatMul",
            inputs=["V_dq", "P_dq_transposed"],
            outputs=["Y"],
        )
        
        qlineary = helper.make_node(
            "QuantizeLinear",
            inputs=["Y", "Y_scale", "Y_zp"],
            outputs=["Y_q"],
        )
        
        dqlineary = helper.make_node(
            "DequantizeLinear",
            inputs=["Y_q", "Y_scale", "Y_zp"],
            outputs=["Y_dq"],
        )

        reshapey = helper.make_node(
            "Reshape",
            inputs=["Y_dq", "shapeout"],
            outputs=["Y_reshaped"],
        )

        qlinearyreshaped = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_reshaped", "Y_scale", "Y_zp"],
            outputs=["Y_reshaped_q"],
        )

        graph = helper.make_graph(
            [
                dqlinearx,
                shapex,
                reshapex,
                axes_slices,
                steps_slices,
                starts_sliceq,
                ends_sliceq,
                sliceq,
                starts_slicek,
                ends_slicek,
                slicek,
                starts_slicev,
                ends_slicev,
                slicev,
                qlinearq,
                qlineark,
                dqlinearq,
                dqlineark,
                transposeq,
                matmulqk,
                qlinearqk,
                dqlinearqk,
                mul,
                qlinearqk_scaled,
                dqlinearqk_scaled,
                softmax,
                qlinearp,
                dqlinearp,
                transposep,
                reshapev,
                qlinearv,
                dqlinearv,
                qlinearv2,
                matmulvp,
                shapeout,
                reshapey,
                qlineary,
                dqlineary,
                qlinearyreshaped,
                dequantconst,
            ],
            "attention_test",
            [X],
            [Y_out, V_out, Y_q],
            initializer=[
                mul_value,
                X_scale,
                Y_scale,
                V_scale,
                Q_scale,
                K_scale,
                QK_scale,
                QK_scaled_scale,
                mul_scale,
                P_scale,
                X_zp,
                Y_zp,
                V_zp,
                Q_zp,
                K_zp,
                QK_zp,
                QK_scaled_zp,
                P_zp,
                mul_zp,
            ],
        )

        model = helper.make_model(graph, producer_name="qonnx")
        model = onnx.shape_inference.infer_shapes(model)
        # onnx.save(model, "attention_test.onnx")
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        y, v, y_q = sess.run(None, {"X": input_tensor})

        # Optional: cast ORT output to expected numpy dtype (ORT should already produce correct dtype)
        y = y.astype(np_out_matmul_type, copy=False)
        v = v.astype(np_out_v_type, copy=False)
        y_q = y_q.astype(np_out_matmul_type, copy=False)

        ########## NEW CONFIG GENERATION ##########

        # Derived parameters from the attention graph
        dim_heads = 2
        dim_seq_qk = 32
        dim_q = 400
        dim_k = 400
        dim_v = 64
        dim_p = 400
        dim_seq_vp = 400

        assert config_dict["IN_CH"] == 256
        assert config_dict["OUT_CH"] == 128
        assert config_dict["OUT_CH"] == dim_heads * dim_v

        # Conservative serial testbench parallelism
        reduce_par = 1
        ch_par = 1
        w_par = 1
        heads_par = 1

        # Accumulator widths
        acc_bits_qk = int(np.ceil(np.log2(dim_k))) + in_bits + in_bits
        acc_bits_vp = int(np.ceil(np.log2(dim_p))) + out_v_bits + out_matmul_bits
        acc_bits_softmax = int(np.ceil(np.log2(dim_p))) + exp_precision
        div_bits_softmax = 32
        mul_bits = const_bits + in_bits

        # Use signed 8-bit everywhere here, matching the graph
        def ap_int_typename(bits, unsigned_flag):
            return f"ap_uint<{bits}>" if unsigned_flag else f"ap_int<{bits}>"

        tinput_name = ap_int_typename(in_bits, in_unsigned)
        tsplit_name = ap_int_typename(in_bits, in_unsigned)
        tqk_name = ap_int_typename(out_matmul_bits, out_matmul_unsigned)
        tqkscaled_name = ap_int_typename(out_matmul_bits, out_matmul_unsigned)
        tsoftmax_name = ap_int_typename(out_matmul_bits, out_matmul_unsigned)
        tmul_name = ap_int_typename(mul_bits, const_unsigned)
        ty_name = ap_int_typename(out_matmul_bits, out_matmul_unsigned)
        tv_name = ap_int_typename(out_v_bits, out_v_unsigned)

        # LUT settings for softmax
        softmax_input_bits = in_bits
        lut_bits = exp_precision
        lut_size = 1 << softmax_input_bits

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()

        # Original config values
        for key, value in config_dict.items():
            if isinstance(value, bool):
                cwr.add_line(f"const bool {key} = {'true' if value else 'false'};")
            elif isinstance(value, float):
                cwr.add_line(f"const float {key} = {float(value)}f;")
            else:
                cwr.add_line(f"const int {key} = {int(value)};")

        # Derived dimensions
        cwr.add_line(f"const int DIM_HEADS = {dim_heads};")
        cwr.add_line(f"const int DIM_SEQ_QK = {dim_seq_qk};")
        cwr.add_line(f"const int DIM_SEQ_VP = {dim_seq_vp};")
        cwr.add_line(f"const int DIM_Q = {dim_q};")
        cwr.add_line(f"const int DIM_K = {dim_k};")
        cwr.add_line(f"const int DIM_V = {dim_v};")
        cwr.add_line(f"const int DIM_P = {dim_p};")

        # Parallelism
        cwr.add_line(f"const int REDUCE_PAR = {reduce_par};")
        cwr.add_line(f"const int CH_PAR = {ch_par};")
        cwr.add_line(f"const int W_PAR = {w_par};")
        cwr.add_line(f"const int HEADS_PAR = {heads_par};")

        # Mul block dimensions
        cwr.add_line(f"const int MUL_HEIGHT = {dim_p};")
        cwr.add_line(f"const int MUL_WIDTH = {dim_seq_vp};")
        cwr.add_line(f"const int MUL_CH = {dim_heads};")
        cwr.add_line(f"const int MUL_W_PAR = 1;")
        cwr.add_line(f"const int MUL_CH_PAR = 1;")

        # Softmax LUT config
        cwr.add_line(f"const int LUT_SIZE = {lut_size};")

        # Scalar constant for the Mul block
        const_bits = int(config_dict.get("CONST_DATAWIDTH", 9))
        const_unsigned = bool(config_dict.get("CONST_IS_UNSIGNED", False))
        const_typename = f"ap_uint<{const_bits}>" if const_unsigned else f"ap_int<{const_bits}>"

        cwr.add_line(f"using TConst = {const_typename};")
        cwr.add_line(f"const TConst constant_scaler = {int(config_dict['MUL_CONST'])};")

        # Element types
        cwr.add_line(f"using TInput = {tinput_name};")
        cwr.add_line(f"using TSplit = {tsplit_name};")
        cwr.add_line(f"using TQK = {tqk_name};")
        cwr.add_line(f"using TQKScaled = {tqkscaled_name};")
        cwr.add_line(f"using TSoftmax = {tsoftmax_name};")
        cwr.add_line(f"using TYOutput = {ty_name};")
        cwr.add_line(f"using TMul = {tmul_name};")

        # AXI stream word types
        cwr.add_line("using TInputWord = std::array<TInput, 1>;")
        cwr.add_line("using TSplitWord = std::array<TSplit, 1>;")
        cwr.add_line("using TQKWord = std::array<TQK, 1>;")
        cwr.add_line("using TQKScaledWord = std::array<TQKScaled, 1>;")
        cwr.add_line("using TSoftmaxWord = std::array<TSoftmax, 1>;")
        cwr.add_line("using TYOutputWord = std::array<TYOutput, 1>;")

        # Accumulator / LUT types
        cwr.add_line(f"using TAccQK = ap_int<{acc_bits_qk}>;")
        cwr.add_line(f"using TAccVP = ap_int<{acc_bits_vp}>;")
        cwr.add_line(f"using TAccSoftmax = ap_uint<{acc_bits_softmax}>;")
        cwr.add_line(f"using TDivSoftmax = ap_uint<{div_bits_softmax}>;")
        cwr.add_line(f"using TLUT = ap_uint<{lut_bits}>;")

        # Quantizers / activations
        shift_qk = int(round(np.log2(
            float(config_dict["QK_SCALE"]) /
            (float(config_dict["Q_SCALE"]) * float(config_dict["K_SCALE"]))
        )))

        shift_mul = int(round(np.log2(
            float(config_dict["QK_SCALED_SCALE"]) /
            (float(config_dict["QK_SCALE"]) * float(config_dict["CONST_SCALE"]))
        )))

        shift_vp = int(round(np.log2(
            float(config_dict["Y_SCALE"]) /
            (float(config_dict["V_SCALE"]) * float(config_dict["P_SCALE"]))
        )))

        shift_softmax = int(round(np.log2(
            float(config_dict["P_SCALE"]) / 2 ** -( div_bits_softmax - exp_precision)
        )))

        cwr.add_line("using SplitQuantizer = DequantQuantEqual<TSplit>;")
        cwr.add_line(f"using QKQuantizer = DequantQuantPo2<{shift_qk}, TAccQK, TQK>;")
        cwr.add_line("using MulActivation = DequantQuantEqual<TMul>;")
        cwr.add_line(f"using MulQuantizer = DequantQuantPo2<{shift_mul}, TMul, TQKScaled>;")
        cwr.add_line(f"using SoftmaxQuantizer = DequantQuantPo2<{shift_softmax}, TDivSoftmax, TSoftmax>;")
        cwr.add_line(f"using VPQuantizer = DequantQuantPo2<{shift_vp}, TAccVP, TYOutput>;")

        # Input and golden tensors
        cwr.add_lines(
            csnake.Variable(
                "input_tensor",
                primitive="TInput",
                value=input_tensor,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "output_tensor_v",
                primitive="TSplit",
                value=v,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "output_tensor_y",
                primitive="TYOutput",
                value=y,
            ).generate_initialization()
        )
        # cwr.add_lines(
        #     csnake.Variable(
        #         "output_tensor_P",
        #         primitive="float",
        #         value=p,
        #     ).generate_initialization()
        # )
        # cwr.add_lines(
        #     csnake.Variable(
        #         "output_tensor_attention",
        #         primitive="TYOutput",
        #         value=y_q,
        #     ).generate_initialization()
        # )

        # Softmax LUT contents
        cwr.add_lines(
            self.generate_lut_memory(
                softmax_input_bits=softmax_input_bits,
                exp_precision=exp_precision,
                softmax_scale=config_dict["QK_SCALED_SCALE"],
                lut_bits=lut_bits,
            )
        )

        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_8bit_po2_signed(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_MATMUL_DATAWIDTH": 8,
            "OUTPUT_V_DATAWIDTH": 8,
            "CONST_DATAWIDTH": 8,
            "EXP_PRECISION": 12,
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_MATMUL_IS_UNSIGNED": False,
            "OUTPUT_V_IS_UNSIGNED": False,
            "CONST_IS_UNSIGNED": False,
            "IN_CH": 256,
            "IN_HEIGHT": 20,
            "IN_WIDTH": 20,
            "OUT_CH": 128,
            "OUT_HEIGHT": 20,
            "OUT_WIDTH": 20,
            "X_SCALE": 2**-4,
            "Q_SCALE": 2**-4,
            "K_SCALE": 2**-4,
            "QK_SCALE": 2**-1,
            "CONST_SCALE": 2**-9,
            "QK_SCALED_SCALE": 2**-3,
            "P_SCALE": 2**-8,
            "Y_SCALE": 2**-4,
            "V_SCALE": 2**-4,
            "MUL_CONST": 91,
            "X_ZP": 0,
            "Y_ZP": 0,
            "V_ZP": 0,
            "Q_ZP": 0,
            "K_ZP": 0,
            "Y_ZP": 0,
            "QK_ZP": 0,
            "CONST_ZP": 0,
            "QK_SCALED_ZP": 0,
            "P_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps, workdir=".", clean=False)
