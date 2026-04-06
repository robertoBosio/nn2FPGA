import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from tests.base_hls_test import BaseHLSTest

class TestQKMatMul(BaseHLSTest):

    @property
    def operator_filename(self):
        return "YoloAttention/QKMatMul"

    @property
    def unit_filename(self):
        return "YoloAttention/QKMatMul"

    def generate_config_file(self, config_dict):
        """
        Generate a self-contained C++ test config (input/output tensors and operator typedefs),
        supporting signed/unsigned input and signed/unsigned output independently.

        Expected config_dict keys (in addition to your existing ones):
        INPUT_IS_UNSIGNED (bool, optional; default False)
        OUTPUT_IS_UNSIGNED (bool, optional; default False)
        INPUT_DATAWIDTH / OUTPUT_DATAWIDTH (int)
        """

        inq_unsigned = bool(config_dict.get("INPUTQ_IS_UNSIGNED", False))
        ink_unsigned = bool(config_dict.get("INPUTK_IS_UNSIGNED", False))
        out_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False))

        in_qbits = int(config_dict["INPUTQ_DATAWIDTH"])
        in_kbits = int(config_dict["INPUTK_DATAWIDTH"])
        out_bits = int(config_dict["OUTPUT_DATAWIDTH"])

        onnx_in_qtype = self.get_tensorproto_dtype(in_qbits, inq_unsigned)
        onnx_in_ktype = self.get_tensorproto_dtype(in_kbits, ink_unsigned)
        onnx_out_type = self.get_tensorproto_dtype(out_bits, out_unsigned)
        np_in_qtype = self.get_numpy_dtype(in_qbits, inq_unsigned)
        np_in_ktype = self.get_numpy_dtype(in_kbits, ink_unsigned)
        np_out_type = self.get_numpy_dtype(out_bits, out_unsigned)

        # Random input tensor in correct integer domain/range
        in_info = np.iinfo(np_in_qtype)
        input_tensorq = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,  # randint upper bound is exclusive
            size=(
                1,
                config_dict["DIM_HEADS"],
                config_dict["DIM_SEQ"],
                config_dict["DIM_Q"],
            ),
            dtype=np_in_qtype,
        )

        in_info = np.iinfo(np_in_ktype)
        input_tensork = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,  # randint upper bound is exclusive
            size=(
                1,
                config_dict["DIM_HEADS"],
                config_dict["DIM_SEQ"],
                config_dict["DIM_K"],
            ),
            dtype=np_in_ktype,
        )

        Q = helper.make_tensor_value_info(
            "Q",
            onnx_in_qtype,
            [1, config_dict["DIM_HEADS"], config_dict["DIM_SEQ"], config_dict["DIM_Q"]],
        )
        K = helper.make_tensor_value_info(
            "K",
            onnx_in_ktype,
            [1, config_dict["DIM_HEADS"], config_dict["DIM_SEQ"], config_dict["DIM_K"]],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_out_type,
            [1, config_dict["DIM_HEADS"], config_dict["DIM_Q"], config_dict["DIM_K"]],
        )

        Q_scale = helper.make_tensor("Q_scale", TensorProto.FLOAT, [], [float(config_dict["Q_SCALE"])])
        K_scale = helper.make_tensor("K_scale", TensorProto.FLOAT, [], [float(config_dict["K_SCALE"])])
        Y_scale = helper.make_tensor("Y_scale", TensorProto.FLOAT, [], [float(config_dict["Y_SCALE"])])
        Q_zp = helper.make_tensor("Q_zp", onnx_in_qtype, [], [int(config_dict["Q_ZP"])])
        K_zp = helper.make_tensor("K_zp", onnx_in_ktype, [], [int(config_dict["K_ZP"])])
        Y_zp = helper.make_tensor("Y_zp", onnx_out_type, [], [int(config_dict["Y_ZP"])])

        dqlinearq = helper.make_node(
            "DequantizeLinear",
            inputs=["Q", "Q_scale", "Q_zp"],
            outputs=["Q_dq"],
        )

        transpose = helper.make_node(
            "Transpose",
            inputs=["Q_dq"],
            outputs=["Q_dq_transposed"],
            perm=[0,1,3,2]
        )

        dqlineark = helper.make_node(
            "DequantizeLinear",
            inputs=["K", "K_scale", "K_zp"],
            outputs=["K_dq"],
        )

        matmul = helper.make_node(
            "MatMul",
            inputs=["Q_dq_transposed", "K_dq"],
            outputs=["Y_dq"],
        )

        qlinear = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dq", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [
                dqlinearq,
                dqlineark,
                transpose,
                matmul,
                qlinear
            ],
            "qkmatmul_test",
            [Q, K],
            [Y],
            initializer=[Q_scale, Q_zp, K_scale, K_zp, Y_scale, Y_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        y = sess.run(None, {"Q": input_tensorq, "K": input_tensork})[0]

        # Optional: cast ORT output to expected numpy dtype (ORT should already produce correct dtype)
        y = y.astype(np_out_type, copy=False)

        shift = int(
            np.log2(
                float(config_dict["Y_SCALE"])
                / float(config_dict["Q_SCALE"] * config_dict["K_SCALE"])
            )
        )
        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.include("DequantQuant.hpp")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        acc_bits = (
            int(np.floor(np.log2(config_dict["DIM_SEQ"]) + 1)) + in_qbits + in_kbits
        )

        for key, value in config_dict.items():
            if key in ["Q_SCALE", "K_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {float(value)}f;")
            else:
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                    cwr.add_line(f"const bool {key} = {value_str};")
                else:
                    cwr.add_line(f"const int {key} = {int(value)};")

        typedef_suffix = "u" if inq_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{in_qbits}> TQInput;")
        cwr.add_line(f"using TQInputWord = std::array<TQInput, REDUCE_PAR>;")
        typedef_suffix = "u" if ink_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{in_kbits}> TKInput;")
        cwr.add_line(f"using TKInputWord = std::array<TKInput, REDUCE_PAR>;")

        typedef_suffix = "u" if out_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{out_bits}> TOutput;")
        cwr.add_line(f"using TOutputWord = std::array<TOutput, 1>;")

        cwr.add_line(f"typedef ap_int<{acc_bits}> TAcc;")

        cwr.add_line(f"typedef DequantQuantPo2<{shift}, TAcc, TOutput> Quantizer;")

        cwr.add_lines(
            csnake.Variable(
                "q_tensor",
                primitive="TQInput",
                value=input_tensorq,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "k_tensor",
                primitive="TKInput",
                value=input_tensork,
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

    def test_8bit_po2_signed(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTQ_DATAWIDTH": 8,
            "INPUTK_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "INPUTQ_IS_UNSIGNED": False,
            "INPUTK_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "DIM_HEADS": 2,
            "DIM_Q": 2,
            "DIM_K": 2,
            "DIM_SEQ": 2,
            "REDUCE_PAR": 1,
            "HEADS_PAR": 1,
            "Q_SCALE": 2**-4,
            "K_SCALE": 2**-4,
            "Y_SCALE": 2**-1,
            "Q_ZP": 0,
            "K_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
