import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from tests.base_hls_test import BaseHLSTest

class TestVPMatMul(BaseHLSTest):

    @property
    def operator_filename(self):
        return "YoloAttention/VPMatMul"

    @property
    def unit_filename(self):
        return "YoloAttention/VPMatMul"

    def generate_config_file(self, config_dict):
        """
        Generate a self-contained C++ test config (input/output tensors and operator typedefs),
        supporting signed/unsigned input and signed/unsigned output independently.

        Expected config_dict keys (in addition to your existing ones):
        INPUT_IS_UNSIGNED (bool, optional; default False)
        OUTPUT_IS_UNSIGNED (bool, optional; default False)
        INPUT_DATAWIDTH / OUTPUT_DATAWIDTH (int)
        """

        inv_unsigned = bool(config_dict.get("INPUTV_IS_UNSIGNED", False))
        inp_unsigned = bool(config_dict.get("INPUTP_IS_UNSIGNED", False))
        out_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False))

        in_vbits = int(config_dict["INPUTV_DATAWIDTH"])
        in_pbits = int(config_dict["INPUTP_DATAWIDTH"])
        out_bits = int(config_dict["OUTPUT_DATAWIDTH"])

        onnx_in_vtype = self.get_tensorproto_dtype(in_vbits, inv_unsigned)
        onnx_in_ptype = self.get_tensorproto_dtype(in_pbits, inp_unsigned)
        onnx_out_type = self.get_tensorproto_dtype(out_bits, out_unsigned)
        np_in_vtype = self.get_numpy_dtype(in_vbits, inv_unsigned)
        np_in_ptype = self.get_numpy_dtype(in_pbits, inp_unsigned)
        np_out_type = self.get_numpy_dtype(out_bits, out_unsigned)

        # Random input tensor in correct integer domain/range
        in_info = np.iinfo(np_in_vtype)
        input_tensorv = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,  # randint upper bound is exclusive
            size=(
                1,
                config_dict["DIM_HEADS"],
                config_dict["DIM_V"],
                config_dict["DIM_SEQ"],
            ),
            dtype=np_in_vtype,
        )

        in_info = np.iinfo(np_in_ptype)
        input_tensorp = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,  # randint upper bound is exclusive
            size=(
                1,
                config_dict["DIM_HEADS"],
                config_dict["DIM_P"],
                config_dict["DIM_SEQ"],
            ),
            dtype=np_in_ptype,
        )

        V = helper.make_tensor_value_info(
            "V",
            onnx_in_vtype,
            [1, config_dict["DIM_HEADS"], config_dict["DIM_V"], config_dict["DIM_SEQ"]],
        )
        P = helper.make_tensor_value_info(
            "P",
            onnx_in_ptype,
            [1, config_dict["DIM_HEADS"], config_dict["DIM_P"], config_dict["DIM_SEQ"]],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_out_type,
            [1, config_dict["DIM_HEADS"], config_dict["DIM_V"], config_dict["DIM_P"]],
        )

        V_scale = helper.make_tensor("V_scale", TensorProto.FLOAT, [], [float(config_dict["V_SCALE"])])
        P_scale = helper.make_tensor("P_scale", TensorProto.FLOAT, [], [float(config_dict["P_SCALE"])])
        Y_scale = helper.make_tensor("Y_scale", TensorProto.FLOAT, [], [float(config_dict["Y_SCALE"])])
        V_zp = helper.make_tensor("V_zp", onnx_in_vtype, [], [int(config_dict["V_ZP"])])
        P_zp = helper.make_tensor("P_zp", onnx_in_ptype, [], [int(config_dict["P_ZP"])])
        Y_zp = helper.make_tensor("Y_zp", onnx_out_type, [], [int(config_dict["Y_ZP"])])

        dqlinearv = helper.make_node(
            "DequantizeLinear",
            inputs=["V", "V_scale", "V_zp"],
            outputs=["V_dq"],
        )


        dqlinearp = helper.make_node(
            "DequantizeLinear",
            inputs=["P", "P_scale", "P_zp"],
            outputs=["P_dq"],
        )

        transpose = helper.make_node(
            "Transpose",
            inputs=["P_dq"],
            outputs=["P_dq_transposed"],
            perm=[0,1,3,2]
        )

        matmul = helper.make_node(
            "MatMul",
            inputs=["V_dq", "P_dq_transposed"],
            outputs=["Y_dq"],
        )

        qlinear = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dq", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [
                dqlinearv,
                dqlinearp,
                transpose,
                matmul,
                qlinear
            ],
            "qkmatmul_test",
            [V, P],
            [Y],
            initializer=[V_scale, V_zp, P_scale, P_zp, Y_scale, Y_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        y = sess.run(None, {"V": input_tensorv, "P": input_tensorp})[0]

        # Optional: cast ORT output to expected numpy dtype (ORT should already produce correct dtype)
        y = y.astype(np_out_type, copy=False)

        shift = int(
            np.log2(
                float(config_dict["Y_SCALE"])
                / float(config_dict["V_SCALE"] * config_dict["P_SCALE"])
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
            int(np.floor(np.log2(config_dict["DIM_SEQ"]) + 1)) + in_vbits + in_pbits
        )

        for key, value in config_dict.items():
            if key in ["V_SCALE", "P_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {float(value)}f;")
            else:
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                    cwr.add_line(f"const bool {key} = {value_str};")
                else:
                    cwr.add_line(f"const int {key} = {int(value)};")

        typedef_suffix = "u" if inv_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{in_vbits}> TVInput;")
        cwr.add_line(f"using TVInputWord = std::array<TVInput, REDUCE_PAR>;")
        typedef_suffix = "u" if inp_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{in_pbits}> TPInput;")
        cwr.add_line(f"using TPInputWord = std::array<TPInput, REDUCE_PAR>;")

        typedef_suffix = "u" if out_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{out_bits}> TOutput;")
        cwr.add_line(f"using TOutputWord = std::array<TOutput, 1>;")

        cwr.add_line(f"typedef ap_int<{acc_bits}> TAcc;")

        cwr.add_line(f"typedef DequantQuantPo2<{shift}, TAcc, TOutput> Quantizer;")

        cwr.add_lines(
            csnake.Variable(
                "v_tensor",
                primitive="TVInput",
                value=input_tensorv,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "p_tensor",
                primitive="TPInput",
                value=input_tensorp,
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
            "INPUTV_DATAWIDTH": 8,
            "INPUTP_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "INPUTV_IS_UNSIGNED": False,
            "INPUTP_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "DIM_HEADS": 2,
            "DIM_V": 4,
            "DIM_P": 4,
            "DIM_SEQ": 4,
            "REDUCE_PAR": 2,
            "HEADS_PAR": 2,
            "V_SCALE": 2**-4,
            "P_SCALE": 2**-4,
            "Y_SCALE": 2**-1,
            "V_ZP": 0,
            "P_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
