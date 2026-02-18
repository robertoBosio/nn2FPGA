import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest


class TestStreamingAveragePool(BaseHLSTest):

    @property
    def operator_filename(self):
        return "StreamingAveragePool"
    @property
    def unit_filename(self):
        return "StreamingAveragePool"

    def generate_config_file(self, config_dict):
        in_unsigned = bool(config_dict.get("INPUT_IS_UNSIGNED", False))
        out_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False))

        in_bits = int(config_dict["INPUT_DATAWIDTH"])
        out_bits = int(config_dict["OUTPUT_DATAWIDTH"])

        onnx_in_type = self.get_tensorproto_dtype(in_bits, in_unsigned)
        onnx_out_type = self.get_tensorproto_dtype(out_bits, out_unsigned)
        np_in_type = self.get_numpy_dtype(in_bits, in_unsigned)
        np_out_type = self.get_numpy_dtype(out_bits, out_unsigned)

        # random tensors
        in_info = np.iinfo(np_in_type)
        input_tensor = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,
            size=(
                1,
                config_dict["OUT_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ),
            dtype=np_in_type,
        )

        X = helper.make_tensor_value_info(
            "X",
            onnx_in_type,
            [
                1,
                config_dict["OUT_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_out_type,
            [
                1,
                config_dict["OUT_CH"],
                1,
                1,
            ],
        )

        X_scale = helper.make_tensor(
            "X_scale", TensorProto.FLOAT, [], [config_dict["X_SCALE"]]
        )
        X_zp = helper.make_tensor("X_zp", onnx_in_type, [], [config_dict["X_ZP"]])
        Y_scale = helper.make_tensor(
            "Y_scale", TensorProto.FLOAT, [], [config_dict["Y_SCALE"]]
        )
        Y_zp = helper.make_tensor("Y_zp", onnx_out_type, [], [config_dict["Y_ZP"]])

        dqlinear = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "X_scale", "X_zp"],
            outputs=["X_dq"],
        )

        avg_pool = helper.make_node(
            "AveragePool",
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
            ceil_mode=config_dict["CEIL_MODE"],
            count_include_pad=1,
        )

        qlinear = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dq", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [dqlinear, avg_pool, qlinear],
            "qconv_test",
            [X],
            [Y],
            initializer=[X_scale, X_zp, Y_scale, Y_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        y = sess.run(None, {"X": input_tensor})[0]

        shift = int(np.log2(config_dict["Y_SCALE"] / config_dict["X_SCALE"]))
        tacc_bits = int(in_bits + int(np.ceil(np.log2(config_dict["FH"] * config_dict["FW"]))))
        tdiv_bits = int(np.ceil(np.log2((config_dict["FH"] * config_dict["FW"]) + 1)))
        config_dict["ACC_DATAWIDTH"] = tacc_bits
        config_dict["DIV_DATAWIDTH"] = tdiv_bits

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            if key in ["X_SCALE", "W_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {value}f;")
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
        typedef_suffix = "u" if in_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{tacc_bits}> TAcc;")
        cwr.add_line(f"typedef ap_uint<{tdiv_bits}> TDiv;")
        cwr.add_line(f"typedef DequantQuantPo2<{shift}, TAcc, TOutput> Quantizer;")
        cwr.add_lines(
            csnake.Variable(
                "input_tensor",
                primitive=f"TInput",
                value=input_tensor,
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

    def test_3x3_signed_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_HEIGHT": 20,
            "IN_WIDTH": 20,
            "OUT_CH": 20,
            "CH_PAR": 1,
            "W_PAR": 1,
            "FH": 3,
            "FW": 3,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
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

    def test_2x2_nopad_signed_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_HEIGHT": 20,
            "IN_WIDTH": 20,
            "OUT_CH": 20,
            "CH_PAR": 1,
            "W_PAR": 1,
            "FH": 2,
            "FW": 2,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "DIL_H": 1,
            "DIL_W": 1,
            "PAD_T": 0,
            "PAD_B": 0,
            "PAD_L": 0,
            "PAD_R": 0,
            "CEIL_MODE": 0,
            "X_SCALE": 2**-5,
            "X_ZP": 0,
            "Y_SCALE": 2**-5,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_7x7_nopad_mixed_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": True,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_HEIGHT": 7,
            "IN_WIDTH": 7,
            "OUT_CH": 20,
            "CH_PAR": 1,
            "W_PAR": 1,
            "FH": 7,
            "FW": 7,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "DIL_H": 1,
            "DIL_W": 1,
            "PAD_T": 0,
            "PAD_B": 0,
            "PAD_L": 0,
            "PAD_R": 0,
            "CEIL_MODE": 0,
            "X_SCALE": 2**-5,
            "X_ZP": 0,
            "Y_SCALE": 2**-5,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)