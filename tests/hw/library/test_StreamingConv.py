import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest

class TestStreamingConv(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "StreamingConv"

    @property
    def unit_filename(self) -> str:
        return "StreamingConv"

    def generate_config_file(self, config_dict):
        IN_HEIGHT = (
            (config_dict["OUT_HEIGHT"] - 1) * config_dict["STRIDE_H"]
            + config_dict["DIL_H"] * (config_dict["FH"] - 1)
            + 1
            - config_dict["PAD_T"]
            - config_dict["PAD_B"]
        )
        IN_WIDTH = (
            (config_dict["OUT_WIDTH"] - 1) * config_dict["STRIDE_W"]
            + config_dict["DIL_W"] * (config_dict["FW"] - 1)
            + 1
            - config_dict["PAD_L"]
            - config_dict["PAD_R"]
        )

        # random tensors
        input_tensor = np.random.randint(
            0 if config_dict.get("INPUT_IS_UNSIGNED", False) else -128,
            255 if config_dict.get("INPUT_IS_UNSIGNED", False) else 127,
            size=(1, config_dict["IN_CH"], IN_HEIGHT, IN_WIDTH),
            dtype=self.get_numpy_dtype(
                config_dict["INPUT_DATAWIDTH"],
                config_dict.get("INPUT_IS_UNSIGNED", False),
            ),
        )
        weight_tensor = np.random.randint(
            0 if config_dict.get("WEIGHT_IS_UNSIGNED", False) else -128,
            255 if config_dict.get("WEIGHT_IS_UNSIGNED", False) else 127,
            size=(
                config_dict["OUT_CH"],
                config_dict["IN_CH"],
                config_dict["FH"],
                config_dict["FW"],
            ),
            dtype=self.get_numpy_dtype(
                config_dict["WEIGHT_DATAWIDTH"],
                config_dict.get("WEIGHT_IS_UNSIGNED", False),
            ),
        )
        bias_tensor = np.random.randint(
            0 if config_dict.get("BIAS_IS_UNSIGNED", False) else -10000,
            20000 if not config_dict.get("BIAS_IS_UNSIGNED", False) else 10000,
            size=(config_dict["OUT_CH"],),
            dtype=self.get_numpy_dtype(
                config_dict["BIAS_DATAWIDTH"],
                config_dict.get("BIAS_IS_UNSIGNED", False),
            ),
        )

        # IO
        X = helper.make_tensor_value_info(
            "X",
            self.get_tensorproto_dtype(
                config_dict.get("INPUT_DATAWIDTH", 8),
                config_dict.get("INPUT_IS_UNSIGNED", False),
            ),
            [1, config_dict["IN_CH"], IN_HEIGHT, IN_WIDTH],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            self.get_tensorproto_dtype(
                config_dict.get("OUTPUT_DATAWIDTH", 8),
                config_dict.get("OUTPUT_IS_UNSIGNED", False),
            ),
            [
                1,
                config_dict["OUT_CH"],
                config_dict["OUT_HEIGHT"],
                config_dict["OUT_WIDTH"],
            ],
        )

        # Inits
        W = helper.make_tensor(
            "W",
            self.get_tensorproto_dtype(
                config_dict.get("WEIGHT_DATAWIDTH", 8),
                config_dict.get("WEIGHT_IS_UNSIGNED", False),
            ),
            [
                config_dict["OUT_CH"],
                config_dict["IN_CH"],
                config_dict["FH"],
                config_dict["FW"],
            ],
            weight_tensor.flatten().tolist(),
        )

        B = helper.make_tensor(
            "B",
            self.get_tensorproto_dtype(
                config_dict.get("BIAS_DATAWIDTH", 32),
                config_dict.get("BIAS_IS_UNSIGNED", False),
            ),
            [config_dict["OUT_CH"]],
            bias_tensor.flatten().tolist(),
        )
        X_scale = helper.make_tensor(
            "X_scale", TensorProto.FLOAT, [], [config_dict["X_SCALE"]]
        )
        X_zp = helper.make_tensor(
            "X_zp",
            self.get_tensorproto_dtype(
                config_dict.get("INPUT_DATAWIDTH", 8),
                config_dict.get("INPUT_IS_UNSIGNED", False),
            ),
            [],
            [config_dict["X_ZP"]],
        )
        W_scale = helper.make_tensor(
            "W_scale", TensorProto.FLOAT, [], [config_dict["W_SCALE"]]
        )
        W_zp = helper.make_tensor(
            "W_zp",
            self.get_tensorproto_dtype(
                config_dict.get("WEIGHT_DATAWIDTH", 8),
                config_dict.get("WEIGHT_IS_UNSIGNED", False),
            ),
            [],
            [config_dict["W_ZP"]],
        )
        Y_scale = helper.make_tensor(
            "Y_scale", TensorProto.FLOAT, [], [config_dict["Y_SCALE"]]
        )
        Y_zp = helper.make_tensor(
            "Y_zp",
            self.get_tensorproto_dtype(
                config_dict.get("OUTPUT_DATAWIDTH", 8),
                config_dict.get("OUTPUT_IS_UNSIGNED", False),
            ),
            [],
            [config_dict["Y_ZP"]],
        )

        qconv = helper.make_node(
            "QLinearConv",
            inputs=[
                "X",
                "X_scale",
                "X_zp",
                "W",
                "W_scale",
                "W_zp",
                "Y_scale",
                "Y_zp",
                "B",
            ],
            outputs=["Y"],
            strides=[config_dict["STRIDE_H"], config_dict["STRIDE_W"]],
            pads=[
                config_dict["PAD_T"],
                config_dict["PAD_B"],
                config_dict["PAD_L"],
                config_dict["PAD_R"],
            ],
            dilations=[config_dict["DIL_H"], config_dict["DIL_W"]],
            group=config_dict["GROUP"],
            kernel_shape=[config_dict["FH"], config_dict["FW"]],
        )

        graph = helper.make_graph(
            [qconv],
            "qconv_test",
            [X],
            [Y],
            initializer=[W, B, X_scale, X_zp, W_scale, W_zp, Y_scale, Y_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        y = sess.run(None, {"X": input_tensor})[0]

        shift = int(np.log2(config_dict["Y_SCALE"] / (config_dict["X_SCALE"] * config_dict["W_SCALE"])))

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
                    cwr.add_line(f"const int {key} = {value};")
        typedef_suffix = "u" if config_dict.get("INPUT_IS_UNSIGNED", False) else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['INPUT_DATAWIDTH']}> TInput;")
        typedef_suffix = "u" if config_dict.get("WEIGHT_IS_UNSIGNED", False) else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['WEIGHT_DATAWIDTH']}> TWeight;")
        typedef_suffix = "u" if config_dict.get("BIAS_IS_UNSIGNED", False) else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['BIAS_DATAWIDTH']}> TBias;")
        typedef_suffix = "u" if config_dict.get("OUTPUT_IS_UNSIGNED", False) else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['OUTPUT_DATAWIDTH']}> TOutput;")
        typedef_suffix = (
            "u"
            if (
                config_dict.get("INPUT_IS_UNSIGNED", False)
                and config_dict.get("WEIGHT_IS_UNSIGNED", False)
                and config_dict.get("BIAS_IS_UNSIGNED", False)
            )
            else ""
        )
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['ACC_DATAWIDTH']}> TAcc;")
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['ACC_DATAWIDTH'] - 1}> TPartialAcc;")
        cwr.add_line(f"typedef DequantQuantPo2<{shift}, TAcc, TOutput> Quantizer;")
        cwr.add_line(f"typedef DequantQuantEqual<TAcc> Activation;")
        cwr.add_lines(
            csnake.Variable(
                "input_tensor",
                primitive=f"TInput",
                value=input_tensor,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "weight_tensor",
                primitive=f"TWeight",
                value=weight_tensor,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "bias_tensor",
                primitive=f"TBias",
                value=bias_tensor,
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

    def test_pointwise_pertensor_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "INPUT_DATAWIDTH": 8,
            "WEIGHT_DATAWIDTH": 8,
            "BIAS_DATAWIDTH": 32,
            "OUTPUT_DATAWIDTH": 8,
            "OUT_HEIGHT": 4,
            "OUT_WIDTH": 4,
            "IN_CH": 4,
            "OUT_CH": 4,
            "FH": 1,
            "FW": 1,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PAD_T": 0,
            "PAD_B": 0,
            "PAD_L": 0,
            "PAD_R": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "IN_CH_PAR": 2,
            "OUT_CH_PAR": 2,
            "W_PAR": 1,
            "GROUP": 1,
            "X_SCALE": 2**-5,
            "W_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "X_ZP": 0,
            "W_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_pointwise_pertensor_po2_stride(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "INPUT_DATAWIDTH": 8,
            "WEIGHT_DATAWIDTH": 8,
            "BIAS_DATAWIDTH": 32,
            "OUTPUT_DATAWIDTH": 8,
            "OUT_HEIGHT": 2,
            "OUT_WIDTH": 2,
            "IN_CH": 4,
            "OUT_CH": 4,
            "FH": 1,
            "FW": 1,
            "STRIDE_H": 2,
            "STRIDE_W": 2,
            "PAD_T": 0,
            "PAD_B": 0,
            "PAD_L": 0,
            "PAD_R": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "IN_CH_PAR": 2,
            "OUT_CH_PAR": 2,
            "W_PAR": 2,
            "GROUP": 1,
            "X_SCALE": 2**-5,
            "W_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "X_ZP": 0,
            "W_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 3,
        }
        self.run(config_dict, hls_steps)

    def test_3x3_pertensor_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "INPUT_DATAWIDTH": 8,
            "WEIGHT_DATAWIDTH": 8,
            "BIAS_DATAWIDTH": 32,
            "OUTPUT_DATAWIDTH": 8,
            "OUT_HEIGHT": 4,
            "OUT_WIDTH": 4,
            "IN_CH": 4,
            "OUT_CH": 4,
            "FH": 3,
            "FW": 3,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PAD_T": 1,
            "PAD_B": 1,
            "PAD_L": 1,
            "PAD_R": 1,
            "DIL_H": 1,
            "DIL_W": 1,
            "IN_CH_PAR": 2,
            "OUT_CH_PAR": 2,
            "W_PAR": 1,
            "GROUP": 1,
            "X_SCALE": 2**-5,
            "W_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "X_ZP": 0,
            "W_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 1,
        }
        self.run(config_dict, hls_steps)

    def test_1x5_pertensor_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "INPUT_DATAWIDTH": 8,
            "WEIGHT_DATAWIDTH": 8,
            "BIAS_DATAWIDTH": 32,
            "OUTPUT_DATAWIDTH": 8,
            "OUT_HEIGHT": 2,
            "OUT_WIDTH": 2,
            "IN_CH": 4,
            "OUT_CH": 4,
            "FH": 1,
            "FW": 5,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PAD_T": 0,
            "PAD_B": 0,
            "PAD_L": 0,
            "PAD_R": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "IN_CH_PAR": 2,
            "OUT_CH_PAR": 2,
            "W_PAR": 2,
            "GROUP": 1,
            "X_SCALE": 2**-5,
            "W_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "X_ZP": 0,
            "W_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 2,
        }
        self.run(config_dict, hls_steps)

    def test_fullyconnected_pertensor_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "ACC_DATAWIDTH": 32,
            "INPUT_DATAWIDTH": 8,
            "WEIGHT_DATAWIDTH": 8,
            "BIAS_DATAWIDTH": 32,
            "OUTPUT_DATAWIDTH": 8,
            "OUT_HEIGHT": 1,
            "OUT_WIDTH": 1,
            "IN_CH": 4,
            "OUT_CH": 4,
            "FH": 1,
            "FW": 1,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PAD_T": 0,
            "PAD_B": 0,
            "PAD_L": 0,
            "PAD_R": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "IN_CH_PAR": 4,
            "OUT_CH_PAR": 2,
            "W_PAR": 1,
            "GROUP": 1,
            "X_SCALE": 2**-5,
            "W_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "X_ZP": 0,
            "W_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 3,
        }
        self.run(config_dict, hls_steps)

    def test_pointwise_pertensor_chpar12_wpar4(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "ACC_DATAWIDTH": 33,
            "INPUT_DATAWIDTH": 8,
            "WEIGHT_DATAWIDTH": 8,
            "BIAS_DATAWIDTH": 32,
            "OUTPUT_DATAWIDTH": 8,
            "OUT_HEIGHT": 14,
            "OUT_WIDTH": 14,
            "IN_CH": 384,
            "OUT_CH": 96,
            "FH": 1,
            "FW": 1,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PAD_T": 0,
            "PAD_B": 0,
            "PAD_L": 0,
            "PAD_R": 0,
            "DIL_H": 1,
            "DIL_W": 1,
            "IN_CH_PAR": 1,
            "OUT_CH_PAR": 96,
            "W_PAR": 1,
            "GROUP": 1,
            "X_SCALE": 2**-8,
            "W_SCALE": 2**-8,
            "Y_SCALE": 2**-6,
            "X_ZP": 0,
            "W_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_pointwise_pertensor_uint(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "ACC_DATAWIDTH": 25,
            "INPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": True,
            "WEIGHT_DATAWIDTH": 8,
            "WEIGHT_IS_UNSIGNED": False,
            "BIAS_DATAWIDTH": 32,
            "BIAS_IS_UNSIGNED": False,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": True,
            "OUT_HEIGHT": 16,
            "OUT_WIDTH": 16,
            "IN_CH": 16,
            "OUT_CH": 32,
            "FH": 3,
            "FW": 3,
            "STRIDE_H": 2,
            "STRIDE_W": 2,
            "PAD_T": 1,
            "PAD_B": 1,
            "PAD_L": 1,
            "PAD_R": 1,
            "DIL_H": 1,
            "DIL_W": 1,
            "IN_CH_PAR": 4,
            "OUT_CH_PAR": 4,
            "W_PAR": 1,
            "GROUP": 1,
            "X_SCALE": 2**-5,
            "W_SCALE": 2**-8,
            "Y_SCALE": 2**-5,
            "X_ZP": 0,
            "W_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
