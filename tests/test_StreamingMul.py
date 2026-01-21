import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest

class TestStreamingMul(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "StreamingMul"

    @property
    def unit_filename(self) -> str:
        return "StreamingMul"

    def generate_config_file(self, config_dict):

        onnx_ina_type = self.get_tensorproto_dtype(
            config_dict["INPUTA_DATAWIDTH"],
            config_dict.get("INPUTA_IS_UNSIGNED", False),
        )
        onnx_inb_type = self.get_tensorproto_dtype(
            config_dict["INPUTB_DATAWIDTH"],
            config_dict.get("INPUTB_IS_UNSIGNED", False),
        )
        onnx_out_type = self.get_tensorproto_dtype(
            config_dict["OUTPUT_DATAWIDTH"],
            config_dict.get("OUTPUT_IS_UNSIGNED", False),
        )

        # random tensors
        np_ina_type = self.get_numpy_dtype(
            config_dict["INPUTA_DATAWIDTH"],
            config_dict.get("INPUTA_IS_UNSIGNED", False),
        )
        ina_info = np.iinfo(np_ina_type)
        input_tensorA = np.random.randint(
            int(ina_info.min),
            int(ina_info.max) + 1,
            size=(
                1,
                config_dict["IN_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ),
            dtype=np_ina_type,
        )

        np_inb_type = self.get_numpy_dtype(
            config_dict["INPUTB_DATAWIDTH"],
            config_dict.get("INPUTB_IS_UNSIGNED", False),
        )
        inb_info = np.iinfo(np_inb_type)

        input_tensorB = np.random.randint(
            int(inb_info.min),
            int(inb_info.max) + 1,
            size=(
                1,
                config_dict["IN_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ),
            dtype=np_inb_type,
        )

        # I/O
        A = helper.make_tensor_value_info(
            "A",
            onnx_ina_type,
            [
                1,
                config_dict["IN_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ],
        )
        B = helper.make_tensor_value_info(
            "B",
            onnx_inb_type,
            [
                1,
                config_dict["IN_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_out_type,
            [
                1,
                config_dict["IN_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
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

        mul = helper.make_node(
            "Mul",
            inputs=[
                "A_dequant",
                "B_dequant",
            ],
            outputs=["Y_dequant"],
        )

        graph = helper.make_graph(
            [dequant0, dequant1, mul, quant],
            "mul_test",
            [A, B],
            [Y],
            initializer=[A_scale, B_scale, A_zp, B_zp, Y_scale, Y_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        y = sess.run(None, {"A": input_tensorA, "B": input_tensorB})[0]

        shift = int(np.log2(config_dict["Y_SCALE"] / (config_dict["A_SCALE"] * config_dict["B_SCALE"])))
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
                    cwr.add_line(f"const int {key} = {value};")
        typedef_suffix = "u" if config_dict.get("INPUTA_IS_UNSIGNED", False) else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['INPUTA_DATAWIDTH']}> TInputA;")
        typedef_suffix = "u" if config_dict.get("INPUTB_IS_UNSIGNED", False) else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['INPUTB_DATAWIDTH']}> TInputB;")
        typedef_suffix = (
            "u"
            if (
                config_dict.get("INPUTA_IS_UNSIGNED", False)
                and config_dict.get("INPUTB_IS_UNSIGNED", False)
            )
            else ""
        )
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['MUL_DATAWIDTH']}> TMul;")
        typedef_suffix = "u" if config_dict.get("OUTPUT_IS_UNSIGNED", False) else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['OUTPUT_DATAWIDTH']}> TOutput;")
        cwr.add_line(f"typedef DequantQuantPo2<{shift}, TMul, TOutput> Quantizer;")
        cwr.add_line("typedef DequantQuantEqual<TMul> Activation;")
        # cwr.add_line(f"typedef DequantQuantEqual<TAcc> Activation;")
        cwr.add_lines(
            csnake.Variable(
                "input_tensorA",
                primitive=f"TInputA",
                value=input_tensorA,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "input_tensorB",
                primitive=f"TInputB",
                value=input_tensorB,
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
            "MUL_DATAWIDTH": 16,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 4,
            "IN_CH": 4,
            "CH_PAR": 2,
            "W_PAR": 1,
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
            "MUL_DATAWIDTH": 16,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 4,
            "IN_CH": 4,
            "CH_PAR": 2,
            "W_PAR": 1,
            "A_SCALE": 2**-5,
            "B_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_pertensor_po2_mixed(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTA_IS_UNSIGNED": False,
            "INPUTB_IS_UNSIGNED": True,
            "OUTPUT_IS_UNSIGNED": True,
            "INPUTA_DATAWIDTH": 8,
            "INPUTB_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "MUL_DATAWIDTH": 17,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 4,
            "IN_CH": 4,
            "CH_PAR": 2,
            "W_PAR": 1,
            "A_SCALE": 2**-5,
            "B_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
