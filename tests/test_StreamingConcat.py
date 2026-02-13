import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest

class TestStreamingConcat(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "StreamingConcat"

    @property
    def unit_filename(self) -> str:
        return "StreamingConcat"

    def generate_config_file(self, config_dict, class_name: str = "StreamingConcatHeight"):

        # random tensors
        input_tensor0 = np.random.randint(
            -128,
            127,
            size=(1, config_dict["IN_CH_A"], config_dict["IN_HEIGHT_A"], config_dict["IN_WIDTH_A"]),
            dtype=np.int8,
        )
        input_tensor1 = np.random.randint(
            -128,
            127,
            size=(1, config_dict["IN_CH_B"], config_dict["IN_HEIGHT_B"], config_dict["IN_WIDTH_B"]),
            dtype=np.int8,
        )

        output_shape = [
            1,
            (
                config_dict["IN_CH_A"] + config_dict["IN_CH_B"]
                if config_dict["CONCAT_AXIS"] == 1
                else config_dict["IN_CH_A"]
            ),
            (
                config_dict["IN_HEIGHT_A"] + config_dict["IN_HEIGHT_B"]
                if config_dict["CONCAT_AXIS"] == 2
                else config_dict["IN_HEIGHT_A"]
            ),
            (
                config_dict["IN_WIDTH_A"] + config_dict["IN_WIDTH_B"]
                if config_dict["CONCAT_AXIS"] == 3
                else config_dict["IN_WIDTH_A"]
            ),
        ]

        # I/O
        X0 = helper.make_tensor_value_info(
            "X0", TensorProto.INT8, [1, config_dict["IN_CH_A"], config_dict["IN_HEIGHT_A"], config_dict["IN_WIDTH_A"]]
        )
        X1 = helper.make_tensor_value_info(
            "X1", TensorProto.INT8, [1, config_dict["IN_CH_B"], config_dict["IN_HEIGHT_B"], config_dict["IN_WIDTH_B"]]
        )
        Y = helper.make_tensor_value_info(
            "Y",
            TensorProto.INT8,
            output_shape,
        )

        I_scale = helper.make_tensor(
            "I_scale", TensorProto.FLOAT, [], [config_dict["I_SCALE"]]
        )
        I_zp = helper.make_tensor("I_zp", TensorProto.INT8, [], [config_dict["I_ZP"]])
        Y_scale = helper.make_tensor(
            "Y_scale", TensorProto.FLOAT, [], [config_dict["Y_SCALE"]]
        )
        Y_zp = helper.make_tensor("Y_zp", TensorProto.INT8, [], [config_dict["Y_ZP"]])

        dequant0 = helper.make_node(
            "DequantizeLinear",
            inputs=["X0", "I_scale", "I_zp"],
            outputs=["X0_dequant"],
        )
        dequant1 = helper.make_node(
            "DequantizeLinear",
            inputs=["X1", "I_scale", "I_zp"],
            outputs=["X1_dequant"],
        )
        quant = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dequant", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        concat = helper.make_node(
            "Concat",
            inputs=[
                "X0_dequant",
                "X1_dequant",
            ],
            outputs=["Y_dequant"],
            axis=config_dict["CONCAT_AXIS"],
        )

        graph = helper.make_graph(
            [dequant0, dequant1, concat, quant],
            "concat_test",
            [X0, X1],
            [Y],
            initializer=[I_scale, I_zp, Y_scale, Y_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        y = sess.run(None, {"X0": input_tensor0, "X1": input_tensor1})[0]

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            if key in ["I_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {value}f;")
            else:
                cwr.add_line(f"const int {key} = {value};")
        cwr.add_line(f"const size_t OUT_HEIGHT = {output_shape[2]};")
        cwr.add_line(f"const size_t OUT_WIDTH = {output_shape[3]};")
        cwr.add_line(f"const size_t OUT_CH = {output_shape[1]};")
        cwr.add_line(f"typedef ap_int<{config_dict['INPUT_DATAWIDTH']}> TInput;")
        cwr.add_line(f"typedef ap_int<{config_dict['OUTPUT_DATAWIDTH']}> TOutput;")
        cwr.add_line(f"typedef DequantQuantPo2<0, TInput, TOutput> Quantizer;")
        cwr.add_line(
            f"using TInputWord = std::array<TInput, {config_dict['CH_PAR']}>;"
        )
        cwr.add_line(
            f"using TOutputWord = std::array<TOutput, {config_dict['CH_PAR']}>;"
        )
        if config_dict["CONCAT_AXIS"] == 1:
            cwr.add_line(
                f"using StreamingConcat = {class_name}<TInputWord, TInput, TOutputWord, TOutput, Quantizer, IN_HEIGHT_A, IN_WIDTH_A, IN_CH_A, IN_CH_B, W_PAR, CH_PAR>;"
            )
        elif config_dict["CONCAT_AXIS"] == 2:
            cwr.add_line(
                f"using StreamingConcat = {class_name}<TInputWord, TInput, TOutputWord, TOutput, Quantizer, IN_HEIGHT_A, IN_HEIGHT_B, IN_WIDTH_A, IN_CH_A, W_PAR, CH_PAR>;"
            )
        elif config_dict["CONCAT_AXIS"] == 3:
            cwr.add_line(
                f"using StreamingConcat = {class_name}<TInputWord, TInput, TOutputWord, TOutput, Quantizer, IN_HEIGHT_A, IN_WIDTH_A, IN_WIDTH_B, IN_CH_A, W_PAR, CH_PAR>;"
            )
        cwr.add_lines(
            csnake.Variable(
                "input_tensor0",
                primitive=f"ap_int<{config_dict['INPUT_DATAWIDTH']}>",
                value=input_tensor0,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "input_tensor1",
                primitive=f"ap_int<{config_dict['INPUT_DATAWIDTH']}>",
                value=input_tensor1,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "output_tensor",
                primitive=f"ap_int<{config_dict['OUTPUT_DATAWIDTH']}>",
                value=y,
            ).generate_initialization()
        )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_channelconcat_pertensor_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "IN_HEIGHT_A": 4,
            "IN_HEIGHT_B": 4,
            "IN_WIDTH_A": 4,
            "IN_WIDTH_B": 4,
            "IN_CH_A": 4,
            "IN_CH_B": 8,
            "CH_PAR": 2,
            "W_PAR": 1,
            "CONCAT_AXIS": 1,
            "I_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "I_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict, hls_steps, workdir=".", class_name="StreamingConcatChannel"
        )
    
    def test_heightconcat_pertensor_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "IN_HEIGHT_A": 4,
            "IN_HEIGHT_B": 8,
            "IN_WIDTH_A": 4,
            "IN_WIDTH_B": 4,
            "IN_CH_A": 4,
            "IN_CH_B": 4,
            "CH_PAR": 2,
            "W_PAR": 1,
            "CONCAT_AXIS": 2,
            "I_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "I_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict, hls_steps, workdir=".", class_name="StreamingConcatHeight"
        )
    
    def test_widthconcat_pertensor_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "IN_HEIGHT_A": 4,
            "IN_HEIGHT_B": 4,
            "IN_WIDTH_A": 4,
            "IN_WIDTH_B": 8,
            "IN_CH_A": 4,
            "IN_CH_B": 4,
            "CH_PAR": 2,
            "W_PAR": 1,
            "CONCAT_AXIS": 3,
            "I_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "I_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict, hls_steps, workdir=".", class_name="StreamingConcatWidth"
        )
