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
        in_unsigned = bool(config_dict.get("INPUT_IS_UNSIGNED", False))
        out_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False))

        in_bits = int(config_dict["INPUT_DATAWIDTH"])
        out_bits = int(config_dict["OUTPUT_DATAWIDTH"])

        onnx_in_type = self.get_tensorproto_dtype(in_bits, in_unsigned)
        onnx_out_type = self.get_tensorproto_dtype(out_bits, out_unsigned)
        np_in_type = self.get_numpy_dtype(in_bits, in_unsigned)

        # random tensors
        in_info = np.iinfo(np_in_type)
        input_tensor0 = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,
            size=(
                1,
                config_dict["IN_DIM2_A"],
                config_dict["IN_DIM0_A"],
                config_dict["IN_DIM1_A"],
            ),
            dtype=np_in_type,
        )
        input_tensor1 = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,
            size=(
                1,
                config_dict["IN_DIM2_B"],
                config_dict["IN_DIM0_B"],
                config_dict["IN_DIM1_B"],
            ),
            dtype=np_in_type,
        )

        output_shape = [
            1,
            (
                config_dict["IN_DIM2_A"] + config_dict["IN_DIM2_B"]
                if config_dict["CONCAT_AXIS"] == 1
                else config_dict["IN_DIM2_A"]
            ),
            (
                config_dict["IN_DIM0_A"] + config_dict["IN_DIM0_B"]
                if config_dict["CONCAT_AXIS"] == 2
                else config_dict["IN_DIM0_A"]
            ),
            (
                config_dict["IN_DIM1_A"] + config_dict["IN_DIM1_B"]
                if config_dict["CONCAT_AXIS"] == 3
                else config_dict["IN_DIM1_A"]
            ),
        ]

        # I/O
        X0 = helper.make_tensor_value_info(
            "X0",
            onnx_in_type,
            [
                1,
                config_dict["IN_DIM2_A"],
                config_dict["IN_DIM0_A"],
                config_dict["IN_DIM1_A"],
            ],
        )
        X1 = helper.make_tensor_value_info(
            "X1",
            onnx_in_type,
            [
                1,
                config_dict["IN_DIM2_B"],
                config_dict["IN_DIM0_B"],
                config_dict["IN_DIM1_B"],
            ],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_out_type,
            output_shape,
        )

        I_scale = helper.make_tensor(
            "I_scale", TensorProto.FLOAT, [], [config_dict["X_SCALE"]]
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
            if key in ["X_SCALE", "W_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {value}f;")
            else:
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                    cwr.add_line(f"const bool {key} = {value_str};")
                else:
                    cwr.add_line(f"const int {key} = {int(value)};")
        cwr.add_line(f"const size_t OUT_DIM0 = {output_shape[2]};")
        cwr.add_line(f"const size_t OUT_DIM1 = {output_shape[3]};")
        cwr.add_line(f"const size_t OUT_DIM2 = {output_shape[1]};")
        typedef_suffix = "u" if in_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['INPUT_DATAWIDTH']}> TInput;")
        typedef_suffix = "u" if out_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{config_dict['OUTPUT_DATAWIDTH']}> TOutput;")
        cwr.add_line(f"typedef DequantQuantPo2<0, TInput, TOutput> Quantizer;")
        cwr.add_line(
            f"using TInputWord = std::array<TInput, {config_dict['DIM2_UNROLL']}>;"
        )
        cwr.add_line(
            f"using TOutputWord = std::array<TOutput, {config_dict['DIM2_UNROLL']}>;"
        )
        if config_dict["CONCAT_AXIS"] == 1:
            cwr.add_line(
                f"using StreamingConcat = {class_name}<TInputWord, TInput, TOutputWord, TOutput, Quantizer, IN_DIM0_A, IN_DIM1_A, IN_DIM2_A, IN_DIM2_B, DIM1_UNROLL, DIM2_UNROLL>;"
            )
        elif config_dict["CONCAT_AXIS"] == 2:
            cwr.add_line(
                f"using StreamingConcat = {class_name}<TInputWord, TInput, TOutputWord, TOutput, Quantizer, IN_DIM0_A, IN_DIM0_B, IN_DIM1_A, IN_DIM2_A, DIM1_UNROLL, DIM2_UNROLL>;"
            )
        elif config_dict["CONCAT_AXIS"] == 3:
            cwr.add_line(
                f"using StreamingConcat = {class_name}<TInputWord, TInput, TOutputWord, TOutput, Quantizer, IN_DIM0_A, IN_DIM1_A, IN_DIM1_B, IN_DIM2_A, DIM1_UNROLL, DIM2_UNROLL>;"
            )
        cwr.add_lines(
            csnake.Variable(
                "input_tensor0",
                primitive=f"TInput",
                value=input_tensor0,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "input_tensor1",
                primitive=f"TInput",
                value=input_tensor1,
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

    def test_channelconcat_pertensor_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_DIM0_A": 4,
            "IN_DIM0_B": 4,
            "IN_DIM1_A": 4,
            "IN_DIM1_B": 4,
            "IN_DIM2_A": 4,
            "IN_DIM2_B": 8,
            "DIM2_UNROLL": 2,
            "DIM1_UNROLL": 1,
            "CONCAT_AXIS": 1,
            "X_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "I_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict, hls_steps, workdir=".", class_name="StreamingConcatDim2"
        )

    def test_heightconcat_pertensor_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_DIM0_A": 4,
            "IN_DIM0_B": 8,
            "IN_DIM1_A": 4,
            "IN_DIM1_B": 4,
            "IN_DIM2_A": 4,
            "IN_DIM2_B": 4,
            "DIM2_UNROLL": 2,
            "DIM1_UNROLL": 1,
            "CONCAT_AXIS": 2,
            "X_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "I_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict, hls_steps, workdir=".", class_name="StreamingConcatDim0"
        )

    def test_widthconcat_pertensor_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_DATAWIDTH": 8,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_DIM0_A": 4,
            "IN_DIM0_B": 4,
            "IN_DIM1_A": 4,
            "IN_DIM1_B": 8,
            "IN_DIM2_A": 4,
            "IN_DIM2_B": 4,
            "DIM2_UNROLL": 2,
            "DIM1_UNROLL": 1,
            "CONCAT_AXIS": 3,
            "X_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "I_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict, hls_steps, workdir=".", class_name="StreamingConcatDim1"
        )
