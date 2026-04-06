import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from tests.base_hls_test import BaseHLSTest

class TestSplitReshape(BaseHLSTest):

    @property
    def operator_filename(self):
        return "YoloAttention/ReshapeV"

    @property
    def unit_filename(self):
        return "YoloAttention/ReshapeV"

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
        out_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False))

        in_bits = int(config_dict["INPUT_DATAWIDTH"])
        out_bits = int(config_dict["OUTPUT_DATAWIDTH"])

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
                config_dict["IN_HEADS"],
                config_dict["IN_DIM"],
                config_dict["IN_SEQ"],
            ),
            dtype=np_in_type,
        )

        X = helper.make_tensor_value_info(
            "X",
            onnx_in_type,
            [1, config_dict["IN_HEADS"], config_dict["IN_DIM"], config_dict["IN_SEQ"]],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_out_type,
            [1, config_dict["OUT_CH"], config_dict["OUT_HEIGHT"], config_dict["OUT_WIDTH"]],
        )

        X_scale = helper.make_tensor("X_scale", TensorProto.FLOAT, [], [float(config_dict["X_SCALE"])])
        X_zp = helper.make_tensor("X_zp", onnx_in_type, [], [int(config_dict["X_ZP"])])
        Y_scale = helper.make_tensor("Y_scale", TensorProto.FLOAT, [], [float(config_dict["Y_SCALE"])])
        Y_zp = helper.make_tensor("Y_zp", onnx_out_type, [], [int(config_dict["Y_ZP"])])

        dqlinear = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "X_scale", "X_zp"],
            outputs=["X_dq"],
        )

        shape = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["shape"],
            value=helper.make_tensor(
                name="const_shape",
                data_type=TensorProto.INT64,
                dims=[4],
                vals=[
                    1,
                    config_dict["OUT_CH"],
                    config_dict["OUT_HEIGHT"],
                    config_dict["OUT_WIDTH"],
                ],
            ),
        )
        reshape = helper.make_node(
            "Reshape",
            inputs=["X_dq", "shape"],
            outputs=["X_reshaped"],
        )


        qlinearv = helper.make_node(
            "QuantizeLinear",
            inputs=["X_reshaped", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [
                shape,
                dqlinear,
                reshape,
                qlinearv,
            ],
            "qsplit_test",
            [X],
            [Y],
            initializer=[X_scale, X_zp, Y_scale, Y_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        y = sess.run(None, {"X": input_tensor})[0]

        # Optional: cast ORT output to expected numpy dtype (ORT should already produce correct dtype)
        y = y.astype(np_out_type, copy=False)

        shift = int(np.log2(float(config_dict["Y_SCALE"]) / float(config_dict["X_SCALE"])))
        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.include("DequantQuant.hpp")
        cwr.add_line("namespace test_config {")
        cwr.indent()

        for key, value in config_dict.items():
            if key in ["X_SCALE"]:
                cwr.add_line(f"const float {key} = {float(value)}f;")
            else:
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                    cwr.add_line(f"const bool {key} = {value_str};")
                else:
                    cwr.add_line(f"const int {key} = {int(value)};")

        typedef_suffix = "u" if in_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{in_bits}> TInput;")
        cwr.add_line(f"using TInputWord = std::array<TInput, REDUCE_PAR>;")

        typedef_suffix = "u" if out_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{out_bits}> TOutput;")
        cwr.add_line(f"using TOutputWord = std::array<TOutput, REDUCE_PAR>;")

        cwr.add_line(f"typedef DequantQuantPo2<{shift}, TInput, TOutput> Quantizer;")

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

        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_yolov10n(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_HEADS": 2,
            "IN_DIM": 64,
            "IN_SEQ": 400,
            "OUT_HEIGHT": 20,
            "OUT_WIDTH": 20,
            "OUT_CH": 128,
            "REDUCE_PAR": 2,
            "X_SCALE": 2**-2,
            "Y_SCALE": 2**-2,
            "X_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
