import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest

class TestStreamingSplit(BaseHLSTest):

    @property
    def operator_filename(self):
        return "StreamingSplit"

    @property
    def unit_filename(self):
        return "StreamingSplit"

    def generate_config_file(self, config_dict):
        """
        Generate a self-contained C++ test config (input/output tensors and operator typedefs),
        supporting signed/unsigned input and signed/unsigned output independently.

        Expected config_dict keys (in addition to your existing ones):
        INPUT_IS_UNSIGNED (bool, optional; default False)
        OUTPUT_IS_UNSIGNED (bool, optional; default False)
        INPUT_DATAWIDTH / OUTPUT_DATAWIDTH (int)
        """

        split_array = []
        if config_dict['SPLIT_AXIS'] == 1:  # Channel axis
            config_dict['OUT_CH0'] = config_dict['SPLIT_POINT']
            config_dict['OUT_CH1'] = config_dict['IN_CH'] - config_dict['SPLIT_POINT']
            config_dict['OUT_HEIGHT0'] = config_dict['IN_HEIGHT']
            config_dict['OUT_HEIGHT1'] = config_dict['IN_HEIGHT']
            config_dict['OUT_WIDTH0'] = config_dict['IN_WIDTH']
            config_dict['OUT_WIDTH1'] = config_dict['IN_WIDTH']
            split_array = [config_dict['OUT_CH0'], config_dict['OUT_CH1']]
            class_name = "StreamingSplitChannels"
        elif config_dict['SPLIT_AXIS'] == 2:  # Height axis
            config_dict['OUT_HEIGHT0'] = config_dict['SPLIT_POINT']
            config_dict['OUT_HEIGHT1'] = config_dict['IN_HEIGHT'] - config_dict['SPLIT_POINT']
            config_dict['OUT_CH0'] = config_dict['IN_CH']
            config_dict['OUT_CH1'] = config_dict['IN_CH']
            config_dict['OUT_WIDTH0'] = config_dict['IN_WIDTH']
            config_dict['OUT_WIDTH1'] = config_dict['IN_WIDTH']
            split_array = [config_dict['OUT_HEIGHT0'], config_dict['OUT_HEIGHT1']]
            class_name = "StreamingSplitHeights"
        elif config_dict['SPLIT_AXIS'] == 3:  # Width axis
            config_dict['OUT_WIDTH0'] = config_dict['SPLIT_POINT']
            config_dict['OUT_WIDTH1'] = config_dict['IN_WIDTH'] - config_dict['SPLIT_POINT']
            config_dict['OUT_CH0'] = config_dict['IN_CH']
            config_dict['OUT_CH1'] = config_dict['IN_CH']
            config_dict['OUT_HEIGHT0'] = config_dict['IN_HEIGHT']
            config_dict['OUT_HEIGHT1'] = config_dict['IN_HEIGHT']
            split_array = [config_dict['OUT_WIDTH0'], config_dict['OUT_WIDTH1']]
            class_name = "StreamingSplitWidths"

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
        Y0 = helper.make_tensor_value_info(
            "Y0",
            onnx_out_type,
            [1, config_dict["OUT_CH0"], config_dict["OUT_HEIGHT0"], config_dict["OUT_WIDTH0"]],
        )
        Y1 = helper.make_tensor_value_info(
            "Y1",
            onnx_out_type,
            [1, config_dict["OUT_CH1"], config_dict["OUT_HEIGHT1"], config_dict["OUT_WIDTH1"]],
        )

        X_scale = helper.make_tensor("X_scale", TensorProto.FLOAT, [], [float(config_dict["X_SCALE"])])
        Y_scale = helper.make_tensor("Y_scale", TensorProto.FLOAT, [], [float(config_dict["Y_SCALE"])])

        # ZPs must match the tensor element types
        X_zp = helper.make_tensor("X_zp", onnx_in_type, [], [int(config_dict["X_ZP"])])
        Y_zp = helper.make_tensor("Y_zp", onnx_out_type, [], [int(config_dict["Y_ZP"])])

        dqlinear = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "X_scale", "X_zp"],
            outputs=["X_dq"],
        )
        split = helper.make_tensor("split", TensorProto.INT64, [2], split_array)

        Split_node = helper.make_node(
            "Split",
            inputs=["X_dq", "split"],
            outputs=["Y0_dq", "Y1_dq"],
            axis=config_dict['SPLIT_AXIS'],
        )

        qlinear0 = helper.make_node(
            "QuantizeLinear",
            inputs=["Y0_dq", "Y_scale", "Y_zp"],
            outputs=["Y0"],
        )

        qlinear1 = helper.make_node(
            "QuantizeLinear",
            inputs=["Y1_dq", "Y_scale", "Y_zp"],
            outputs=["Y1"],
        )
        
        graph = helper.make_graph(
            [dqlinear, Split_node, qlinear0, qlinear1],
            "qsplit_test",
            [X],
            [Y0, Y1],
            initializer=[X_scale, X_zp, Y_scale, Y_zp, split],
        )
        model = helper.make_model(graph, producer_name="qonnx")

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        y0, y1 = sess.run(None, {"X": input_tensor})

        # Optional: cast ORT output to expected numpy dtype (ORT should already produce correct dtype)
        y0 = y0.astype(np_out_type, copy=False)
        y1 = y1.astype(np_out_type, copy=False)

        # shift based on Po2 scales (assumes ratio is power-of-two)
        shift = int(np.log2(float(config_dict["Y_SCALE"]) / float(config_dict["X_SCALE"])))

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.include("DequantQuant.hpp")
        cwr.add_line("namespace test_config {")
        cwr.indent()

        for key, value in config_dict.items():
            if key in ["X_SCALE", "W_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {float(value)}f;")
            else:
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                    cwr.add_line(f"const bool {key} = {value_str};")
                else:
                    cwr.add_line(f"const int {key} = {int(value)};")

        typedef_suffix = "u" if in_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{in_bits}> TInput;")
        cwr.add_line(f"using TInputWord = std::array<TInput, CH_PAR>;")

        typedef_suffix = "u" if out_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{out_bits}> TOutput;")
        cwr.add_line(f"using TOutputWord = std::array<TOutput, CH_PAR>;")

        cwr.add_line(f"typedef DequantQuantPo2<{shift}, TInput, TOutput> Quantizer;")
        cwr.add_line(
            f"using StreamingSplit = {class_name}<TInputWord, TInput, TOutputWord, TOutput, Quantizer, SPLIT_POINT, IN_HEIGHT, IN_WIDTH, IN_CH, CH_PAR, W_PAR>;"
        )

        cwr.add_lines(
            csnake.Variable(
                "input_tensor",
                primitive="TInput",
                value=input_tensor,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "output_tensor0",
                primitive="TOutput",
                value=y0,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "output_tensor1",
                primitive="TOutput",
                value=y1,
            ).generate_initialization()
        )

        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_8bit_po2_signed_splitch(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_HEIGHT": 2,
            "IN_WIDTH": 2,
            "IN_CH": 16,
            "CH_PAR": 2,
            "W_PAR": 2,
            "SPLIT_AXIS": 1,  # Channel axis
            "SPLIT_POINT": 4,
            "X_SCALE": 2**-2,
            "Y_SCALE": 2**-3,
            "X_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_8bit_po2_signed_splitw(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_HEIGHT": 2,
            "IN_WIDTH": 16,
            "IN_CH": 2,
            "CH_PAR": 2,
            "W_PAR": 2,
            "SPLIT_AXIS": 3,  # Width axis
            "SPLIT_POINT": 4,
            "X_SCALE": 2**-2,
            "Y_SCALE": 2**-3,
            "X_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_8bit_po2_signed_splith(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_HEIGHT": 16,
            "IN_WIDTH": 2,
            "IN_CH": 2,
            "CH_PAR": 2,
            "W_PAR": 2,
            "SPLIT_AXIS": 2,  # Height axis
            "SPLIT_POINT": 4,
            "X_SCALE": 2**-2,
            "Y_SCALE": 2**-3,
            "X_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)