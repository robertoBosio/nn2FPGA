import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest

class TestStreamingSigmoid(BaseHLSTest):

    @property
    def operator_filename(self):
        return "StreamingLUT"

    @property
    def unit_filename(self):
        return "StreamingLUT"

    def generate_lut_memory(self, config_dict):
        """
        Generate LUT contents for (DequantizeLinear -> HardSigmoid -> Mul -> QuantizeLinear),
        supporting signed/unsigned input and signed/unsigned output independently.

        Expected config_dict keys:
        INPUT_DATAWIDTH (int)
        INPUT_IS_UNSIGNED (bool, optional; default False)
        OUTPUT_IS_UNSIGNED (bool, optional; default False)
        X_SCALE (float), X_ZP (int)
        Y_SCALE (float), Y_ZP (int)
        SIGMOID_ALPHA (float)
        B_VALUE (float)
        """
        nbits = int(config_dict["INPUT_DATAWIDTH"])
        out_bits = int(config_dict["OUTPUT_DATAWIDTH"])
        LUT_entries = 1 << nbits

        in_unsigned = bool(config_dict.get("INPUT_IS_UNSIGNED", False))
        out_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False))


        def signed_from_raw_codes(raw: np.ndarray, bits: int) -> np.ndarray:
            """Map 0..2^bits-1 codes to signed two's-complement integers."""
            sign_bit = 1 << (bits - 1)
            full_range = 1 << bits
            vals = raw.astype(np.int64, copy=True)
            vals[vals >= sign_bit] -= full_range
            return vals

        # ----- choose types -----
        onnx_in_type = self.get_tensorproto_dtype(nbits, in_unsigned)
        onnx_out_type = self.get_tensorproto_dtype(out_bits, out_unsigned)
        np_in_type = self.get_numpy_dtype(nbits, in_unsigned)

        # ----- build LUT input tensor -----
        raw_codes = np.arange(LUT_entries, dtype=np.int64)

        if in_unsigned:
            # Unsigned input domain is 0..2^nbits-1
            input_values = raw_codes
        else:
            # Signed input domain is two's-complement mapping
            input_values = signed_from_raw_codes(raw_codes, nbits)

        input_tensor = input_values.astype(np_in_type).reshape((1, LUT_entries, 1, 1))

        # ----- ONNX graph I/O -----
        X = helper.make_tensor_value_info("X", onnx_in_type, [1, LUT_entries, 1, 1])
        Y = helper.make_tensor_value_info("Y", onnx_out_type, [1, LUT_entries, 1, 1])

        # ----- initializers -----
        X_scale = helper.make_tensor("X_scale", TensorProto.FLOAT, [], [float(config_dict["X_SCALE"])])
        Y_scale = helper.make_tensor("Y_scale", TensorProto.FLOAT, [], [float(config_dict["Y_SCALE"])])

        # Zero-points must match tensor element types (and signed/unsigned-ness)
        X_zp = helper.make_tensor("X_zp", onnx_in_type, [], [int(config_dict["X_ZP"])])
        Y_zp = helper.make_tensor("Y_zp", onnx_out_type, [], [int(config_dict["Y_ZP"])])

        B = helper.make_tensor("B", TensorProto.FLOAT, [], [float(config_dict["B_VALUE"])])

        # ----- nodes -----
        dqlinear = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "X_scale", "X_zp"],
            outputs=["X_dq"],
        )

        HardSigmoid = helper.make_node(
            "HardSigmoid",
            alpha=float(config_dict["SIGMOID_ALPHA"]),
            inputs=["X_dq"],
            outputs=["A"],
        )

        Mul = helper.make_node(
            "Mul",
            inputs=["A", "B"],
            outputs=["Y_dq"],
        )

        qlinear = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dq", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [dqlinear, HardSigmoid, Mul, qlinear],
            "qsigmoid_test",
            [X],
            [Y],
            initializer=[X_scale, X_zp, Y_scale, Y_zp, B],
        )

        model_qonnx = helper.make_model(graph, producer_name="qonnx")

        # ----- run -----
        sess = ort.InferenceSession(
            model_qonnx.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        y = sess.run(None, {"X": input_tensor})[0]

        # Ensure Python ints in list (avoid numpy scalar types)
        lut_values = [int(v) for v in y.flatten().tolist()]

        # ----- emit C++ variable -----
        lut_variable = csnake.Variable(
            name="LUTmem",
            primitive="TOutput",
            value=lut_values,
        )
        return lut_variable.generate_initialization()
 
    def generate_config_file(self, config_dict):
        """
        Generate a self-contained C++ test config (input/output tensors + LUT),
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
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_out_type,
            [1, config_dict["IN_CH"], config_dict["IN_HEIGHT"], config_dict["IN_WIDTH"]],
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

        B = helper.make_tensor("B", TensorProto.FLOAT, [], [float(config_dict["B_VALUE"])])

        HardSigmoid = helper.make_node(
            "HardSigmoid",
            alpha=float(config_dict["SIGMOID_ALPHA"]),
            inputs=["X_dq"],
            outputs=["A"],
        )

        Mul = helper.make_node(
            "Mul",
            inputs=["A", "B"],
            outputs=["Y_dq"],
        )

        qlinear = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dq", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [dqlinear, HardSigmoid, Mul, qlinear],
            "qsigmoid_test",
            [X],
            [Y],
            initializer=[X_scale, X_zp, Y_scale, Y_zp, B],
        )
        model = helper.make_model(graph, producer_name="qonnx")

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        y = sess.run(None, {"X": input_tensor})[0]

        # Optional: cast ORT output to expected numpy dtype (ORT should already produce correct dtype)
        y = y.astype(np_out_type, copy=False)

        # shift based on Po2 scales (assumes ratio is power-of-two)
        shift = int(np.log2(float(config_dict["Y_SCALE"]) / float(config_dict["X_SCALE"])))

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()

        for key, value in config_dict.items():
            if key in ["X_SCALE", "W_SCALE", "Y_SCALE", "SIGMOID_ALPHA", "B_VALUE"]:
                cwr.add_line(f"const float {key} = {float(value)}f;")
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

        cwr.add_lines(self.generate_lut_memory(config_dict).code)

        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_8bit_po2_signed(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_HEIGHT": 2,
            "IN_WIDTH": 2,
            "IN_CH": 2,
            "CH_PAR": 2,
            "W_PAR": 2,
            "X_SCALE": 2**-2,
            "Y_SCALE": 2**-6,
            "X_ZP": 0,
            "Y_ZP": 0,
            "SIGMOID_ALPHA": 0.1666666716337204,
            "B_VALUE": 1.0001220703125,
            "LUT_SIZE": 256,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_8bit_po2_unsigned(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": True,
            "OUTPUT_IS_UNSIGNED": True,
            "IN_HEIGHT": 2,
            "IN_WIDTH": 2,
            "IN_CH": 2,
            "CH_PAR": 2,
            "W_PAR": 2,
            "X_SCALE": 2**-2,
            "Y_SCALE": 2**-6,
            "X_ZP": 0,
            "Y_ZP": 0,
            "SIGMOID_ALPHA": 0.1666666716337204,
            "B_VALUE": 1.0001220703125,
            "LUT_SIZE": 256,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_8bit_po2_mixed(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "INPUT_IS_UNSIGNED": True,
            "OUTPUT_IS_UNSIGNED": False,
            "IN_HEIGHT": 2,
            "IN_WIDTH": 2,
            "IN_CH": 2,
            "CH_PAR": 2,
            "W_PAR": 2,
            "X_SCALE": 2**-2,
            "Y_SCALE": 2**-6,
            "X_ZP": 0,
            "Y_ZP": 0,
            "SIGMOID_ALPHA": 0.1666666716337204,
            "B_VALUE": 1.0001220703125,
            "LUT_SIZE": 256,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)