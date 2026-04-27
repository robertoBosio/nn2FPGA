import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest

class TestStreamingAdd(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "StreamingAdd"

    @property
    def unit_filename(self) -> str:
        return "StreamingAdd"

    def generate_config_file(self, config_dict):
        ina_unsigned = bool(config_dict.get("INPUTA_IS_UNSIGNED", False))
        inb_unsigned = bool(config_dict.get("INPUTB_IS_UNSIGNED", False))
        out_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False))

        ina_bits = int(config_dict["INPUTA_DATAWIDTH"])
        inb_bits = int(config_dict["INPUTB_DATAWIDTH"])
        out_bits = int(config_dict["OUTPUT_DATAWIDTH"])

        onnx_ina_type = self.get_tensorproto_dtype(ina_bits, ina_unsigned)
        onnx_inb_type = self.get_tensorproto_dtype(inb_bits, inb_unsigned)
        onnx_out_type = self.get_tensorproto_dtype(out_bits, out_unsigned)
        np_ina_type = self.get_numpy_dtype(ina_bits, ina_unsigned)
        np_inb_type = self.get_numpy_dtype(inb_bits, inb_unsigned)
        np_out_type = self.get_numpy_dtype(out_bits, out_unsigned)

        # Random input tensor in correct integer domain/range
        in_info = np.iinfo(np_ina_type)
        inputa_tensor = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,  # randint upper bound is exclusive
            size=(
                1,
                config_dict["DIM2"],
                config_dict["DIM0"],
                config_dict["DIM1"],
            ),
            dtype=np_ina_type,
        )

        in_info = np.iinfo(np_inb_type)
        inputb_tensor = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,  # randint upper bound is exclusive
            size=(
                1,
                config_dict["DIM2"],
                config_dict["DIM0"],
                config_dict["DIM1"],
            ),
            dtype=np_inb_type,
        )

        # I/O
        A = helper.make_tensor_value_info(
            "A",
            onnx_ina_type,
            [
                1,
                config_dict["DIM2"],
                config_dict["DIM0"],
                config_dict["DIM1"],
            ],
        )
        B = helper.make_tensor_value_info(
            "B",
            onnx_inb_type,
            [
                1,
                config_dict["DIM2"],
                config_dict["DIM0"],
                config_dict["DIM1"],
            ],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_out_type,
            [
                1,
                config_dict["DIM2"],
                config_dict["DIM0"],
                config_dict["DIM1"],
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

        add = helper.make_node(
            "Add",
            inputs=[
                "A_dequant",
                "B_dequant",
            ],
            outputs=["Y_dequant"],
        )

        graph = helper.make_graph(
            [dequant0, dequant1, add, quant],
            "add_test",
            [A, B],
            [Y],
            initializer=[A_scale, B_scale, A_zp, B_zp, Y_scale, Y_zp],
        )
        model = helper.make_model(graph, producer_name="qonnx")
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        y = sess.run(None, {"A": inputa_tensor, "B": inputb_tensor})[0]

        # Aligning scales in input
        common_scale = min(config_dict["A_SCALE"], config_dict["B_SCALE"])
        align_a = int(np.log2(config_dict["A_SCALE"] / common_scale))
        align_b = int(np.log2(config_dict["B_SCALE"] / common_scale))
        shift = int(np.log2(config_dict["Y_SCALE"] / common_scale))

        tacc_bits = max(
            ina_bits + align_a, inb_bits + align_b
        ) + 1  # +1 for possible overflow in addition
        if ina_unsigned != inb_unsigned:
            tacc_bits += 1  # extra bit for signed + unsigned addition
        config_dict["ACC_DATAWIDTH"] = tacc_bits

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
        typedef_suffix = "u" if ina_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{ina_bits}> TInputA;")
        typedef_suffix = "u" if inb_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{inb_bits}> TInputB;")
        typedef_suffix = "u" if (ina_unsigned and inb_unsigned) else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{tacc_bits}> TAcc;")
        typedef_suffix = "u" if out_unsigned else ""
        cwr.add_line(f"typedef ap_{typedef_suffix}int<{out_bits}> TOutput;")

        # Quantizers
        if align_a != 0:
            cwr.add_line(f"typedef DequantQuantPo2<{-align_a}, TInputA, TAcc> AlignA;")
        else:
            cwr.add_line(f"typedef DequantQuantEqual<TInputA> AlignA;")
        if align_b != 0:
            cwr.add_line(f"typedef DequantQuantPo2<{-align_b}, TInputB, TAcc> AlignB;")
        else:
            cwr.add_line(f"typedef DequantQuantEqual<TInputB> AlignB;")
        cwr.add_line(f"typedef DequantQuantPo2<{shift}, TAcc, TOutput> Quantizer;")
        cwr.add_line("typedef DequantQuantEqual<TAcc> Activation;")

        cwr.add_lines(
            csnake.Variable(
                "input_tensor0",
                primitive=f"TInputA",
                value=inputa_tensor,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "input_tensor1",
                primitive=f"TInputB",
                value=inputb_tensor,
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
            "DIM0": 4,
            "DIM1": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 1,
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
            "DIM0": 4,
            "DIM1": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 1,
            "A_SCALE": 2**-5,
            "B_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)

    def test_pertensor_po2_hybrid(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTA_IS_UNSIGNED": False,
            "INPUTB_IS_UNSIGNED": True,
            "OUTPUT_IS_UNSIGNED": True,
            "INPUTA_DATAWIDTH": 8,
            "INPUTB_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "DIM0": 4,
            "DIM1": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 1,
            "A_SCALE": 2**-5,
            "B_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    
    def test_pertensor_po2_mixed_scales(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUTA_IS_UNSIGNED": False,
            "INPUTB_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "INPUTA_DATAWIDTH": 8,
            "INPUTB_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "DIM0": 4,
            "DIM1": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 1,
            "A_SCALE": 2**-4,
            "B_SCALE": 2**-6,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
