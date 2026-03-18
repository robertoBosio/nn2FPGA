import numpy as np  # tensor generation and math
import onnxruntime as ort # runs the ONNX model on CPU → golden output
import csnake # generates C++ code programmatically
from onnx import TensorProto, helper # ONNX graph building utilities
from .base_hls_test import BaseHLSTest

class TestStreamingMatMul(BaseHLSTest):
  # this class inherits run(), generate_hls_script(), get_tensorproto_dtype(), etc. from base_hls_test.py
    @property
    def operator_filename(self) -> str:
        return "StreamingMatMul" # → looks for StreamingMatMul.hpp

    @property
    def unit_filename(self) -> str:
        return "StreamingMatMul" # → looks for UnitStreamingMatMul.cpp

    def generate_config_file(self, config_dict):
        # MatMul requires A[H,W] @ B[W,N] → inner dims must match
        assert config_dict["IN_CH_A"] == config_dict["IN_CH_B"], "Input channel dimensions must match for matmul"
        assert config_dict["IN_WIDTH_A"] == config_dict["IN_HEIGHT_B"], "Inner dimensions must match for matmul"

        ina_unsigned = bool(config_dict.get("INPUTA_IS_UNSIGNED", False)) # → False (signed INT8)
        inb_unsigned = bool(config_dict.get("INPUTB_IS_UNSIGNED", False)) # → False (signed INT8)
        out_unsigned = bool(config_dict.get("OUTPUT_IS_UNSIGNED", False)) # → False (signed INT8)

        ina_bits = int(config_dict["INPUTA_DATAWIDTH"]) # → 8
        inb_bits = int(config_dict["INPUTB_DATAWIDTH"]) # → 8
        out_bits = int(config_dict["OUTPUT_DATAWIDTH"]) # → 8

        onnx_ina_type = self.get_tensorproto_dtype(ina_bits, ina_unsigned) # → TensorProto.INT8  (used in ONNX graph nodes)
        onnx_inb_type = self.get_tensorproto_dtype(inb_bits, inb_unsigned) # → TensorProto.INT8
        onnx_out_type = self.get_tensorproto_dtype(out_bits, out_unsigned) # → TensorProto.INT8
        np_ina_type = self.get_numpy_dtype(ina_bits, ina_unsigned) # → np.int8  (used to generate random tensors)
        np_inb_type = self.get_numpy_dtype(inb_bits, inb_unsigned) # → np.int8
        np_out_type = self.get_numpy_dtype(out_bits, out_unsigned) # → np.int8

        # _____Random input tensor in correct integer domain/range____
        #----------------------------------------------------
        in_info = np.iinfo(np_ina_type) # → iinfo for int8: min=-128, max=127
        inputa_tensor = np.random.randint(  
            int(in_info.min),
            int(in_info.max) + 1, # randint upper bound is exclusive (128 for int8)
            size=(1,config_dict["IN_CH_A"],config_dict["IN_HEIGHT_A"],config_dict["IN_WIDTH_A"],), # [batch, IN_CH_A, IN_HEIGHT_A, IN_WIDTH_A]
            dtype=np_ina_type,
        )

        in_info = np.iinfo(np_inb_type) # → iinfo for int8: min=-128, max=127
        inputb_tensor = np.random.randint(
            int(in_info.min),
            int(in_info.max) + 1,  # randint upper bound is exclusive (128 for int8)
            size=(
                1,
                config_dict["IN_CH_B"],
                config_dict["IN_HEIGHT_B"],
                config_dict["IN_WIDTH_B"],
            ), # [batch, IN_CH_A, IN_HEIGHT_A, IN_WIDTH_A]
            dtype=np_inb_type,
        )
        #----------------------------------------------------

        # Define I/O Tensor (shape + dtype metadata)
        #---------------------------------
        # These are just metadata declarations — no actual data yet — so we use the ONNX dtypes here (not numpy dtypes)
        A = helper.make_tensor_value_info(
            "A", # str: tensor name (must match node input/output names)
            onnx_ina_type, # int: data type from TensorProto (e.g. TensorProto.FLOAT)
            [
                1,
                config_dict["IN_CH_A"],
                config_dict["IN_HEIGHT_A"],
                config_dict["IN_WIDTH_A"],
            ],# tensor shape: [batch, IN_CH_A, IN_HEIGHT_A, IN_WIDTH_A]
        )
        B = helper.make_tensor_value_info(
            "B",
            onnx_inb_type,
            [
                1,
                config_dict["IN_CH_B"],
                config_dict["IN_HEIGHT_B"],
                config_dict["IN_WIDTH_B"],
            ],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            onnx_out_type,
            [
                1,
                config_dict["IN_CH_A"],
                config_dict["IN_HEIGHT_A"],
                config_dict["IN_WIDTH_B"],
            ],
        )
        #---------------------------------

        #_____Define Constant Tensors (scale + zero-point) ____
        #_____________________________________________________
        A_scale = helper.make_tensor(
            "A_scale", TensorProto.FLOAT, [], [config_dict["A_SCALE"]] # name="A_scale", dtype=FLOAT, shape=[] (scalar), value=0.03125
        )
        B_scale = helper.make_tensor(
            "B_scale", TensorProto.FLOAT, [], [config_dict["B_SCALE"]] # name="B_scale", dtype=FLOAT, shape=[] (scalar), value=0.03125
        )
        A_zp = helper.make_tensor(
            "A_zp",
            onnx_ina_type,
            [],
            [config_dict["A_ZP"]], # zero-point = 0 (no offset)
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
        #_____________________________________________________


        #______Define Graph Nodes______
        #----------------------------------------------------
        dequant0 = helper.make_node(
            "DequantizeLinear", # ONNX op name
            inputs=["A", "A_scale", "A_zp"], # reads these tensors
            outputs=["A_dequant"],  # produces this tensor
        )
        # computes: A_dequant = (A - A_zp) * A_scale
        #         = (INT8_value - 0) * 0.03125
#         → converts INT8 → FLOAT
        dequant1 = helper.make_node(
            "DequantizeLinear",
            inputs=["B", "B_scale", "B_zp"],
            outputs=["B_dequant"],
        ) # computes: B_dequant = (B - 0) * 0.03125

        quant = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dequant", "Y_scale", "Y_zp"],
            outputs=["Y"],
        ) # computes: Y = clip(round(Y_dequant / Y_scale) + Y_zp, -128, 127)
          # = clip(round(Y_dequant / 0.03125) + 0, -128, 127)
          #→ converts FLOAT → INT8

        matmul = helper.make_node(
            "MatMul",
            inputs=[
                "A_dequant",
                "B_dequant",
            ], # both FLOAT now
            outputs=["Y_dequant"],
        ) # computes: Y_dequant = A_dequant @ B_dequant
          # per channel: [H_A, W] @ [W, W_B] → [H_A, W_B]  (FLOAT)


        #______Assemble the Graph______

        graph = helper.make_graph(
            [dequant0, dequant1, matmul, quant], # nodes in execution order
            "matmul_test", # graph name
            [A, B],      # graph inputs
            [Y],        # graph outputs
            initializer=[A_scale, B_scale, A_zp, B_zp, Y_scale, Y_zp], #  constant tensors available to all nodes
        ) 

        #_____Run on CPU → Golden Output______

        model = helper.make_model(graph, producer_name="qonnx")
        sess = ort.InferenceSession(
            model.SerializeToString(), # serialize to bytes
            providers=["CPUExecutionProvider"]  # run on CPU
        )
        y = sess.run(None,  # return all outputs
        {"A": inputa_tensor, "B": inputb_tensor} # feed actual data
        )[0] # y is now the INT8 golden output → stored in output_tensor in test_config.hpp

        #____Scale & Accumulator Calculation____
        # Aligning scales in input
        shift = int(np.log2(config_dict["Y_SCALE"] / (config_dict["A_SCALE"] * config_dict["B_SCALE"])))
        # = log2(2^-5 / (2^-5 * 2^-5))
        # = log2(2^-5 / 2^-10)
        # = log2(2^5) = 5
        # → hardware does acc >> 5 instead of float division

        tacc_bits = ina_bits + inb_bits + int(np.ceil(np.log2(config_dict["IN_WIDTH_A"])))
        # = 8 + 8 + 4 = 20
        # ↑ enough bits to hold the full unquantized dot product without overflow

        if ina_unsigned != inb_unsigned:
            tacc_bits += 1  # extra bit for signed + unsigned addition
        config_dict["ACC_DATAWIDTH"] = tacc_bits

        #______test_config.hpp Generation via csnake______
        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        # writes all config_dict entries as C++ constants:
        #   const int IN_HEIGHT_A = 16;
        #   const int CH_PAR = 4;
        #   const float A_SCALE = 0.03125f;
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
        cwr.add_line(f"typedef DequantQuantPo2<{shift}, TAcc, TOutput> Quantizer;")

        # embeds tensors as C++ array literals:
        #   const TInputA input_tensor0[1][16][16][16] = {{...}};
        #   const TInputB input_tensor1[1][16][16][16] = {{...}};
        #   const TOutput output_tensor[1][16][16][16] = {{...}}; ← golden from ONNX
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
        np.random.seed(42)  # reproducible random tensors
        config_dict = {
            "INPUTA_IS_UNSIGNED": False,
            "INPUTB_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "INPUTA_DATAWIDTH": 8,
            "INPUTB_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "IN_HEIGHT_A": 16,
            "IN_WIDTH_A": 16,
            "IN_CH_A": 16,
            "IN_HEIGHT_B": 16,
            "IN_WIDTH_B": 16,
            "IN_CH_B": 16,
            "CH_PAR": 4,
            "W_PAR": 4,
            "A_SCALE": 2**-5,
            "B_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps) # → generates test_config.hpp → runs Vitis HLS csim → checks "Passed."
    def test_pertensor_po2_signed_1(self, hls_steps):
        np.random.seed(42)  # reproducible random tensors
        config_dict = {
            "INPUTA_IS_UNSIGNED": False,
            "INPUTB_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "INPUTA_DATAWIDTH": 8,
            "INPUTB_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "IN_HEIGHT_A": 16,
            "IN_WIDTH_A": 16,
            "IN_CH_A": 16,
            "IN_HEIGHT_B": 16,
            "IN_WIDTH_B": 16,
            "IN_CH_B": 16,
            "CH_PAR": 4,
            "W_PAR": 4,
            "A_SCALE": 2**-4,
            "B_SCALE": 2**-7,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
    def test_pertensor_po2_signed_2(self, hls_steps):
        np.random.seed(42)  # reproducible random tensors
        config_dict = {
            "INPUTA_IS_UNSIGNED": False,
            "INPUTB_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "INPUTA_DATAWIDTH": 8,
            "INPUTB_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "IN_HEIGHT_A": 16,
            "IN_WIDTH_A": 16,
            "IN_CH_A": 16,
            "IN_HEIGHT_B": 16,
            "IN_WIDTH_B": 16,
            "IN_CH_B": 16,
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
        
    def test_pertensor_po2_signed_3(self, hls_steps):
        np.random.seed(42)  # reproducible random tensors
        config_dict = {
            "INPUTA_IS_UNSIGNED": True,
            "INPUTB_IS_UNSIGNED": False,
            "OUTPUT_IS_UNSIGNED": False,
            "INPUTA_DATAWIDTH": 8,
            "INPUTB_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "IN_HEIGHT_A": 16,
            "IN_WIDTH_A": 16,
            "IN_CH_A": 16,
            "IN_HEIGHT_B": 16,
            "IN_WIDTH_B": 16,
            "IN_CH_B": 16,
            "CH_PAR": 4,
            "W_PAR": 4,
            "A_SCALE": 2**-5,
            "B_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "A_ZP": 0,
            "B_ZP": 0,
            "Y_ZP": 0,
            "PIPELINE_DEPTH": 5,
        }   
        self.run(config_dict, hls_steps)