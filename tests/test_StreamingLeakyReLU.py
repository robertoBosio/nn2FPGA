import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper
from .base_hls_test import BaseHLSTest

class TestStreamingLeakyReLU(BaseHLSTest):

    @property
    def operator_filename(self):
        return "StreamingLeakyReLU"

    @property
    def unit_filename(self):
        return "StreamingLeakyReLU"
    
    def generate_lut_memory(self, config_dict):
        nbits = config_dict["INPUT_DATAWIDTH"]
        LUT_entries = 1 << nbits

        # 1) Choose container dtype for ONNX and NumPy
        if nbits <= 8:
            onnx_int_type = TensorProto.INT8
            np_int_type = np.int8
        elif nbits <= 16:
            onnx_int_type = TensorProto.INT16
            np_int_type = np.int16
        elif nbits <= 32:
            onnx_int_type = TensorProto.INT32
            np_int_type = np.int32
        else:
            raise ValueError(f"Unsupported bitwidth {nbits} (> 32).")

        # 2) Raw code values: 0 .. 2^nbits - 1
        raw_codes = np.arange(LUT_entries, dtype=np.int64)

        # 3) Sign-extend from nbits to container width
        sign_bit = 1 << (nbits - 1)
        full_range = 1 << nbits

        signed_values = raw_codes.copy()
        signed_values[signed_values >= sign_bit] -= full_range
        # At this point, signed_values holds the *mathematical* two's complement values
        # corresponding to each n-bit code.

        # 4) Cast to container dtype expected by ONNX and shape it
        input_tensor = signed_values.astype(np_int_type).reshape((1, LUT_entries, 1, 1))

        # Define the input tensor to accomodate enough values to fill the LUT
        X = helper.make_tensor_value_info(
            "X",
            TensorProto.INT8,
            [
                1,
                LUT_entries,
                1,
                1,
            ],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            TensorProto.INT8,
            [
                1,
                LUT_entries,
                1,
                1,
            ],
        )

        X_scale = helper.make_tensor(
            "X_scale", TensorProto.FLOAT, [], [config_dict["X_SCALE"]]
        )
        X_zp = helper.make_tensor("X_zp", TensorProto.INT8, [], [config_dict["X_ZP"]])
        Y_scale = helper.make_tensor(
            "Y_scale", TensorProto.FLOAT, [], [config_dict["Y_SCALE"]]
        )
        Y_zp = helper.make_tensor("Y_zp", TensorProto.INT8, [], [config_dict["Y_ZP"]])

        dqlinear = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "X_scale", "X_zp"],
            outputs=["X_dq"],
        )

        LeakyReLU = helper.make_node(
            "LeakyRelu",
            alpha=config_dict["LEAKY_ALPHA"],
            inputs=["X_dq"],
            outputs=["Y_dq"],
        )

        qlinear = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dq", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [dqlinear, LeakyReLU, qlinear],
            "qleakyrelu_test",
            [X],
            [Y],
            initializer=[X_scale, X_zp, Y_scale, Y_zp],
        )
        model_qonnx = helper.make_model(graph, producer_name="qonnx")
        sess = ort.InferenceSession(
            model_qonnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        y = sess.run(None, {"X": input_tensor})[0]
        lut_values = y.flatten().tolist()

        lut_variable = csnake.Variable(
            name=f"LUTmem",
            primitive=f"TOutput",
            value=lut_values,
        )
        return lut_variable.generate_initialization()
    
    def generate_config_file(self, config_dict):

        # random tensors
        input_tensor = np.random.randint(
            -128,
            127,
            size=(
                1,
                config_dict["IN_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ),
            dtype=np.int8,
        )

        X = helper.make_tensor_value_info(
            "X",
            TensorProto.INT8,
            [
                1,
                config_dict["IN_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ],
        )
        Y = helper.make_tensor_value_info(
            "Y",
            TensorProto.INT8,
            [
                1,
                config_dict["IN_CH"],
                config_dict["IN_HEIGHT"],
                config_dict["IN_WIDTH"],
            ],
        )

        X_scale = helper.make_tensor(
            "X_scale", TensorProto.FLOAT, [], [config_dict["X_SCALE"]]
        )
        X_zp = helper.make_tensor("X_zp", TensorProto.INT8, [], [config_dict["X_ZP"]])
        Y_scale = helper.make_tensor(
            "Y_scale", TensorProto.FLOAT, [], [config_dict["Y_SCALE"]]
        )
        Y_zp = helper.make_tensor("Y_zp", TensorProto.INT8, [], [config_dict["Y_ZP"]])

        dqlinear = helper.make_node(
            "DequantizeLinear",
            inputs=["X", "X_scale", "X_zp"],
            outputs=["X_dq"],
        )

        leakyrelu = helper.make_node(
            "LeakyRelu",
            alpha=config_dict["LEAKY_ALPHA"],
            inputs=["X_dq"],
            outputs=["Y_dq"],
        )

        qlinear = helper.make_node(
            "QuantizeLinear",
            inputs=["Y_dq", "Y_scale", "Y_zp"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [dqlinear, leakyrelu, qlinear],
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

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            if key in ["X_SCALE", "W_SCALE", "Y_SCALE", "LEAKY_ALPHA"]:
                cwr.add_line(f"const float {key} = {value}f;")
            else:
                cwr.add_line(f"const int {key} = {value};")
        cwr.add_line(f"typedef ap_int<{config_dict['INPUT_DATAWIDTH']}> TInput;")
        cwr.add_line(f"typedef ap_uint<{config_dict['OUTPUT_DATAWIDTH']}> TOutput;")
        cwr.add_line(f"typedef DequantQuantPo2<0, TInput, TOutput> Quantizer;")
        cwr.add_lines(
            csnake.Variable(
                "input_tensor",
                primitive=f"ap_int<{config_dict['INPUT_DATAWIDTH']}>",
                value=input_tensor,
            ).generate_initialization()
        )
        cwr.add_lines(
            csnake.Variable(
                "output_tensor",
                primitive=f"ap_int<{config_dict['OUTPUT_DATAWIDTH']}>",
                value=y,
            ).generate_initialization()
        )
        cwr.add_lines(
            self.generate_lut_memory(config_dict).code
        )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_7x7_po2(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 8,
            "OUTPUT_DATAWIDTH": 8,
            "IN_HEIGHT": 2,
            "IN_WIDTH": 2,
            "IN_CH": 2,
            "CH_PAR": 2,
            "W_PAR": 2,
            "X_SCALE": 2**-5,
            "Y_SCALE": 2**-5,
            "X_ZP": 0,
            "Y_ZP": 0,
            "LEAKY_ALPHA": 0.1,
            "LUT_SIZE": 256,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps)
