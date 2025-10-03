import numpy as np
import csnake
from .base_hls_test import BaseHLSTest

class TestStreamingMemory(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "StreamingMemory"

    @property
    def unit_filename(self) -> str:
        return "StreamingMemory"

    @staticmethod
    def pack_values_to_int32words(arr: np.ndarray, bitwidth: int) -> np.ndarray:
        """
        Packs values from the input array into 32-bit words, ensuring that each value
        fits entirely within a word (no value is split between two words). Uses padding
        as needed.

        Args:
            arr (np.ndarray): Input array of unsigned integers.
            bitwidth (int): Number of bits per value. Must be <= 32.

        Returns:
            np.ndarray: Packed 32-bit words as a 1D array of dtype=np.uint32.
        """
        bitwidth = int(bitwidth)
        if bitwidth > 32 or bitwidth <= 0:
            raise ValueError("bitwidth must be between 1 and 32")

        arr = arr.flatten()  # Ensure the input is a 1D array
        values_per_word = 32 // bitwidth  # Max number of values per word

        # Pad the array to make its length a multiple of values_per_word
        padded_len = int(
            ((len(arr) + values_per_word - 1) // values_per_word) * values_per_word
        )
        padded_arr = np.zeros(padded_len, dtype=np.uint32)
        padded_arr[: len(arr)] = arr

        packed = []
        for i in range(0, padded_len, values_per_word):
            word = 0
            for j in range(values_per_word):
                word |= (padded_arr[i + j] & ((1 << bitwidth) - 1)) << (bitwidth * j)
            packed.append(word)

        return np.array(packed, dtype=np.uint32)

    def generate_config_file(self, config_dict, dtype) -> str:

        weight_tensor = np.random.randint(
            -(2 ** (config_dict["OUTPUT_DATAWIDTH"] - 1)),
            2 ** (config_dict["OUTPUT_DATAWIDTH"] - 1) - 1,
            size=(
                config_dict["OUT_CH"],
                config_dict["IN_CH"],
                config_dict["FH"],
                config_dict["FW"],
            ),
            dtype=dtype,
        )
        # compute grouping parameters
        CH_GROUPS = (config_dict["OUT_CH"] * config_dict["IN_CH"]) // (
            config_dict["OUT_CH_PAR"] * config_dict["IN_CH_PAR"]
        )

        # reshape
        reshaped = weight_tensor.reshape(
            CH_GROUPS,
            config_dict["OUT_CH_PAR"] * config_dict["IN_CH_PAR"],
            config_dict["FH"] * config_dict["FW"],
        )

        # pack weights
        packed = TestStreamingMemory.pack_values_to_int32words(
            reshaped, config_dict["OUTPUT_DATAWIDTH"]
        )

        # Dump the tensors in a hpp file
        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            cwr.add_line(f"const int {key} = {value};")
        cwr.add_line(f"typedef ap_uint<{config_dict['INPUT_DATAWIDTH']}> TInput;")
        cwr.add_line(f"typedef ap_int<{config_dict['OUTPUT_DATAWIDTH']}> TOutput;")
        cwr.add_line(f"typedef DequantQuantEqual<TOutput> Quantizer;")
        weight_tensor_variable = csnake.Variable(
            "weight_tensor",
            primitive=f"ap_int<{config_dict['OUTPUT_DATAWIDTH']}>",
            value=reshaped,
        )
        packed_tensor_variable = csnake.Variable(
            "packed_weights", primitive="TInput", value=packed
        )
        cwr.add_lines(weight_tensor_variable.generate_initialization())
        cwr.add_lines(packed_tensor_variable.generate_initialization())
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_8bit_nopadding_2D(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 32,
            "OUTPUT_DATAWIDTH": 8,
            "OUT_HEIGHT": 4,
            "OUT_WIDTH": 4,
            "IN_CH": 4,
            "OUT_CH": 4,
            "FH": 1,
            "FW": 1,
            "IN_CH_PAR": 2,
            "OUT_CH_PAR": 2,
            "W_PAR": 2,
            "DATA_PER_WORD": 4,
            "DATA_TO_SHIFT": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps, workdir=".", dtype=np.int8)

    def test_6bit_padding_2D(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 32,
            "OUTPUT_DATAWIDTH": 6,
            "OUT_HEIGHT": 4,
            "OUT_WIDTH": 4,
            "IN_CH": 4,
            "OUT_CH": 4,
            "FH": 1,
            "FW": 1,
            "IN_CH_PAR": 2,
            "OUT_CH_PAR": 2,
            "W_PAR": 2,
            "DATA_PER_WORD": 5,
            "DATA_TO_SHIFT": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps, workdir=".", dtype=np.int8)

    def test_8bit_1D(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 32,
            "OUTPUT_DATAWIDTH": 8,
            "OUT_HEIGHT": 4,
            "OUT_WIDTH": 4,
            "IN_CH": 1,
            "OUT_CH": 4,
            "FH": 1,
            "FW": 1,
            "IN_CH_PAR": 1,
            "OUT_CH_PAR": 2,
            "W_PAR": 2,
            "DATA_PER_WORD": 4,
            "DATA_TO_SHIFT": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps, workdir=".", dtype=np.int8)

    def test_4bit_padding_4D(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 32,
            "OUTPUT_DATAWIDTH": 4,
            "OUT_HEIGHT": 4,
            "OUT_WIDTH": 4,
            "IN_CH": 8,
            "OUT_CH": 8,
            "FH": 3,
            "FW": 3,
            "IN_CH_PAR": 2,
            "OUT_CH_PAR": 2,
            "W_PAR": 2,
            "DATA_PER_WORD": 8,
            "DATA_TO_SHIFT": 0,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps, workdir=".", dtype=np.int8)

    def test_8bit_shift(self, hls_steps):
        np.random.seed(42)
        config_dict = {
            "INPUT_DATAWIDTH": 32,
            "OUTPUT_DATAWIDTH": 8,
            "OUT_HEIGHT": 4,
            "OUT_WIDTH": 4,
            "IN_CH": 4,
            "OUT_CH": 4,
            "FH": 1,
            "FW": 1,
            "IN_CH_PAR": 2,
            "OUT_CH_PAR": 2,
            "W_PAR": 2,
            "DATA_PER_WORD": 4,
            "DATA_TO_SHIFT": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(config_dict, hls_steps, workdir=".", dtype=np.int8)