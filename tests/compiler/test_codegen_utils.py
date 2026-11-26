import pytest

from nn2fpga.compiler.core.tensor_type import FloatTensorType, QuantizedTensorType
from nn2fpga.compiler.utils.codegen_utils import HLSCodegen


def test_vitis_codegen_types_match_existing_spelling():
    codegen = HLSCodegen("vitis")
    quant = QuantizedTensorType(8, 1, 1.0, 0)

    assert codegen.scalar_type(quant) == "ap_int<8>"
    assert codegen.word_type(quant, 4) == "std::array<ap_int<8>, 4>"
    assert codegen.channel_type("TWord") == "hls::stream<TWord>"
    assert codegen.axis_word_type(128) == "ap_axiu<128, 0, 0, 0>"
    assert codegen.supports_step_calls


def test_catapult_codegen_types_use_ac_types():
    codegen = HLSCodegen("catapult")
    quant = QuantizedTensorType(8, 0, 1.0, 0)

    assert codegen.scalar_type(quant) == "ac_int<8, false>"
    assert codegen.word_type(quant, 4) == "std::array<ac_int<8, false>, 4>"
    assert codegen.channel_type("TWord") == "ac_channel<TWord>"
    assert codegen.axis_word_type(128) == "AxisWord<128>"
    assert not codegen.supports_step_calls


def test_catapult_codegen_rejects_float_types():
    with pytest.raises(ValueError, match="does not yet support float"):
        HLSCodegen("catapult").scalar_type(FloatTensorType())
