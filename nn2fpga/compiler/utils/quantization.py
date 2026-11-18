import numpy as np
import qonnx.custom_op.general.quant as qonnx_quant


# Save original references before monkey-patching.
_original_min_int = qonnx_quant.min_int
_original_max_int = qonnx_quant.max_int


def _np_min_int(signed: bool, narrow_range: bool, bit_width: int) -> np.generic:
    val = _original_min_int(signed, narrow_range, bit_width)
    return np.array(val)


def _np_max_int(signed: bool, narrow_range: bool, bit_width: int) -> np.generic:
    val = _original_max_int(signed, narrow_range, bit_width)
    return np.array(val)


def safe_int_quant_call(*args, **kwargs):
    # Monkey-patch min/max to return NumPy-safe scalars for QONNX quant.
    qonnx_quant.min_int = _np_min_int
    qonnx_quant.max_int = _np_max_int

    try:
        return qonnx_quant.quant(*args, **kwargs)
    finally:
        qonnx_quant.min_int = _original_min_int
        qonnx_quant.max_int = _original_max_int


def quant_array(inp_tensor, scale, zeropt, bitwidth, signed, narrow, rounding_mode):
    """Quantize an input tensor with QONNX and return the resulting integer values."""

    quantized_tensor = safe_int_quant_call(
        inp_tensor,
        scale=scale,
        zeropt=zeropt,
        bitwidth=bitwidth,
        signed=signed,
        narrow=narrow,
        rounding_mode=rounding_mode,
    )

    # int_quant has already rounded the integer code. Dividing its rescaled
    # output by scale can leave a value infinitesimally below that integer;
    # recover the code by rounding instead of truncating toward zero.
    quantized_tensor = quantized_tensor / scale
    quantized_tensor = quantized_tensor + zeropt
    return np.rint(quantized_tensor).astype(np.int32)


def quant_array_no_clipping(inp_tensor, scale, zeropt, bitwidth, signed, rounding_mode):
    """Apply QONNX scaling and rounding without clipping to estimate required bitwidth."""

    quantized_tensor = np.asarray(inp_tensor) / scale
    quantized_tensor = quantized_tensor + zeropt
    if bitwidth == 1 and signed:
        quantized_tensor = np.where(
            quantized_tensor >= 0.0,
            np.ones(quantized_tensor.shape, dtype=quantized_tensor.dtype),
            -np.ones(quantized_tensor.shape, dtype=quantized_tensor.dtype),
        )
    else:
        rounding_fx = qonnx_quant.resolve_rounding_mode(rounding_mode)
        quantized_tensor = rounding_fx(quantized_tensor)
    return quantized_tensor.astype(np.int32)
