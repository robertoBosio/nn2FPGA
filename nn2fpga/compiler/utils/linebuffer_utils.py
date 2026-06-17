def normalize_pads(pads):
    """Return pads as [top, left, bottom, right]."""
    pads = list(pads)
    if len(pads) == 2:
        return [pads[0], pads[1], pads[0], pads[1]]
    return pads


def circular_linebuffer_latency(input_shape, pads, width_unroll, channel_unroll):
    """Latency of StreamingCircularLineBuffer's padded virtual-image scan."""
    pads = normalize_pads(pads)
    padded_height = input_shape[-3] + pads[0] + pads[2]
    padded_width = input_shape[-2] + pads[1] + pads[3]
    return padded_height * (padded_width // width_unroll) * (
        input_shape[-1] // channel_unroll
    )


def circular_linebuffer_width_compatible(
    input_shape, output_shape, pads, width_unroll
):
    """Check circular linebuffer W_PAR constraints for padded and output widths."""
    pads = normalize_pads(pads)
    padded_width = input_shape[-2] + pads[1] + pads[3]
    return (
        input_shape[-2] % width_unroll == 0
        and padded_width % width_unroll == 0
        and output_shape[-2] % width_unroll == 0
    )


def validate_circular_linebuffer_attrs(kernel_shape, pads, dilations=None):
    """Fail early for linebuffer cases unsupported by the circular implementation."""
    pads = normalize_pads(pads)
    if dilations is not None and list(dilations) != [1, 1]:
        raise ValueError(
            "StreamingCircularLineBuffer currently supports dilation [1, 1] only"
        )
    if all(k == 1 for k in kernel_shape) and any(pads):
        raise ValueError(
            "StreamingCircularLineBuffer does not support padded 1x1 operators"
        )


def has_pointwise_linebuffer_bypass(kernel_shape, strides, pads, width_unroll):
    """Return true for the known 1x1/no-pad/no-stride direct-stream bypass."""
    pads = normalize_pads(pads)
    return (
        all(k == 1 for k in kernel_shape)
        and all(s == 1 for s in strides)
        and width_unroll == 1
        and not any(pads)
    )


def circular_linebuffer_compatible_or_bypassed(
    input_shape, output_shape, kernel_shape, strides, pads, width_unroll, dilations=None
):
    """Return true if this DSE point can use or bypass the circular linebuffer."""
    if has_pointwise_linebuffer_bypass(kernel_shape, strides, pads, width_unroll):
        return True
    validate_circular_linebuffer_attrs(kernel_shape, pads, dilations)
    return circular_linebuffer_width_compatible(
        input_shape, output_shape, pads, width_unroll
    )
