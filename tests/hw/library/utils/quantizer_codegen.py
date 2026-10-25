from fractions import Fraction

import numpy as np

from nn2fpga.compiler.utils.scale_ratio import ratio_to_shift_form


def classify_linear_quantizer(
    acc_scale,
    out_scale,
    same_type,
    max_den=1 << 20,
    atol=1e-12,
    shift_rel_tol=1e-3,
    max_shift=31,
):
    ratio = float(acc_scale) / float(out_scale)

    if np.isclose(ratio, 1.0, atol=atol) and same_type:
        return "equal", None

    log2_ratio = np.log2(ratio)
    rounded = np.round(log2_ratio)
    if np.isclose(log2_ratio, rounded, atol=atol):
        shift = int(-rounded)
        return "po2", shift

    frac = Fraction(ratio).limit_denominator(max_den)
    num, den = int(frac.numerator), int(frac.denominator)

    approx = ratio_to_shift_form(
        ratio, max_den=max_den, rel_tol=shift_rel_tol, max_shift=max_shift
    )
    if approx is not None:
        return "float_shift", approx

    return "float", (num, den)


def emit_linear_quantizer_typedef(
    cwr,
    alias,
    acc_type,
    out_type,
    acc_scale,
    out_scale,
    same_type,
    zero_point=0,
    max_den=1 << 20,
    atol=1e-12,
    shift_rel_tol=1e-3,
    max_shift=31,
):
    mode, params = classify_linear_quantizer(
        acc_scale,
        out_scale,
        same_type,
        max_den=max_den,
        atol=atol,
        shift_rel_tol=shift_rel_tol,
        max_shift=max_shift,
    )

    if mode == "equal" and zero_point != 0:
        mode = "po2"
        params = 0

    if mode == "equal":
        cwr.add_line(f"typedef DequantQuantEqual<{out_type}> {alias};")
    elif mode == "po2":
        cwr.add_line(f"const int {alias}_SHIFT = {int(params)};")
        cwr.add_line(
            f"typedef DequantQuantPo2<{alias}_SHIFT, {acc_type}, {out_type}, {int(zero_point)}> {alias};"
        )
    elif mode == "float_shift":
        num, shift = params
        cwr.add_line(f"const int {alias}_NUM = {num};")
        cwr.add_line(f"const int {alias}_SHIFT = {shift};")
        cwr.add_line(
            f"typedef DequantQuantFloatShift<{alias}_NUM, {alias}_SHIFT, {acc_type}, {out_type}, {int(zero_point)}> {alias};"
        )
    else:
        num, den = params
        cwr.add_line(f"const int {alias}_NUM = {num};")
        cwr.add_line(f"const int {alias}_DEN = {den};")
        cwr.add_line(
            f"typedef DequantQuantFloat<{alias}_NUM, {alias}_DEN, {acc_type}, {out_type}, {int(zero_point)}> {alias};"
        )


def emit_requantizer_typedef(
    cwr,
    alias,
    in_type,
    out_type,
    in_scale,
    out_scale,
    input_zero_point=0,
    output_zero_point=0,
    max_den=1 << 20,
    atol=1e-12,
):
    ratio = float(in_scale) / float(out_scale)
    input_zero_point = int(input_zero_point)
    output_zero_point = int(output_zero_point)

    if (
        np.isclose(ratio, 1.0, atol=atol)
        and input_zero_point == output_zero_point
        and in_type == out_type
    ):
        cwr.add_line(f"typedef DequantQuantEqual<{out_type}> {alias};")
        return

    log2_ratio = np.log2(ratio)
    rounded = np.round(log2_ratio)
    if np.isclose(log2_ratio, rounded, atol=atol):
        shift = int(-rounded)
        cwr.add_line(f"const int {alias}_SHIFT = {shift};")
        if input_zero_point == 0:
            cwr.add_line(
                f"typedef DequantQuantPo2<{alias}_SHIFT, {in_type}, {out_type}, {output_zero_point}> {alias};"
            )
        else:
            cwr.add_line(
                f"typedef RequantizePo2<{alias}_SHIFT, {input_zero_point}, {output_zero_point}, {in_type}, {out_type}> {alias};"
            )
        return

    frac = Fraction(ratio).limit_denominator(max_den)
    cwr.add_line(f"const int {alias}_NUM = {int(frac.numerator)};")
    cwr.add_line(f"const int {alias}_DEN = {int(frac.denominator)};")
    if input_zero_point == 0:
        cwr.add_line(
            f"typedef DequantQuantFloat<{alias}_NUM, {alias}_DEN, {in_type}, {out_type}, {output_zero_point}> {alias};"
        )
    else:
        cwr.add_line(
            f"typedef RequantizeFloat<{alias}_NUM, {alias}_DEN, {input_zero_point}, {output_zero_point}, {in_type}, {out_type}> {alias};"
        )
