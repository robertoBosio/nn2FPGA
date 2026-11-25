from fractions import Fraction
from typing import Optional

MAX_DEN = 1 << 20
REL_TOL = 1e-3
MAX_SHIFT = 31


def ratio_to_fraction(ratio: float, max_den: int = MAX_DEN) -> tuple[int, int]:
    frac = Fraction(float(ratio)).limit_denominator(max_den)
    return int(frac.numerator), int(frac.denominator)


def ratio_to_shift_form(
    ratio: float,
    max_den: int = MAX_DEN,
    rel_tol: float = REL_TOL,
    max_shift: int = MAX_SHIFT,
) -> Optional[tuple[int, int]]:
    num, den = ratio_to_fraction(ratio, max_den=max_den)
    exact = Fraction(num, den)
    for shift in range(max_shift + 1):
        n = int(Fraction(num << shift, den).__round__())
        if n <= 0:
            continue
        if n >= 1 << 31:
            break
        if abs(Fraction(n, 1 << shift) - exact) / exact <= rel_tol:
            return n, shift
    return None
