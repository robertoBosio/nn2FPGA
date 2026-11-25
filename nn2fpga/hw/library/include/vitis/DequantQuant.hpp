#pragma once
#include "ap_int.h"
#include "ap_float.h"
#include "hls_math.h"
#include <type_traits>

template <typename TOut,
          bool Signed =
              std::is_same<typename TOut::Base::Base,
                           _AP_ROOT_TYPE<TOut::Base::width, true>>::value>
struct LimitsImpl;

template <typename TOut> struct LimitsImpl<TOut, true> {
  static inline TOut max() { return TOut(~(TOut(1) << (TOut::width - 1))); }
  static inline TOut min() { return (TOut(1) << (TOut::width - 1)); }
};

template <typename TOut> struct LimitsImpl<TOut, false> {
  static inline TOut max() { return TOut(-1); }
  static inline TOut min() { return TOut(0); }
};

template <typename T> struct is_ap_signed {
  static constexpr bool value =
      std::is_same<typename T::Base::Base,
                   _AP_ROOT_TYPE<T::Base::width, true>>::value;
};

/* A utility function to compute the number of bits needed to represent a value. */
constexpr int const_bit_width(unsigned x) {
  int bits = 0;
  do {
    ++bits;
    x >>= 1;
  } while (x != 0);
  return bits;
}

template <typename TAcc>
inline bool is_negative_value(TAcc, std::false_type) {
#pragma HLS inline
  return false;
}

template <typename TAcc>
inline bool is_negative_value(TAcc acc, std::true_type) {
#pragma HLS inline
  return acc.sign();
}

template <int Shift, typename TAcc, typename TOut, int ZeroPoint = 0>
struct DequantQuantPo2 {

  TOut operator()(TAcc acc) const {
#pragma HLS inline
    return run(acc, std::integral_constant<bool, (Shift > 0)>{});
  }

private:
  static TOut run(TAcc acc, std::true_type) {
    constexpr int WideBits =
        (TAcc::width > TOut::width ? TAcc::width : TOut::width) + 2;
    using TWide = ap_int<WideBits>;

    // Absolute value in unsigned to get a true remainder magnitude.
    const bool neg = is_negative_value(acc, std::integral_constant<bool, is_ap_signed<TAcc>::value>{});
    ap_uint<TAcc::width> uacc = hls::abs(acc);

    const ap_uint<Shift> half = ap_uint<Shift>(1) << (Shift - 1);
    const ap_uint<Shift> rem = uacc.range(Shift - 1, 0);
    const ap_uint<TAcc::width> trunc = uacc >> Shift;

    // Round-to-nearest, ties-to-even on the magnitude
    ap_uint<TAcc::width> rounded_mag = trunc;
    if (rem > half || (rem == half && (trunc & 1))) {
      rounded_mag = trunc + 1;
    }

    // Restore sign
    TAcc rounded;
    if (is_ap_signed<TAcc>::value) {
      rounded = neg ? TAcc(-rounded_mag) : TAcc(rounded_mag);
    } else {
      rounded = TAcc(rounded_mag);
    }

    TWide shifted = TWide(rounded) + TWide(ZeroPoint);

    // Saturate to OUT_WIDTH
    if (shifted > TWide(LimitsImpl<TOut>::max())) {
      shifted = TWide(LimitsImpl<TOut>::max());
    }
    if (shifted < TWide(LimitsImpl<TOut>::min())) {
      shifted = TWide(LimitsImpl<TOut>::min());
    }
    return TOut(shifted);
  }

  static TOut run(TAcc acc, std::false_type) {
    constexpr int L = (-Shift); // left shift amount (>= 0)
    constexpr bool need_signed =
        is_ap_signed<TAcc>::value || is_ap_signed<TOut>::value;

    // Widen enough to prevent wrap-around during the shift.
    // For signed, add +1 for sign growth safety.
    using TWide =
        typename std::conditional<need_signed, ap_int<TAcc::width + L + 1>,
                                  ap_uint<TAcc::width + L>>::type;

    TWide wide = TWide(acc);
    wide = wide << L;
    wide += TWide(ZeroPoint);

    const TWide wmax = TWide(LimitsImpl<TOut>::max());
    const TWide wmin = TWide(LimitsImpl<TOut>::min());

    if (wide > wmax)
      wide = wmax;
    if (wide < wmin)
      wide = wmin;

    return TOut(wide);
  }
};

// ---------------------------------------------------------------------------
// DequantQuantFloat
//
// Despite the name there is no floating point arithmetic in here: the functor
// evaluates an exact integer rational scaling of the accumulator,
//
//     y = clamp( round_half_even( acc * Num / Den ) + ZeroPoint )
//
// where Num / Den is the (already quantised) ratio of the surrounding
// dequantize/quantize scales,
//
//     Num / Den ~= input_scale * weights_scale / output_scale
//
// Pipeline:
//   1. abs(acc) and fold the sign out, so the division runs on magnitudes.
//   2. prod = |acc| * Num            (constant multiply -> shift/add tree)
//   3. quotient / remainder wrt Den  (constant divide/modulo -> expensive)
//   4. round half to even on the rational quotient
//   5. restore the sign, add ZeroPoint, saturate to TOut
//
// Cost note: steps 3 and 4 are the reason this functor is far from free.
// For a non-power-of-two Den, Vitis HLS strength-reduces `prod / Den` into a
// constant multiply of width ProductBits x bits(ceil(2^k / Den)), which maps
// onto ceil(P/27) * ceil(D/18) DSP48E2 slices, and keeps `prod % Den` as a
// dedicated (DSP-free but LUT/FF heavy) divider core. A power-of-two Den
// would collapse both into a shift and a mask.
// ---------------------------------------------------------------------------
template <int Num, int Den, typename TAcc, typename TOut, int ZeroPoint = 0>
struct DequantQuantFloat {
  static_assert(Num > 0, "Num must be positive");
  static_assert(Den > 0, "Den must be positive");

  TOut operator()(TAcc acc) const {
#pragma HLS inline

    // Bit widths: NumBits = ceil(log2(Num)) + 1, ProductBits holds the full
    // |acc| * Num product, WideBits adds headroom for the negation, the zero
    // point and the subsequent saturation comparisons.
    constexpr int NumBits = const_bit_width(Num);
    constexpr int ProductBits = TAcc::width + NumBits;
    constexpr int WideBits = ProductBits + 2;

    // Sign is handled separately: everything up to step 4 works on the
    // magnitude, which lets the divide/modulo below be unsigned.
    const bool neg = acc.sign();
    ap_uint<TAcc::width> uacc = hls::abs(acc);
    ap_uint<ProductBits> prod = ap_uint<ProductBits>(uacc) * ap_uint<NumBits>(Num);

    // Exact rational division by the constant Den. HLS turns the division into
    // a constant multiply (the DSP cost) and keeps the modulo as a divider.
    ap_uint<ProductBits> quotient = prod / Den;
    ap_uint<ProductBits> remainder = prod % Den;

    // Round half to even on remainder/Den: compare 2*remainder against Den so
    // no division is needed for the tie test itself.
    ap_uint<ProductBits + 1> double_remainder = ap_uint<ProductBits + 1>(remainder) << 1;
    ap_uint<ProductBits> rounded_mag = quotient;
    if (double_remainder > Den || (double_remainder == Den && (quotient & 1))) {
      rounded_mag = quotient + 1;
    }

    // Restore the sign on the rounded magnitude and re-apply the output zero
    // point. WideBits keeps both steps wrap-around free.
    ap_int<WideBits> shifted;
    if (is_ap_signed<TAcc>::value && neg) {
      shifted = -ap_int<WideBits>(rounded_mag);
    } else {
      shifted = ap_int<WideBits>(rounded_mag);
    }
    shifted += ap_int<WideBits>(ZeroPoint);

    // Saturate (never wrap) to the output type's range before casting down.
    const ap_int<WideBits> lo = ap_int<WideBits>(LimitsImpl<TOut>::min());
    const ap_int<WideBits> hi = ap_int<WideBits>(LimitsImpl<TOut>::max());
    if (shifted < lo)
      shifted = lo;
    if (shifted > hi)
      shifted = hi;

    return TOut(shifted);
  }
};

// ---------------------------------------------------------------------------
// DequantQuantFloatShift
//
// Same contract as DequantQuantFloat, but the scale ratio is restricted to a
// power-of-two denominator:
//
//     y = clamp( round_half_even( acc * Num / 2^Shift ) + ZeroPoint )
//
// Num / 2^Shift still approximates input_scale * weights_scale / output_scale.
// Here only the denominator is forced to a power of two, so the division
// collapses to a shift while the numerator keeps the scale accurate.
//
// Relative to DequantQuantFloat this removes both expensive operations:
//     prod / Den  ->  prod >> Shift
//     prod % Den  ->  prod & (2^Shift - 1)
// The rounding rule, sign handling, zero point and saturation are identical to
// DequantQuantFloat, so the two are directly comparable element by element.
// ---------------------------------------------------------------------------
template <int Num, int Shift, typename TAcc, typename TOut, int ZeroPoint = 0>
struct DequantQuantFloatShift {
  static_assert(Num > 0, "Num must be positive");
  static_assert(Shift >= 0, "Shift must be non-negative");

  static constexpr int NumBits = const_bit_width(Num);
  static constexpr int ProductBits = TAcc::width + NumBits;
  static constexpr int WideBits = ProductBits + 2;

  TOut operator()(TAcc acc) const {
#pragma HLS inline

    const bool neg = acc.sign();
    ap_uint<TAcc::width> uacc = hls::abs(acc);
    ap_uint<ProductBits> prod = ap_uint<ProductBits>(uacc) * ap_uint<NumBits>(Num);

    const ap_uint<ProductBits> rounded_mag =
        rounded_magnitude(prod, std::integral_constant<bool, (Shift > 0)>{});

    ap_int<WideBits> shifted;
    if (is_ap_signed<TAcc>::value && neg) {
      shifted = -ap_int<WideBits>(rounded_mag);
    } else {
      shifted = ap_int<WideBits>(rounded_mag);
    }
    shifted += ap_int<WideBits>(ZeroPoint);

    // Saturate (never wrap) to the output type's range before casting down.
    const ap_int<WideBits> lo = ap_int<WideBits>(LimitsImpl<TOut>::min());
    const ap_int<WideBits> hi = ap_int<WideBits>(LimitsImpl<TOut>::max());
    if (shifted < lo)
      shifted = lo;
    if (shifted > hi)
      shifted = hi;

    return TOut(shifted);
  }

private:
  // Round half to even on prod / 2^Shift. Deliberately the same test shape as
  // DequantQuantFloat uses for 2*remainder vs Den: compare the Shift low bits
  // against half of 2^Shift, break ties towards an even quotient.
  static ap_uint<ProductBits> rounded_magnitude(ap_uint<ProductBits> prod,
                                                std::true_type) {
    // Truncating to Shift bits yields exactly the low-order remainder, and it
    // also degrades safely to the full product once Shift exceeds ProductBits.
    const ap_uint<Shift> rem = prod;
    const ap_uint<Shift> half = ap_uint<Shift>(1) << (Shift - 1);
    const ap_uint<ProductBits> trunc =
        (Shift >= ProductBits) ? ap_uint<ProductBits>(0)
                               : ap_uint<ProductBits>(prod >> Shift);

    if (rem > half || (rem == half && (trunc & 1))) {
      return ap_uint<ProductBits>(trunc + 1);
    }
    return trunc;
  }

  // Shift == 0 means the ratio is a whole number; prod is already exact.
  static ap_uint<ProductBits> rounded_magnitude(ap_uint<ProductBits> prod,
                                                std::false_type) {
    return prod;
  }
};

template <int ZeroPoint, typename TIn, typename TAcc>
struct DequantAlignEqual {

  TAcc operator()(TIn in) const {
#pragma HLS inline
    constexpr int WideBits = (TIn::width > TAcc::width ? TIn::width : TAcc::width) + 1;
    using TWide = ap_int<WideBits>;
    return TAcc(TWide(in) - TWide(ZeroPoint));
  }
};

template <int Shift, int ZeroPoint, typename TIn, typename TAcc>
struct DequantAlignPo2 {

  TAcc operator()(TIn in) const {
#pragma HLS inline
    constexpr int WideBits = (TIn::width > TAcc::width ? TIn::width : TAcc::width) + 1;
    using TWide = ap_int<WideBits>;
    DequantQuantPo2<Shift, TAcc, TAcc> quantizer;
    const TAcc centered = TAcc(TWide(in) - TWide(ZeroPoint));
    return quantizer(centered);
  }
};

template <int Num, int Den, int ZeroPoint, typename TIn, typename TAcc>
struct DequantAlignFloat {

  TAcc operator()(TIn in) const {
#pragma HLS inline
    constexpr int WideBits = (TIn::width > TAcc::width ? TIn::width : TAcc::width) + 1;
    using TWide = ap_int<WideBits>;
    DequantQuantFloat<Num, Den, TAcc, TAcc> quantizer;
    const TAcc centered = TAcc(TWide(in) - TWide(ZeroPoint));
    return quantizer(centered);
  }
};

template <typename TIn, int Shift> struct DequantPo2ToFloat {
  ap_float<32, 8> operator()(TIn acc) const {
#pragma HLS inline
    ap_float<32, 8> val = ap_float<32, 8>(acc);
    ap_uint<8> exp = val.exponent_ref();
    exp -= Shift;
    val.exponent_ref() = exp;
    return val;
  }
};

template <int InputZeroPoint, int OutputZeroPoint, typename TIn, typename TOut>
struct RequantizeEqual {
  TOut operator()(TIn in) const {
#pragma HLS inline
    constexpr int WideBits = (TIn::width > TOut::width ? TIn::width : TOut::width) + 1;
    using TWide = ap_int<WideBits>;

    TWide centered = TWide(in) - TWide(InputZeroPoint);
    centered += TWide(OutputZeroPoint);

    const TWide wmax = TWide(LimitsImpl<TOut>::max());
    const TWide wmin = TWide(LimitsImpl<TOut>::min());

    if (centered > wmax)
      centered = wmax;
    if (centered < wmin)
      centered = wmin;

    return TOut(centered);
  }
};

template <int Shift, int InputZeroPoint, int OutputZeroPoint, typename TIn,
          typename TOut>
struct RequantizePo2 {
  TOut operator()(TIn in) const {
#pragma HLS inline
    constexpr int WideBits = (TIn::width > TOut::width ? TIn::width : TOut::width) + 1;
    using TWide = ap_int<WideBits>;
    using TCentered = ap_int<WideBits>;
    DequantQuantPo2<Shift, TCentered, TOut, OutputZeroPoint> quantizer;
    const TCentered centered = TCentered(TWide(in) - TWide(InputZeroPoint));
    return quantizer(centered);
  }
};

template <int Num, int Den, int InputZeroPoint, int OutputZeroPoint,
          typename TIn, typename TOut>
struct RequantizeFloat {
  TOut operator()(TIn in) const {
#pragma HLS inline
    constexpr int WideBits = (TIn::width > TOut::width ? TIn::width : TOut::width) + 1;
    using TWide = ap_int<WideBits>;
    using TCentered = ap_int<WideBits>;
    DequantQuantFloat<Num, Den, TCentered, TOut, OutputZeroPoint> quantizer;
    const TCentered centered = TCentered(TWide(in) - TWide(InputZeroPoint));
    return quantizer(centered);
  }
};

template <typename T> struct DequantQuantEqual {
  T operator()(T acc) const {
#pragma HLS inline
    return acc;
  }
};
