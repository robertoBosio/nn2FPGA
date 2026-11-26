#pragma once

#include <ac_int.h>
#include <cstddef>
#include <type_traits>

template <unsigned X> struct CatapultConstBitWidth {
  static constexpr int value = 1 + CatapultConstBitWidth<(X >> 1)>::value;
};

template <> struct CatapultConstBitWidth<0> { static constexpr int value = 0; };

template <typename TOut, bool Signed = TOut::sign> struct CatapultLimits;

template <typename TOut> struct CatapultLimits<TOut, true> {
  static TOut max() {
    return TOut((ac_int<TOut::width, false>(1) << (TOut::width - 1)) - 1);
  }

  static TOut min() {
    return TOut(ac_int<TOut::width, false>(1) << (TOut::width - 1));
  }
};

template <typename TOut> struct CatapultLimits<TOut, false> {
  static TOut max() { return TOut(-1); }
  static TOut min() { return TOut(0); }
};

template <typename TAcc, typename TOut, int WideBits>
inline TOut catapult_saturate(ac_int<WideBits, true> value) {
  const ac_int<WideBits, true> lo = ac_int<WideBits, true>(CatapultLimits<TOut>::min());
  const ac_int<WideBits, true> hi = ac_int<WideBits, true>(CatapultLimits<TOut>::max());
  if (value < lo)
    value = lo;
  if (value > hi)
    value = hi;
  return TOut(value);
}

template <int Shift, typename TAcc, typename TOut, int ZeroPoint = 0>
struct DequantQuantPo2 {
  TOut operator()(TAcc acc) const { return run(acc, std::integral_constant<bool, (Shift > 0)>()); }

private:
  static TOut run(TAcc acc, std::true_type) {
    enum { WideBits = (TAcc::width > TOut::width ? TAcc::width : TOut::width) + 2 };
    using TMag = ac_int<TAcc::width + 1, false>;
    using TWide = ac_int<WideBits, true>;

    const TWide wide_acc = TWide(acc);
    const bool negative = TAcc::sign && wide_acc < 0;
    const TMag magnitude = negative ? TMag(-wide_acc) : TMag(wide_acc);
    const ac_int<Shift, false> half = ac_int<Shift, false>(1) << (Shift - 1);
    const ac_int<Shift, false> remainder = magnitude;
    TMag rounded_magnitude = magnitude >> Shift;
    if (remainder > half || (remainder == half && (rounded_magnitude & 1)))
      rounded_magnitude += 1;

    TWide scaled = TWide(rounded_magnitude);
    if (negative)
      scaled = -scaled;
    scaled += TWide(ZeroPoint);
    return catapult_saturate<TAcc, TOut>(scaled);
  }

  static TOut run(TAcc acc, std::false_type) {
    enum { LeftShift = -Shift };
    enum { WideBits = (TAcc::width > TOut::width ? TAcc::width : TOut::width) + LeftShift + 2 };
    ac_int<WideBits, true> scaled = ac_int<WideBits, true>(acc);
    scaled <<= LeftShift;
    scaled += ZeroPoint;
    return catapult_saturate<TAcc, TOut>(scaled);
  }
};

template <int Num, int Den, typename TAcc, typename TOut, int ZeroPoint = 0>
struct DequantQuantFloat {
  static_assert(Num > 0, "Num must be positive");
  static_assert(Den > 0, "Den must be positive");

  TOut operator()(TAcc acc) const {
    enum { NumBits = CatapultConstBitWidth<Num>::value };
    enum { ProductBits = TAcc::width + NumBits + 1 };
    enum { WideBits = ProductBits + 2 };
    using TMag = ac_int<TAcc::width + 1, false>;
    using TProduct = ac_int<ProductBits, false>;
    using TWide = ac_int<WideBits, true>;

    const TWide wide_acc = TWide(acc);
    const bool negative = TAcc::sign && wide_acc < 0;
    const TMag magnitude = negative ? TMag(-wide_acc) : TMag(wide_acc);
    const TProduct product = TProduct(magnitude) * TProduct(Num);
    const TProduct quotient = product / Den;
    const TProduct remainder = product % Den;
    TProduct rounded_magnitude = quotient;
    if ((remainder << 1) > Den || ((remainder << 1) == Den && (quotient & 1)))
      rounded_magnitude += 1;

    TWide scaled = TWide(rounded_magnitude);
    if (negative)
      scaled = -scaled;
    scaled += ZeroPoint;
    return catapult_saturate<TAcc, TOut>(scaled);
  }
};

template <int Num, int Shift, typename TAcc, typename TOut, int ZeroPoint = 0>
struct DequantQuantFloatShift {
  static_assert(Num > 0, "Num must be positive");
  static_assert(Shift >= 0, "Shift must be non-negative");

  TOut operator()(TAcc acc) const {
    enum { NumBits = CatapultConstBitWidth<Num>::value };
    enum { ProductBits = TAcc::width + NumBits + 1 };
    enum { WideBits = ProductBits + 2 };
    using TMag = ac_int<TAcc::width + 1, false>;
    using TProduct = ac_int<ProductBits, false>;
    using TWide = ac_int<WideBits, true>;

    const TWide wide_acc = TWide(acc);
    const bool negative = TAcc::sign && wide_acc < 0;
    const TMag magnitude = negative ? TMag(-wide_acc) : TMag(wide_acc);
    const TProduct product = TProduct(magnitude) * TProduct(Num);
    const TProduct rounded_magnitude = round(product, std::integral_constant<bool, (Shift > 0)>());

    TWide scaled = TWide(rounded_magnitude);
    if (negative)
      scaled = -scaled;
    scaled += ZeroPoint;
    return catapult_saturate<TAcc, TOut>(scaled);
  }

private:
  enum { NumBits = CatapultConstBitWidth<Num>::value };
  enum { ProductBits = TAcc::width + NumBits + 1 };
  using TProduct = ac_int<ProductBits, false>;

  static TProduct round(TProduct product, std::true_type) {
    const ac_int<Shift, false> half = ac_int<Shift, false>(1) << (Shift - 1);
    const ac_int<Shift, false> remainder = product;
    TProduct truncated = product >> Shift;
    if (remainder > half || (remainder == half && (truncated & 1)))
      truncated += 1;
    return truncated;
  }

  static TProduct round(TProduct product, std::false_type) { return product; }
};

template <int Shift, int InputZeroPoint, int OutputZeroPoint, typename TIn,
          typename TOut>
struct RequantizePo2 {
  TOut operator()(TIn input) const {
    using TAcc = ac_int<TIn::width + 1, true>;
    return DequantQuantPo2<Shift, TAcc, TOut, OutputZeroPoint>()(
        TAcc(input) - InputZeroPoint);
  }
};

template <int Num, int Den, int InputZeroPoint, int OutputZeroPoint,
          typename TIn, typename TOut>
struct RequantizeFloat {
  TOut operator()(TIn input) const {
    using TAcc = ac_int<TIn::width + 1, true>;
    return DequantQuantFloat<Num, Den, TAcc, TOut, OutputZeroPoint>()(
        TAcc(input) - InputZeroPoint);
  }
};

template <int ZeroPoint, typename TIn, typename TAcc> struct DequantAlignEqual {
  TAcc operator()(TIn input) const {
    enum { WideBits = (TIn::width > TAcc::width ? TIn::width : TAcc::width) + 1 };
    return TAcc(ac_int<WideBits, true>(input) - ZeroPoint);
  }
};

template <int Shift, int ZeroPoint, typename TIn, typename TAcc> struct DequantAlignPo2 {
  TAcc operator()(TIn input) const {
    enum { WideBits = (TIn::width > TAcc::width ? TIn::width : TAcc::width) + 1 };
    const TAcc centered = TAcc(ac_int<WideBits, true>(input) - ZeroPoint);
    return DequantQuantPo2<Shift, TAcc, TAcc>()(centered);
  }
};

template <int Num, int Den, int ZeroPoint, typename TIn, typename TAcc>
struct DequantAlignFloat {
  TAcc operator()(TIn input) const {
    enum { WideBits = (TIn::width > TAcc::width ? TIn::width : TAcc::width) + 1 };
    const TAcc centered = TAcc(ac_int<WideBits, true>(input) - ZeroPoint);
    return DequantQuantFloat<Num, Den, TAcc, TAcc>()(centered);
  }
};

template <typename T> struct DequantQuantEqual {
  T operator()(T acc) const { return acc; }
};
