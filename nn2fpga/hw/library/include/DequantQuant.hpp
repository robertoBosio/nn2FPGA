#pragma once
#include "ap_int.h"
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


template <int Shift, typename TAcc, typename TOut> struct DequantQuantPo2 {

  TOut operator()(TAcc acc) const {
#pragma HLS inline
    return run(acc, std::integral_constant<bool, (Shift > 0)>{});
  }

private:
  static TOut run(TAcc acc, std::true_type) {

    // Absolute value in unsigned to get a true remainder magnitude.
    const bool neg = acc.sign();
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

    // Saturate to OUT_WIDTH
    if (rounded > LimitsImpl<TOut>::max()) {
      rounded = LimitsImpl<TOut>::max();
    }
    if (is_ap_signed<TAcc>::value) {
      if (rounded < LimitsImpl<TOut>::min()) {
        rounded = LimitsImpl<TOut>::min();
      }
    }
    return TOut(rounded);
  }

  static TOut run(TAcc acc, std::false_type) {
    constexpr int L = (-Shift); // left shift amount (>= 0)
    constexpr bool need_signed =
        is_ap_signed<TAcc>::value || is_ap_signed<TOut>::value;

    // Widen enough to prevent wrap-around during the shift.
    // For signed, add +1 for sign growth safety.
    using TWide = typename std::conditional<need_signed,
                                            ap_int<TAcc::width + L + 1>,
                                            ap_uint<TAcc::width + L>>::type;

    TWide wide = TWide(acc);
    wide = wide << L;

    const TWide wmax = TWide(LimitsImpl<TOut>::max());
    const TWide wmin = TWide(LimitsImpl<TOut>::min());

    if (wide > wmax)
      wide = wmax;
    if (wide < wmin)
      wide = wmin;

    return TOut(wide);
  }
};

template <typename T>
struct DequantQuantEqual
{
    T operator()(T acc) const
    {
#pragma HLS inline
        return acc;
    }
};
