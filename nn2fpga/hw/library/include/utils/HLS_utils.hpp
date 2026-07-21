#pragma once
#include "ap_int.h"
#include "ap_float.h"
#include <cstddef>

// ---------------------------------------------------------------------------
// bits_for: number of bits needed to represent values in [0 .. n-1]
// ---------------------------------------------------------------------------
static constexpr int bits_for(size_t n) {
  int b = 0;
  size_t v = (n > 0) ? (n - 1) : 0;
  while (v) {
    v >>= 1;
    ++b;
  }
  return (b == 0) ? 1 : b; // at least 1 bit
}

// ---------------------------------------------------------------------------
// data_width<T>: static bit-width of an HLS arithmetic type
//
//   ap_int<N>, ap_uint<N>, ap_fixed<W,...>  →  T::width  (built-in)
//   ap_float<E,M>                           →  1 + E + M (sign+exp+mant)
// ---------------------------------------------------------------------------
template <typename T>
struct data_width {
  static constexpr size_t value = T::width;
};

template <int W, int E>
struct data_width<ap_float<W, E>> {
  static constexpr size_t value = W;
};

template <typename T>
inline constexpr size_t data_width_v = data_width<T>::value;

// ---------------------------------------------------------------------------
// get_raw_bits(v): return the raw ap_uint bit-pattern of any HLS type
//
//   ap_int / ap_uint / ap_fixed  →  identity (range() already works)
//   ap_float<W,E>                →  v.bits
// ---------------------------------------------------------------------------
template <typename T>
struct raw_bits {
  static auto get(const T &v) { return v; }
};

template <int W, int E>
struct raw_bits<ap_float<W, E>> {
  static auto get(const ap_float<W, E> &v) { return v.bits_ref(); }
};

template <typename T>
auto get_raw_bits(const T &v) { return raw_bits<T>::get(v); }

// ---------------------------------------------------------------------------
// from_raw_bits<T>: reconstruct a T from a raw ap_uint bit-pattern
//
//   ap_int / ap_uint / ap_fixed  →  direct construction from bits
//   ap_float<W,E>                →  assign via v.bits
// ---------------------------------------------------------------------------
template <typename T>
struct from_raw_bits {
  static T set(const ap_uint<data_width_v<T>> &bits) { return T(bits); }
};

template <int W, int E>
struct from_raw_bits<ap_float<W, E>> {
  static ap_float<W, E> set(const ap_uint<W> &bits) {
    ap_float<W, E> v;
    v.bits_ref() = bits;
    return v;
  }
};

template <typename T> T set_raw_bits(const ap_uint<data_width_v<T>> &bits) {
  return from_raw_bits<T>::set(bits);
}