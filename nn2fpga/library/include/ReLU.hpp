#pragma once
#include "ap_int.h"
#include "hls_math.h"
#include <type_traits>

template <typename T> struct ReLU {
  T operator()(T acc) const {
#pragma HLS inline
    if (acc < 0)
      return T(0);
    else
      return acc;
  }
};