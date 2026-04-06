#pragma once
#include "ap_int.h"
#include <cstddef>

static constexpr int bits_for(size_t n) {
  // bits to represent values [0 .. n-1]
  int b = 0;
  size_t v = (n > 0) ? (n - 1) : 0;
  while (v) {
    v >>= 1;
    ++b;
  }
  return (b == 0) ? 1 : b; // at least 1 bit
}