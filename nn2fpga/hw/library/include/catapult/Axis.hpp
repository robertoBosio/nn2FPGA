#pragma once

#include <ac_int.h>

template <unsigned DataWidth> struct AxisWord {
  ac_int<DataWidth, false> data;
  ac_int<DataWidth / 8, false> keep;
  ac_int<DataWidth / 8, false> strb;
  ac_int<1, false> last;
};
