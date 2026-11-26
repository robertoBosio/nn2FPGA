#pragma once

#include <ac_channel.h>
#include <cstddef>

template <typename TWord, size_t DIM0, size_t DIM1, size_t DIM2,
          size_t DIM1_UNROLL, size_t DIM2_UNROLL>
class StreamingTensorDuplicator {
  static_assert(DIM0 > 0 && DIM1 > 0 && DIM2 > 0,
                "dimensions must be greater than 0");
  static_assert(DIM1_UNROLL > 0 && DIM2_UNROLL > 0,
                "unroll factors must be greater than 0");
  static_assert(DIM1 % DIM1_UNROLL == 0,
                "DIM1 must be divisible by DIM1_UNROLL");
  static_assert(DIM2 % DIM2_UNROLL == 0,
                "DIM2 must be divisible by DIM2_UNROLL");

public:
  template <size_t HLS_TAG>
  void run(ac_channel<TWord> i_data[DIM1_UNROLL],
           ac_channel<TWord> o_data0[DIM1_UNROLL],
           ac_channel<TWord> o_data1[DIM1_UNROLL]) {
    for (size_t i = 0;
         i < DIM0 * DIM1 * DIM2 / (DIM1_UNROLL * DIM2_UNROLL); ++i) {
      for (size_t lane = 0; lane < DIM1_UNROLL; ++lane) {
        const TWord word = i_data[lane].read();
        o_data0[lane].write(word);
        o_data1[lane].write(word);
      }
    }
  }
};
