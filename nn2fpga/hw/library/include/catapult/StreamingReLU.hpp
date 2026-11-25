#pragma once

#include <ac_channel.h>
#include <cstddef>

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t DIM0, size_t DIM1,
          size_t DIM2, size_t DIM1_UNROLL, size_t DIM2_UNROLL>
class StreamingReLU {
  static_assert(DIM2 % DIM2_UNROLL == 0,
                "DIM2 must be a multiple of DIM2_UNROLL");
  static_assert(DIM2_UNROLL > 0, "DIM2_UNROLL must be greater than 0");
  static_assert(DIM1 % DIM1_UNROLL == 0,
                "DIM1 must be a multiple of DIM1_UNROLL");
  static_assert(DIM1_UNROLL > 0, "DIM1_UNROLL must be greater than 0");
  static_assert(DIM0 > 0 && DIM1 > 0, "DIM0 and DIM1 must be greater than 0");

public:
  template <size_t HLS_TAG>
  void run(ac_channel<TInputWord> i_data[DIM1_UNROLL],
           ac_channel<TOutputWord> o_data[DIM1_UNROLL]) {
    for (size_t i_hw = 0; i_hw < DIM0 * DIM1 / DIM1_UNROLL; i_hw++) {
    STREAMINGRELU_RUN_LOOP:
      for (size_t i_ch = 0; i_ch < DIM2 / DIM2_UNROLL; i_ch++) {
        Quantizer quantizer;
        for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL;
             i_dim1_par++) {
          TInputWord input_word = i_data[i_dim1_par].read();
          TOutputWord output_word;
          for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL;
               i_dim2_par++) {
            TInput output_value = input_word[i_dim2_par] < 0
                                      ? TInput(0)
                                      : input_word[i_dim2_par];
            output_word[i_dim2_par] = quantizer(output_value);
          }
          o_data[i_dim1_par].write(output_word);
        }
      }
    }
  }
};
