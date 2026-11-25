#pragma once

#include <ac_channel.h>
#include <cstddef>

template <typename TInputWordA, typename TInputA, typename TInputWordB,
          typename TInputB, typename TOutputWord, typename TOutput,
          typename TAcc, typename OutputTransform, typename AlignA,
          typename AlignB, size_t DIM0, size_t DIM1, size_t DIM2,
          size_t DIM1_UNROLL, size_t DIM2_UNROLL>
class StreamingAdd {
  static_assert(DIM1 % DIM1_UNROLL == 0,
                "DIM1 must be divisible by DIM1_UNROLL");
  static_assert(DIM2 % DIM2_UNROLL == 0,
                "DIM2 must be divisible by DIM2_UNROLL");
  static_assert(DIM0 > 0 && DIM1 > 0 && DIM2 > 0,
                "dimensions must be greater than 0");
  static_assert(DIM1_UNROLL > 0 && DIM2_UNROLL > 0,
                "unroll factors must be greater than 0");

public:
  template <size_t HLS_TAG>
  void run(ac_channel<TInputWordA> i_dataA[DIM1_UNROLL],
           ac_channel<TInputWordB> i_dataB[DIM1_UNROLL],
           ac_channel<TOutputWord> o_data[DIM1_UNROLL]) {
    for (size_t i_dim012 = 0;
         i_dim012 < DIM0 * DIM1 * DIM2 / (DIM2_UNROLL * DIM1_UNROLL);
         i_dim012++) {
      TInputWordA inputA_word;
      TInputWordB inputB_word;
      TOutputWord output_word;
      OutputTransform output_transform;
      AlignA align_a;
      AlignB align_b;

      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        inputA_word = i_dataA[i_dim1_par].read();
        inputB_word = i_dataB[i_dim1_par].read();

        for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL; i_dim2_par++) {
          TAcc sum = align_a(inputA_word[i_dim2_par]) +
                     align_b(inputB_word[i_dim2_par]);
          output_word[i_dim2_par] = output_transform(sum);
        }
        o_data[i_dim1_par].write(output_word);
      }
    }
  }
};
