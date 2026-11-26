#pragma once

#include <ac_channel.h>
#include <cstddef>

template <typename TInputWord, typename TOutputWord, typename TOutput,
          typename Quantizer, size_t DATA_PER_WORD, size_t DIM0, size_t DIM1,
          size_t DIM2, size_t DIM1_UNROLL, size_t DIM2_UNROLL,
          bool ONLY_ONCE = false>
class AXISToStream {
  static constexpr size_t ITER =
      DIM0 * DIM1 * DIM2 / (DIM1_UNROLL * DIM2_UNROLL);

  static_assert(DATA_PER_WORD > 0, "DATA_PER_WORD must be positive");
  static_assert(DIM1_UNROLL > 0 && DIM2_UNROLL > 0,
                "parallelism factors must be positive");
  static_assert(DIM1_UNROLL == 1 || DIM2 == DIM2_UNROLL,
                "DIM2 must equal DIM2_UNROLL when DIM1_UNROLL is greater than one");
  static_assert(DIM2 % DIM2_UNROLL == 0,
                "DIM2 must divide DIM2_UNROLL");
  static_assert(DIM1 % DIM1_UNROLL == 0,
                "DIM1 must divide DIM1_UNROLL");

public:
  template <size_t HLS_TAG>
  void run(ac_channel<TInputWord> &i_data,
           ac_channel<TOutputWord> o_data[DIM1_UNROLL]) {
    TOutput circular_buffer[DATA_PER_WORD * 2];
    size_t head = 0;
    size_t tail = 0;
    size_t size = 0;
    static bool only_once_flag = false;

    if (!only_once_flag) {
      for (size_t i_output_word = 0; i_output_word < ITER; i_output_word++) {
        if (size < DIM1_UNROLL * DIM2_UNROLL) {
          TInputWord input_data = i_data.read();
          for (size_t i = 0; i < DATA_PER_WORD; i++) {
            circular_buffer[head] = TOutput(
                input_data.data.template slc<TOutput::width>(
                    static_cast<unsigned>(TOutput::width * i)));
            head = (head + 1) % (DATA_PER_WORD * 2);
          }
          size += DATA_PER_WORD;
        }

        Quantizer quantizer;
        for (size_t w = 0; w < DIM1_UNROLL; w++) {
          TOutputWord output_data;
          for (size_t ch = 0; ch < DIM2_UNROLL; ch++) {
            output_data[ch] = quantizer(circular_buffer[tail]);
            tail = (tail + 1) % (DATA_PER_WORD * 2);
          }
          o_data[w].write(output_data);
        }
        size -= DIM1_UNROLL * DIM2_UNROLL;
      }
    }
    if (ONLY_ONCE)
      only_once_flag = true;
  }
};
