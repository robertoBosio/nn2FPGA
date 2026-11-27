#pragma once

#include <ac_channel.h>
#include <cstddef>

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename Quantizer, size_t ITER, size_t DATA_PER_WORD, size_t DIM0,
          size_t DIM1, size_t DIM2, size_t DIM1_UNROLL, size_t DIM2_UNROLL>
class StreamToAXIS {
  static constexpr size_t TOTAL_ELEMS = DIM0 * DIM1 * DIM2;
  static constexpr size_t READS = TOTAL_ELEMS / (DIM1_UNROLL * DIM2_UNROLL);
  static constexpr size_t OUTPUT_WORDS =
      (TOTAL_ELEMS + DATA_PER_WORD - 1) / DATA_PER_WORD;
  static constexpr size_t LAST_WORD_ELEMS =
      TOTAL_ELEMS - (OUTPUT_WORDS - 1) * DATA_PER_WORD;
  static constexpr size_t LAST_WORD_BYTES =
      LAST_WORD_ELEMS * TInput::width / 8;

  static_assert(DATA_PER_WORD >= DIM1_UNROLL * DIM2_UNROLL,
                "DATA_PER_WORD must contain a full input word");
  static_assert(TOTAL_ELEMS % (DIM1_UNROLL * DIM2_UNROLL) == 0,
                "tensor size must divide input parallelism");
  static_assert(DIM1_UNROLL == 1 || DIM2 == DIM2_UNROLL,
                "DIM2 must equal DIM2_UNROLL when DIM1_UNROLL is greater than one");
  static_assert(DIM2 % DIM2_UNROLL == 0,
                "DIM2 must divide DIM2_UNROLL");
  static_assert(DIM1 % DIM1_UNROLL == 0,
                "DIM1 must divide DIM1_UNROLL");

public:
  #pragma hls_design block
  template <size_t HLS_TAG>
  void run(ac_channel<TInputWord> input_data_stream[DIM1_UNROLL],
           ac_channel<TOutputWord> &output_data_stream) {
    TInput circular_buffer[DATA_PER_WORD * 2];
    size_t head = 0;
    size_t tail = 0;
    size_t size = 0;
    size_t output_word_index = 0;

    for (size_t i_input_word = 0; i_input_word < ITER; i_input_word++) {
      const bool end_of_tensor = i_input_word >= READS;
      if (!end_of_tensor) {
        for (size_t w = 0; w < DIM1_UNROLL; w++) {
          TInputWord input_data = input_data_stream[w].read();
          for (size_t ch = 0; ch < DIM2_UNROLL; ch++) {
            circular_buffer[head] = input_data[ch];
            head = (head + 1) % (DATA_PER_WORD * 2);
          }
        }
        size += DIM1_UNROLL * DIM2_UNROLL;
      }

      if (size >= DATA_PER_WORD || (end_of_tensor && size > 0)) {
        TOutputWord output_data{};
        using TKeep = decltype(output_data.keep);
        Quantizer quantizer;
        const bool last_word = output_word_index == OUTPUT_WORDS - 1;
        const size_t elements = last_word ? LAST_WORD_ELEMS : DATA_PER_WORD;
        for (size_t i = 0; i < elements; i++) {
          output_data.data.set_slc(
              static_cast<unsigned>(i * TInput::width),
              ac_int<TInput::width, false>(quantizer(circular_buffer[tail])));
          tail = (tail + 1) % (DATA_PER_WORD * 2);
        }
        output_data.last = last_word;
        if (last_word)
          output_data.keep = TKeep((TKeep(1) << LAST_WORD_BYTES) - 1);
        else
          output_data.keep = TKeep(-1);
        output_data.strb = output_data.keep;
        output_data_stream.write(output_data);
        size -= elements;
        output_word_index++;
      }
    }
  }
};
