#pragma once

#include <ac_channel.h>
#include <ac_int.h>
#include <cstddef>

template <typename TInputWord, typename TOutput, typename TOutputWord,
          size_t DATA_PER_WORD, size_t DATA_TO_SHIFT, size_t TIMES,
          size_t WORDS, size_t ARRAY_PAR, size_t WORD_PAR,
          bool USE_URAM = false, int MEM_LATENCY = -1>
class StreamingMemory {
  static constexpr size_t CH_GROUPS = WORDS / (ARRAY_PAR * WORD_PAR);

  static_assert(DATA_PER_WORD > 0, "DATA_PER_WORD must be positive");
  static_assert(WORDS % (ARRAY_PAR * WORD_PAR) == 0,
                "WORDS must be divisible by the output parallelism");

  static void write_output(ac_channel<TOutputWord> o_data[ARRAY_PAR],
                           const TOutput mem[WORD_PAR][ARRAY_PAR]) {
    for (size_t i_array = 0; i_array < ARRAY_PAR; ++i_array) {
      TOutputWord output_word;
      for (size_t i_word = 0; i_word < WORD_PAR; ++i_word)
        output_word[i_word] = mem[i_word][i_array];
      o_data[i_array].write(output_word);
    }
  }

  static void initialize(ac_channel<TInputWord> i_shift_data[1],
                         TOutput mem[CH_GROUPS][WORD_PAR][ARRAY_PAR]) {
    size_t i_array = 0;
    size_t i_group = 0;
    size_t i_word = 0;
    TOutput word[WORD_PAR][ARRAY_PAR];

    for (size_t offset = 0; offset < WORDS; offset += DATA_PER_WORD) {
      TInputWord input_word = i_shift_data[0].read();
      for (size_t i_data = 0;
           i_data < DATA_PER_WORD && offset + i_data < WORDS; ++i_data) {
        word[i_word][i_array] = TOutput(
            input_word[0].template slc<TOutput::width>(
                static_cast<unsigned>(TOutput::width * i_data)));
        if (++i_array == ARRAY_PAR) {
          i_array = 0;
          if (++i_word == WORD_PAR) {
            i_word = 0;
            for (size_t word_index = 0; word_index < WORD_PAR; ++word_index)
              for (size_t array_index = 0; array_index < ARRAY_PAR; ++array_index)
                mem[i_group][word_index][array_index] =
                    word[word_index][array_index];
            ++i_group;
          }
        }
      }
    }
  }

 public:
  template <size_t HLS_TAG>
  void run(ac_channel<TInputWord> i_shift_data[1],
           ac_channel<TOutputWord> o_data[ARRAY_PAR],
           ac_channel<TInputWord> o_shift_data[1]) {
    static TOutput mem[CH_GROUPS][WORD_PAR][ARRAY_PAR];
    static bool initialized = false;
    if (!initialized) {
      initialize(i_shift_data, mem);
      for (size_t i = 0; i < DATA_TO_SHIFT; ++i)
        o_shift_data[0].write(i_shift_data[0].read());
      initialized = true;
    }
    for (size_t repeat = 0; repeat < TIMES; ++repeat)
      for (size_t group = 0; group < CH_GROUPS; ++group)
        write_output(o_data, mem[group]);
  }

  template <size_t HLS_TAG>
  void run(ac_channel<TInputWord> i_shift_data[1],
           ac_channel<TOutputWord> o_data[ARRAY_PAR]) {
    static TOutput mem[CH_GROUPS][WORD_PAR][ARRAY_PAR];
    static bool initialized = false;
    if (!initialized) {
      initialize(i_shift_data, mem);
      initialized = true;
    }
    for (size_t repeat = 0; repeat < TIMES; ++repeat)
      for (size_t group = 0; group < CH_GROUPS; ++group)
        write_output(o_data, mem[group]);
  }
};
