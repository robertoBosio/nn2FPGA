#pragma once

#include <ac_channel.h>
#include <cstddef>

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t IN_DIM0, size_t IN_DIM1,
          size_t IN_DIM2, size_t IN_DIM1_UNROLL, size_t OUT_DIM1_UNROLL,
          size_t IN_DIM2_UNROLL, size_t OUT_DIM2_UNROLL>
class BandwidthAdjust {
  static_assert(IN_DIM0 > 0 && IN_DIM1 > 0 && IN_DIM2 > 0,
                "dimensions must be greater than 0");
  static_assert(IN_DIM1 % IN_DIM1_UNROLL == 0 &&
                    IN_DIM1 % OUT_DIM1_UNROLL == 0,
                "width must be divisible by both unroll factors");
  static_assert(IN_DIM2 % IN_DIM2_UNROLL == 0 &&
                    IN_DIM2 % OUT_DIM2_UNROLL == 0,
                "channels must be divisible by both unroll factors");

  static TOutputWord quantize_word(const TInputWord &input) {
    TOutputWord output;
    Quantizer quantizer;
    for (size_t channel = 0; channel < OUT_DIM2_UNROLL; ++channel)
      output[channel] = quantizer(input[channel]);
    return output;
  }

public:
  template <size_t HLS_TAG>
  void run(ac_channel<TInputWord> i_data[IN_DIM1_UNROLL],
           ac_channel<TOutputWord> o_data[OUT_DIM1_UNROLL]) {
    static_assert((IN_DIM1_UNROLL == OUT_DIM1_UNROLL) !=
                      (IN_DIM2_UNROLL == OUT_DIM2_UNROLL),
                  "BandwidthAdjust changes exactly one tensor dimension");

    if (IN_DIM1_UNROLL < OUT_DIM1_UNROLL) {
      for (size_t x = 0; x < IN_DIM0 * IN_DIM1; x += OUT_DIM1_UNROLL)
        for (size_t group = 0; group < OUT_DIM1_UNROLL;
             group += IN_DIM1_UNROLL)
          for (size_t ch = 0; ch < IN_DIM2; ch += IN_DIM2_UNROLL)
            for (size_t lane = 0; lane < IN_DIM1_UNROLL; ++lane)
              o_data[group + lane].write(quantize_word(i_data[lane].read()));
    } else if (IN_DIM1_UNROLL > OUT_DIM1_UNROLL) {
      for (size_t x = 0; x < IN_DIM0 * IN_DIM1; x += IN_DIM1_UNROLL)
        for (size_t group = 0; group < IN_DIM1_UNROLL;
             group += OUT_DIM1_UNROLL)
          for (size_t ch = 0; ch < IN_DIM2; ch += IN_DIM2_UNROLL)
            for (size_t lane = 0; lane < OUT_DIM1_UNROLL; ++lane)
              o_data[lane].write(quantize_word(i_data[group + lane].read()));
    } else if (IN_DIM2_UNROLL < OUT_DIM2_UNROLL) {
      TOutputWord output[OUT_DIM1_UNROLL];
      for (size_t x = 0; x < IN_DIM0 * IN_DIM1; x += IN_DIM1_UNROLL)
        for (size_t ch = 0; ch < IN_DIM2; ch += OUT_DIM2_UNROLL)
          for (size_t part = 0; part < OUT_DIM2_UNROLL;
               part += IN_DIM2_UNROLL)
            for (size_t lane = 0; lane < IN_DIM1_UNROLL; ++lane) {
              const TInputWord input = i_data[lane].read();
              for (size_t i = 0; i < IN_DIM2_UNROLL; ++i) {
                Quantizer quantizer;
                output[lane][part + i] = quantizer(input[i]);
              }
              if (part + IN_DIM2_UNROLL == OUT_DIM2_UNROLL)
                o_data[lane].write(output[lane]);
            }
    } else {
      for (size_t x = 0; x < IN_DIM0 * IN_DIM1; x += IN_DIM1_UNROLL)
        for (size_t ch = 0; ch < IN_DIM2; ch += IN_DIM2_UNROLL) {
          TInputWord input[IN_DIM1_UNROLL];
          for (size_t part = 0; part < IN_DIM2_UNROLL;
               part += OUT_DIM2_UNROLL)
            for (size_t lane = 0; lane < IN_DIM1_UNROLL; ++lane) {
              if (part == 0)
                input[lane] = i_data[lane].read();
              TOutputWord output;
              Quantizer quantizer;
              for (size_t i = 0; i < OUT_DIM2_UNROLL; ++i)
                output[i] = quantizer(input[lane][part + i]);
              o_data[lane].write(output);
            }
        }
    }
  }
};

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t IN_DIM0, size_t IN_DIM1,
          size_t IN_DIM2, size_t IN_DIM1_UNROLL, size_t OUT_DIM1_UNROLL,
          size_t IN_DIM2_UNROLL, size_t OUT_DIM2_UNROLL>
using BandwidthAdjustIncreaseStreams =
    BandwidthAdjust<TInputWord, TInput, TOutputWord, TOutput, Quantizer, IN_DIM0,
                    IN_DIM1, IN_DIM2, IN_DIM1_UNROLL, OUT_DIM1_UNROLL,
                    IN_DIM2_UNROLL, OUT_DIM2_UNROLL>;

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t IN_DIM0, size_t IN_DIM1,
          size_t IN_DIM2, size_t IN_DIM1_UNROLL, size_t OUT_DIM1_UNROLL,
          size_t IN_DIM2_UNROLL, size_t OUT_DIM2_UNROLL>
using BandwidthAdjustDecreaseStreams =
    BandwidthAdjust<TInputWord, TInput, TOutputWord, TOutput, Quantizer, IN_DIM0,
                    IN_DIM1, IN_DIM2, IN_DIM1_UNROLL, OUT_DIM1_UNROLL,
                    IN_DIM2_UNROLL, OUT_DIM2_UNROLL>;

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t IN_DIM0, size_t IN_DIM1,
          size_t IN_DIM2, size_t IN_DIM1_UNROLL, size_t OUT_DIM1_UNROLL,
          size_t IN_DIM2_UNROLL, size_t OUT_DIM2_UNROLL>
using BandwidthAdjustIncreaseWord =
    BandwidthAdjust<TInputWord, TInput, TOutputWord, TOutput, Quantizer, IN_DIM0,
                    IN_DIM1, IN_DIM2, IN_DIM1_UNROLL, OUT_DIM1_UNROLL,
                    IN_DIM2_UNROLL, OUT_DIM2_UNROLL>;

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t IN_DIM0, size_t IN_DIM1,
          size_t IN_DIM2, size_t IN_DIM1_UNROLL, size_t OUT_DIM1_UNROLL,
          size_t IN_DIM2_UNROLL, size_t OUT_DIM2_UNROLL>
using BandwidthAdjustDecreaseWord =
    BandwidthAdjust<TInputWord, TInput, TOutputWord, TOutput, Quantizer, IN_DIM0,
                    IN_DIM1, IN_DIM2, IN_DIM1_UNROLL, OUT_DIM1_UNROLL,
                    IN_DIM2_UNROLL, OUT_DIM2_UNROLL>;
