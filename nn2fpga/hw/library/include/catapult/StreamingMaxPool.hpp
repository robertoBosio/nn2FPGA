#pragma once

#include "catapult/DequantQuant.hpp"
#include <ac_channel.h>
#include <cstddef>

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t OUT_CH,
          size_t OUT_HEIGHT, size_t OUT_WIDTH, size_t FH, size_t FW,
          size_t STRIDE_H, size_t STRIDE_W, size_t CH_PAR, size_t W_PAR>
class StreamingMaxPool {
  static constexpr size_t FW_EXPAND = FW + (W_PAR - 1) * STRIDE_W;

  static_assert(OUT_HEIGHT > 0 && OUT_WIDTH > 0,
                "output dimensions must be positive");
  static_assert(FH > 0 && FW > 0 && STRIDE_H > 0 && STRIDE_W > 0,
                "filter dimensions and strides must be positive");
  static_assert(CH_PAR > 0 && W_PAR > 0,
                "parallelism factors must be positive");
  static_assert(OUT_CH % CH_PAR == 0,
                "OUT_CH must divide CH_PAR");
  static_assert(OUT_WIDTH % W_PAR == 0,
                "OUT_WIDTH must divide W_PAR");

public:
  #pragma hls_design block
  template <size_t HLS_TAG>
  void run(ac_channel<TInputWord> i_data[FH * FW_EXPAND],
           ac_channel<TOutputWord> o_data[W_PAR]) {
    for (size_t i_hw = 0; i_hw < OUT_HEIGHT * OUT_WIDTH / W_PAR; i_hw++) {
      for (size_t i_ch = 0; i_ch < OUT_CH; i_ch += CH_PAR) {
        TInputWord input_data[FH][FW_EXPAND];
        TOutputWord output_data[W_PAR];
        Quantizer quantizer;

        for (size_t fh = 0; fh < FH; fh++)
          for (size_t fw = 0; fw < FW_EXPAND; fw++)
            input_data[fh][fw] = i_data[fh * FW_EXPAND + fw].read();

        for (size_t w = 0; w < W_PAR; w++) {
          for (size_t ch = 0; ch < CH_PAR; ch++) {
            TInput maximum = CatapultLimits<TInput>::min();
            for (size_t fh = 0; fh < FH; fh++)
              for (size_t fw = 0; fw < FW; fw++) {
                TInput value = input_data[fh][fw + w * STRIDE_W][ch];
                maximum = value > maximum ? value : maximum;
              }
            output_data[w][ch] = quantizer(maximum);
          }
          o_data[w].write(output_data[w]);
        }
      }
    }
  }
};
