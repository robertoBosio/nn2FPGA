#pragma once

#include <ac_channel.h>
#include <cstddef>

// Functional Catapult implementation. The compiler simulation path needs the
// same window ordering as Vitis; scheduling-optimized row storage follows later.
template <typename TWord, typename TData, size_t IN_HEIGHT, size_t IN_WIDTH,
          size_t IN_CH, size_t FH, size_t FW, size_t STRIDE_H,
          size_t STRIDE_W, size_t DILATION_H, size_t DILATION_W, size_t PAD_T,
          size_t PAD_L, size_t PAD_B, size_t PAD_R, size_t W_PAR,
          size_t CH_PAR, int PAD_VALUE = 0>
class StreamingCircularLineBuffer {
  static constexpr size_t FW_EXPAND = FW + (W_PAR - 1) * STRIDE_W;
  static constexpr size_t OUT_HEIGHT =
      (IN_HEIGHT + PAD_T + PAD_B - DILATION_H * (FH - 1) - 1) / STRIDE_H + 1;
  static constexpr size_t OUT_WIDTH =
      (IN_WIDTH + PAD_L + PAD_R - DILATION_W * (FW - 1) - 1) / STRIDE_W + 1;
  static constexpr size_t CH_GROUPS = IN_CH / CH_PAR;

  static TWord pad_word() {
    TWord word;
    for (size_t ch = 0; ch < CH_PAR; ch++)
      word[ch] = TData(PAD_VALUE);
    return word;
  }

public:
  static_assert(FH > 0 && FW > 0 && STRIDE_H > 0 && STRIDE_W > 0,
                "filter dimensions and strides must be positive");
  static_assert(DILATION_H == 1 && DILATION_W == 1,
                "only dilation one is supported");
  static_assert(IN_CH % CH_PAR == 0 && IN_WIDTH % W_PAR == 0,
                "input dimensions must divide parallelism");
  static_assert(OUT_WIDTH % W_PAR == 0,
                "output width must divide W_PAR");

  #pragma hls_design block
  template <size_t HLS_TAG>
  void run(ac_channel<TWord> i_data[W_PAR],
           ac_channel<TWord> o_data[FH * FW_EXPAND]) {
    TWord input[IN_HEIGHT][IN_WIDTH][CH_GROUPS];
    for (size_t h = 0; h < IN_HEIGHT; h++)
      for (size_t w = 0; w < IN_WIDTH; w += W_PAR)
        for (size_t ch = 0; ch < CH_GROUPS; ch++)
          for (size_t lane = 0; lane < W_PAR; lane++)
            input[h][w + lane][ch] = i_data[lane].read();

    for (size_t out_h = 0; out_h < OUT_HEIGHT; out_h++)
      for (size_t out_w = 0; out_w < OUT_WIDTH; out_w += W_PAR)
        for (size_t ch = 0; ch < CH_GROUPS; ch++)
          for (size_t fh = 0; fh < FH; fh++)
            for (size_t fw = 0; fw < FW_EXPAND; fw++) {
              const int in_h = int(out_h * STRIDE_H + fh) - int(PAD_T);
              const int in_w = int(out_w * STRIDE_W + fw) - int(PAD_L);
              const TWord word = in_h >= 0 && in_h < int(IN_HEIGHT) &&
                                         in_w >= 0 && in_w < int(IN_WIDTH)
                                     ? input[in_h][in_w][ch]
                                     : pad_word();
              o_data[fh * FW_EXPAND + fw].write(word);
            }
  }
};
