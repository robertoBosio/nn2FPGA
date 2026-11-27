#pragma once

#include <ac_channel.h>
#include <cstddef>

template <typename TInputWord, typename TInput, typename TWeightWord,
          typename TWeight, typename TBiasWord, typename TBias,
          typename TOutputWord, typename TOutput, typename TSum,
          typename TPartialSum, typename OutputTransform, size_t OUT_CH,
          size_t IN_CH, size_t OUT_HEIGHT, size_t OUT_WIDTH, size_t GROUP,
          size_t FH, size_t FW, size_t STRIDE_H, size_t STRIDE_W,
          size_t IN_CH_PAR, size_t OUT_CH_PAR, size_t W_PAR>
class StreamingConv {
  static constexpr size_t FW_EXPAND = FW + (W_PAR - 1) * STRIDE_W;

  static_assert(OUT_HEIGHT > 0 && OUT_WIDTH > 0,
                "output dimensions must be positive");
  static_assert(FH > 0 && FW > 0 && STRIDE_H > 0 && STRIDE_W > 0,
                "filter dimensions and strides must be positive");
  static_assert(IN_CH_PAR > 0 && OUT_CH_PAR > 0 && W_PAR > 0,
                "parallelism factors must be positive");
  static_assert(GROUP == 1, "grouped convolution is not supported");
  static_assert(OUT_CH % OUT_CH_PAR == 0 && IN_CH % IN_CH_PAR == 0,
                "channel counts must divide their parallelism factors");
  static_assert(OUT_WIDTH % W_PAR == 0, "OUT_WIDTH must divide W_PAR");

public:
  #pragma hls_design block
  template <size_t HLS_TAG>
  void run(ac_channel<TInputWord> i_data[FH * FW_EXPAND],
           ac_channel<TWeightWord> i_weights[FH * FW],
           ac_channel<TBiasWord> i_biases[1],
           ac_channel<TOutputWord> o_data[W_PAR]) {
    TPartialSum acc_buff[OUT_CH / OUT_CH_PAR][OUT_CH_PAR * W_PAR];
    TInputWord input_data[FH][FW_EXPAND];

    for (size_t i_hw = 0; i_hw < OUT_HEIGHT * OUT_WIDTH / W_PAR; i_hw++) {
      for (size_t i_ich = 0; i_ich < IN_CH; i_ich += IN_CH_PAR) {
        for (size_t i_och = 0; i_och < OUT_CH; i_och += OUT_CH_PAR) {
          TWeightWord weight_data[FH][FW];
          TPartialSum partial[OUT_CH_PAR * W_PAR];
          TOutputWord output_data;
          TBiasWord bias_data;
          OutputTransform output_transform;

          if (i_och == 0) {
            for (size_t fh = 0; fh < FH; fh++)
              for (size_t fw = 0; fw < FW_EXPAND; fw++)
                input_data[fh][fw] = i_data[fh * FW_EXPAND + fw].read();
          }
          for (size_t fh = 0; fh < FH; fh++)
            for (size_t fw = 0; fw < FW; fw++)
              weight_data[fh][fw] = i_weights[fh * FW + fw].read();
          for (size_t i = 0; i < OUT_CH_PAR * W_PAR; i++)
            partial[i] = 0;
          for (size_t w = 0; w < W_PAR; w++)
            for (size_t och = 0; och < OUT_CH_PAR; och++)
              for (size_t fh = 0; fh < FH; fh++)
                for (size_t fw = 0; fw < FW; fw++)
                  for (size_t ich = 0; ich < IN_CH_PAR; ich++)
                    partial[w * OUT_CH_PAR + och] +=
                        input_data[fh][fw + w * STRIDE_W][ich] *
                        weight_data[fh][fw][och * IN_CH_PAR + ich];
          if (i_ich == 0)
            for (size_t i = 0; i < OUT_CH_PAR * W_PAR; i++)
              acc_buff[i_och / OUT_CH_PAR][i] = 0;
          if (i_ich == IN_CH - IN_CH_PAR)
            bias_data = i_biases[0].read();
          for (size_t w = 0; w < W_PAR; w++) {
            for (size_t och = 0; och < OUT_CH_PAR; och++) {
              const size_t index = w * OUT_CH_PAR + och;
              acc_buff[i_och / OUT_CH_PAR][index] += partial[index];
              if (i_ich == IN_CH - IN_CH_PAR)
                output_data[och] = output_transform(
                    TSum(acc_buff[i_och / OUT_CH_PAR][index]) + bias_data[och]);
            }
            if (i_ich == IN_CH - IN_CH_PAR)
              o_data[w].write(output_data);
          }
        }
      }
    }
  }
};
