#include "StreamingPad.hpp"
#include "StreamingWindowSelector.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

static constexpr size_t OUT_HEIGHT =
    1 + (test_config::IN_HEIGHT + test_config::PAD_T + test_config::PAD_B -
         test_config::DIL_H * (test_config::FH - 1) - 1) /
            test_config::STRIDE_H;
static constexpr size_t OUT_WIDTH =
    1 + (test_config::IN_WIDTH + test_config::PAD_L + test_config::PAD_R -
         test_config::DIL_W * (test_config::FW - 1) - 1) /
            test_config::STRIDE_W;
static constexpr size_t FW_EXPAND =
    test_config::FW + (test_config::W_PAR - 1) * test_config::STRIDE_W;

void wrap_run(
    hls::stream<test_config::TWord> i_data[test_config::W_PAR],
    hls::stream<test_config::TWord> o_data[test_config::FH * FW_EXPAND]) {
#pragma HLS DATAFLOW
  hls::stream<test_config::TWord> buffer_stream[14];
  hls::stream<test_config::TWord> pre_pad[18];

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 2, 5, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel0;
  window_selector_pixel0.run<0>(i_data[0], pre_pad[17], buffer_stream[0]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 2, 4, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel1;
  window_selector_pixel1.run<1>(i_data[3], pre_pad[16], buffer_stream[1]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 2, 3, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel2;
  window_selector_pixel2.run<2>(i_data[2], pre_pad[15], buffer_stream[4]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 2, 2, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel3;
  window_selector_pixel3.run<3>(i_data[1], pre_pad[14], buffer_stream[5]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 2, 1, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel4;
  window_selector_pixel4.run<4>(buffer_stream[0], pre_pad[13],
                                buffer_stream[2]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 2, 0, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel5;
  window_selector_pixel5.run<5>(buffer_stream[1], pre_pad[12],
                                buffer_stream[3]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 1, 5, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel6;
  window_selector_pixel6.run<6>(buffer_stream[2], pre_pad[11],
                                buffer_stream[6]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 1, 4, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel7;
  window_selector_pixel7.run<7>(buffer_stream[3], pre_pad[10],
                                buffer_stream[7]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 1, 3, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel8;
  window_selector_pixel8.run<8>(buffer_stream[4], pre_pad[9],
                                buffer_stream[10]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 1, 2, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel9;
  window_selector_pixel9.run<9>(buffer_stream[5], pre_pad[8],
                                buffer_stream[11]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 1, 1, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel10;
  window_selector_pixel10.run<10>(buffer_stream[6], pre_pad[7],
                                  buffer_stream[8]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 1, 0, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel11;
  window_selector_pixel11.run<11>(buffer_stream[7], pre_pad[6],
                                  buffer_stream[9]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 0, 5, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel12;
  window_selector_pixel12.run<12>(buffer_stream[8], pre_pad[5],
                                  buffer_stream[12]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 0, 4, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel13;
  window_selector_pixel13.run<13>(buffer_stream[9], pre_pad[4],
                                  buffer_stream[13]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 0, 3, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel14;
  window_selector_pixel14.run<14>(buffer_stream[10], pre_pad[3]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 0, 2, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel15;
  window_selector_pixel15.run<15>(buffer_stream[11], pre_pad[2]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 0, 1, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel16;
  window_selector_pixel16.run<16>(buffer_stream[12], pre_pad[1]);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 0, 0, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel17;
  window_selector_pixel17.run<17>(buffer_stream[13], pre_pad[0]);

  StreamingPad<test_config::TWord, test_config::IN_HEIGHT,
               test_config::IN_WIDTH, test_config::IN_CH, test_config::FH,
               test_config::FW, test_config::STRIDE_H, test_config::STRIDE_W,
               test_config::DIL_H, test_config::DIL_W, test_config::PAD_T,
               test_config::PAD_L, test_config::PAD_B, test_config::PAD_R,
               test_config::W_PAR, test_config::CH_PAR>
      pad;
  pad.run<18>(pre_pad, o_data);
}

bool test_run() {
  hls::stream<test_config::TWord> in_stream[test_config::W_PAR];
  hls::stream<test_config::TWord> out_stream[test_config::FH * FW_EXPAND];

  // Fill input streams with test data
  for (size_t i = 0; i < test_config::IN_HEIGHT; i++) {
    for (size_t j = 0; j < test_config::IN_WIDTH; j += test_config::W_PAR) {
      for (size_t ch = 0; ch < test_config::IN_CH;
           ch += test_config::CH_PAR) {
        for (size_t w_par = 0; w_par < test_config::W_PAR; w_par++) {
          test_config::TWord input_word;
          for (size_t ch_par = 0; ch_par < test_config::CH_PAR; ch_par++) {
            input_word[ch_par] =
                test_config::input_tensor[0][ch + ch_par][i][j + w_par];
          }

          // Write only the correct section of the tensor based on position
          in_stream[w_par].write(input_word);
        }
      }
    }
  }

    // Run the operator
  wrap_run(in_stream, out_stream);

  // Check output
  bool flag = true;

  // Check output by generating the window padded and comparing the single pixel
  for (size_t h = 0; h < OUT_HEIGHT; h++) {
    for (size_t w = 0; w < OUT_WIDTH; w += test_config::W_PAR) {
      for (size_t i_ich = 0; i_ich < test_config::IN_CH;
           i_ich += test_config::CH_PAR) {
        for (size_t fh = 0; fh < test_config::FH; fh++) {
          for (size_t fw = 0; fw < FW_EXPAND; fw++) {
            size_t input_index_h =
                (h * test_config::STRIDE_H) - test_config::PAD_T + fh;
            size_t input_index_w =
                (w * test_config::STRIDE_W) - test_config::PAD_L + fw;

            test_config::TWord data = out_stream[fh * FW_EXPAND + fw].read();
            bool cmp;
            if (input_index_h >= 0 && input_index_h < test_config::IN_HEIGHT &&
                input_index_w >= 0 && input_index_w < test_config::IN_WIDTH) {
              for (size_t i_ch_par = 0; i_ch_par < test_config::CH_PAR;
                   i_ch_par++) {
                cmp = (data[i_ch_par] ==
                       test_config::input_tensor[0][i_ich + i_ch_par]
                                                [input_index_h][input_index_w]);
                if (!cmp) {
                  std::cout
                      << "Mismatch at index (h=" << h << ", w=" << w
                      << ", ich=" << i_ich << ", fh=" << fh << ", fw=" << fw
                      << ", ch_par=" << i_ch_par << "). got: " << data[i_ch_par]
                      << ", expected: "
                      << test_config::input_tensor[0][i_ich + i_ch_par]
                                                  [input_index_h][input_index_w]
                      << std::endl;
                }
              }
            } else {
              for (size_t i_ch_par = 0; i_ch_par < test_config::CH_PAR;
                   i_ch_par++) {
                cmp = (data[i_ch_par] == 0);
                if (!cmp) {
                  std::cout << "Mismatch at index (h=" << h << ", w=" << w
                            << ", ich=" << i_ich << ", fh=" << fh
                            << ", fw=" << fw << ", ch_par=" << i_ch_par
                            << "). got: " << data[i_ch_par] << ", expected: 0"
                            << std::endl;
                }
              }
            }
            flag &= cmp;
          }
        }
      }
    }
  }

  // Empty shift output stream
  for (size_t i = 0; i < test_config::FH * FW_EXPAND; i++) {
    if (!out_stream[i].empty()) {
      flag = false;
      std::cout << "Output stream " << i << " not empty after reading."
                << std::endl;
    }
  }

  return flag;
}

bool test_step() {

  static constexpr size_t expectedII =
      OUT_HEIGHT * OUT_WIDTH * test_config::IN_CH /
      (test_config::CH_PAR * test_config::W_PAR);

  // Create input and output streams
  hls::stream<test_config::TWord> i_data[test_config::W_PAR];
  hls::stream<test_config::TWord> o_data[test_config::FH * FW_EXPAND];
  hls::stream<test_config::TWord> buffer_stream[14];
  hls::stream<test_config::TWord> pre_pad[18];

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 2, 5, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel0;
  window_selector_pixel0.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 2, 4, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel1;
  window_selector_pixel1.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 2, 3, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel2;
  window_selector_pixel2.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 2, 2, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel3;
  window_selector_pixel3.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 2, 1, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel4;
  window_selector_pixel4.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 2, 0, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel5;
  window_selector_pixel5.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 1, 5, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel6;
  window_selector_pixel6.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 1, 4, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel7;
  window_selector_pixel7.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 1, 3, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel8;
  window_selector_pixel8.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 1, 2, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel9;
  window_selector_pixel9.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 1, 1, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel10;
  window_selector_pixel10.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 1, 0, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel11;
  window_selector_pixel11.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 0, 5, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel12;
  window_selector_pixel12.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 0, 4, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel13;
  window_selector_pixel13.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 0, 3, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel14;
  window_selector_pixel14.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 0, 2, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel15;
  window_selector_pixel15.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 0, 1, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel16;
  window_selector_pixel16.step_init(test_config::PIPELINE_DEPTH);

  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, 0, 0, test_config::W_PAR,
      test_config::CH_PAR>
      window_selector_pixel17;
  window_selector_pixel17.step_init(test_config::PIPELINE_DEPTH);

  StreamingPad<test_config::TWord, test_config::IN_HEIGHT,
               test_config::IN_WIDTH, test_config::IN_CH, test_config::FH,
               test_config::FW, test_config::STRIDE_H, test_config::STRIDE_W,
               test_config::DIL_H, test_config::DIL_W, test_config::PAD_T,
               test_config::PAD_L, test_config::PAD_B, test_config::PAD_R,
               test_config::W_PAR, test_config::CH_PAR>
      pad;
  pad.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    std::vector<ActorStatus> actor_statuses;
    std::vector<size_t> channel_quantities;

    // Feed input streams
    test_config::TWord input_word;
    for (size_t i = 0; i < test_config::W_PAR; i++) {
      i_data[i].write(input_word); // Dummy data, not used in step
    }
    ActorStatus actor_status;
    actor_status = pad.step(pre_pad, o_data);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel17.step(
        buffer_stream[13],
        pre_pad[0]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel16.step(
        buffer_stream[12],
        pre_pad[1]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel13.step(
        buffer_stream[9],
        pre_pad[4],
        buffer_stream[13]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel12.step(
        buffer_stream[8],
        pre_pad[5],
        buffer_stream[12]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel11.step(
        buffer_stream[7],
        pre_pad[6],
        buffer_stream[9]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel10.step(
        buffer_stream[6],
        pre_pad[7],
        buffer_stream[8]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel15.step(
        buffer_stream[11],
        pre_pad[2]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel14.step(
        buffer_stream[10],
        pre_pad[3]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel7.step(
        buffer_stream[3],
        pre_pad[10],
        buffer_stream[7]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel6.step(
        buffer_stream[2],
        pre_pad[11],
        buffer_stream[6]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel9.step(
        buffer_stream[5],
        pre_pad[8],
        buffer_stream[11]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel8.step(
        buffer_stream[4],
        pre_pad[9],
        buffer_stream[10]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel5.step(
        buffer_stream[1],
        pre_pad[12],
        buffer_stream[3]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel4.step(
        buffer_stream[0],
        pre_pad[13],
        buffer_stream[2]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel3.step(
        i_data[1],
        pre_pad[14],
        buffer_stream[5]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel2.step(
        i_data[2],
        pre_pad[15],
        buffer_stream[4]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel1.step(
        i_data[3],
        pre_pad[16],
        buffer_stream[1]);
    actor_statuses.push_back(actor_status);
    actor_status = window_selector_pixel0.step(
        i_data[0],
        pre_pad[17],
        buffer_stream[0]);
    actor_statuses.push_back(actor_status);

    channel_quantities.push_back(0);
    current_state = CSDFGState(actor_statuses, channel_quantities);
    if (visited_states.find(current_state) != visited_states.end()) {
      II = clock_cycles - visited_states[current_state];
      break;
    }
    visited_states.emplace(current_state, clock_cycles);

    // Prevent infinite loops in case of errors
    clock_cycles++;
    assert(clock_cycles < 10 * expectedII);
  }

  // Flush the output stream.
  for (size_t i = 0; i < test_config::FH * FW_EXPAND; i++) {
    test_config::TWord data;
    while (o_data[i].read_nb(data))
      ;
  }

  bool flag = (II == expectedII);
  std::cout << "Expected II: " << expectedII << ", Measured II: " << II
            << std::endl;
  return flag;
}

int main(int argc, char **argv) {

  bool all_passed = true;

  all_passed &= test_run();

  // Testing the pipeline with csim only, as it is only relevant for fifo depth
  // estimations
  if (argc > 1 && std::string(argv[1]) == "csim") {
    all_passed &= test_step();
  }

  if (!all_passed) {
    std::cout << "Failed." << std::endl;
  } else {
    std::cout << "Passed." << std::endl;
  }

  return all_passed ? 0 : 1;
}