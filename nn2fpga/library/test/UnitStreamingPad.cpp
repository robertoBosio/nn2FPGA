#include "StreamingPad.hpp"
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
    hls::stream<test_config::TWord> i_data[test_config::FH * FW_EXPAND],
    hls::stream<test_config::TWord> o_data[test_config::FH * FW_EXPAND]) {
  StreamingPad<test_config::TWord, test_config::IN_HEIGHT,
               test_config::IN_WIDTH, test_config::IN_CH, test_config::FH,
               test_config::FW, test_config::STRIDE_H, test_config::STRIDE_W,
               test_config::DIL_H, test_config::DIL_W, test_config::PAD_T,
               test_config::PAD_L, test_config::PAD_B, test_config::PAD_R,
               test_config::W_PAR, test_config::CH_PAR>
      pad;
  pad.run(i_data, o_data);
}

bool test_run() {
  hls::stream<test_config::TWord> in_stream[test_config::FH * FW_EXPAND];
  hls::stream<test_config::TWord> out_stream[test_config::FH * FW_EXPAND];

  // Generating unpadded input windows.
  for (size_t h = 0; h < OUT_HEIGHT; h++) {
    for (size_t w = 0; w < OUT_WIDTH; w += test_config::W_PAR) {
      for (size_t i_ich = 0; i_ich < test_config::IN_CH;
           i_ich += test_config::CH_PAR) {
        for (size_t fh = 0; fh < test_config::FH; fh++) {
          for (size_t fw = 0; fw < FW_EXPAND; fw++) {
            test_config::TWord input_data;
            size_t input_index_h =
                (h * test_config::STRIDE_H) - test_config::PAD_T + fh;
            size_t input_index_w =
                (w * test_config::STRIDE_W) - test_config::PAD_L + fw;

            if (input_index_h >= 0 && input_index_h < test_config::IN_HEIGHT &&
                input_index_w >= 0 && input_index_w < test_config::IN_WIDTH) {
              for (size_t i_ch_par = 0; i_ch_par < test_config::CH_PAR;
                   i_ch_par++) {

                input_data[i_ch_par] =
                    test_config::input_tensor[0][i_ich + i_ch_par]
                                             [input_index_h][input_index_w];
              }
              in_stream[fh * FW_EXPAND + fw].write(input_data);
            }
          }
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
  hls::stream<test_config::TWord> in_stream[test_config::FH * FW_EXPAND];
  hls::stream<test_config::TWord> out_stream[test_config::FH * FW_EXPAND];

  // Run the Pad
  StreamingPad<test_config::TWord, test_config::IN_HEIGHT,
               test_config::IN_WIDTH, test_config::IN_CH, test_config::FH,
               test_config::FW, test_config::STRIDE_H, test_config::STRIDE_W,
               test_config::DIL_H, test_config::DIL_W, test_config::PAD_T,
               test_config::PAD_L, test_config::PAD_B, test_config::PAD_R,
               test_config::W_PAR, test_config::CH_PAR>
      pad(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {

    for (size_t fh = 0; fh < test_config::FH; fh++) {
      for (size_t fw = 0; fw < FW_EXPAND; fw++) {
        test_config::TWord input_data;
        in_stream[fh * FW_EXPAND + fw].write(input_data);
      }
    }

    ActorStatus actor_status =
        pad.step(in_stream, out_stream);
    std::vector<ActorStatus> actor_statuses;
    std::vector<size_t> channel_quantities;
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
    while (out_stream[i].read_nb(data))
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