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

static constexpr size_t OUT_HEIGHT = 1 + (test_config::IN_HEIGHT +
                                          test_config::PAD_T + test_config::PAD_B -
                                          test_config::DIL_H * (test_config::FH - 1) - 1) /
                                             test_config::STRIDE_H;
static constexpr size_t OUT_WIDTH = 1 + (test_config::IN_WIDTH +
                                         test_config::PAD_L + test_config::PAD_R -
                                         test_config::DIL_W * (test_config::FW - 1) - 1) /
                                            test_config::STRIDE_W;

void wrap_run(hls::stream<test_config::TWord> input_data_stream[1],
              hls::stream<test_config::TWord> output_data_stream[1],
              hls::stream<test_config::TWord> output_shift_data_stream[1]) {
  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, test_config::POS_H,
      test_config::POS_W, test_config::W_PAR, test_config::CH_PAR>
      window_selector;
  window_selector.run(input_data_stream, output_data_stream,
                      output_shift_data_stream);
}

bool test_run() {
  hls::stream<test_config::TWord> in_stream[1];
  hls::stream<test_config::TWord> out_stream[1];
  hls::stream<test_config::TWord> out_shift_stream[1];
  static constexpr size_t W_STREAM =
      (test_config::POS_W + test_config::PAD_L * (test_config::W_PAR - 1)) %
      test_config::W_PAR;

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
          if (W_STREAM == w_par) {
            in_stream[0].write(input_word);
          }
        }
      }
    }
  }

  // Run the operator
  wrap_run(in_stream, out_stream, out_shift_stream);

  // Check output
  bool flag = true;

  // Check output by generating the window and comparing the single pixel
  for (size_t h = 0; h < OUT_HEIGHT; h++) {
    for (size_t w = 0; w < OUT_WIDTH; w += test_config::W_PAR) {
      for (size_t i_ich = 0; i_ich < test_config::IN_CH;
           i_ich += test_config::CH_PAR) {

        size_t input_index_h = (h * test_config::STRIDE_H) -
                               test_config::PAD_T + test_config::POS_H;
        size_t input_index_w = (w * test_config::STRIDE_W) -
                               test_config::PAD_L + test_config::POS_W;
        if (input_index_h >= 0 && input_index_h < test_config::IN_HEIGHT &&
            input_index_w >= 0 && input_index_w < test_config::IN_WIDTH) {
          test_config::TWord data;
          data = out_stream[0].read();
          for (size_t i_ch_par = 0; i_ch_par < test_config::CH_PAR;
               i_ch_par++) {
            bool cmp =
                (data[i_ch_par] ==
                 test_config::input_tensor[0][i_ich + i_ch_par][input_index_h]
                                          [input_index_w]);
            // if (!cmp) {
              std::cout
                  << "Mismatch at index (h=" << h << ", w=" << w
                  << ", ich=" << i_ich << ", ch_par=" << i_ch_par
                  << "). got: " << data[i_ch_par] << ", expected: "
                  << test_config::input_tensor[0][i_ich + i_ch_par][input_index_h][input_index_w]
                                              
                  << std::endl;
            // }
            flag &= cmp;
          }
        }
      }
    }
  }

  // Empty shift output stream
  test_config::TWord shift_data;
  while (out_shift_stream[0].read_nb(shift_data))
    ;

  return flag;
}

bool test_step() {

  static constexpr size_t expectedII =
      test_config::IN_HEIGHT * test_config::IN_WIDTH * test_config::IN_CH /
      (test_config::CH_PAR * test_config::W_PAR);

  // Create input and output streams
  hls::stream<test_config::TWord> in_stream[1];
  hls::stream<test_config::TWord> out_stream[1];
  hls::stream<test_config::TWord> out_shift_stream[1];

  // Run the WindowSelector
  StreamingWindowSelector<
      test_config::TWord, test_config::IN_HEIGHT, test_config::IN_WIDTH,
      test_config::IN_CH, test_config::FH, test_config::FW,
      test_config::STRIDE_H, test_config::STRIDE_W, test_config::DIL_H,
      test_config::DIL_W, test_config::PAD_T, test_config::PAD_L,
      test_config::PAD_B, test_config::PAD_R, test_config::POS_H,
      test_config::POS_W, test_config::W_PAR, test_config::CH_PAR>
      window_selector(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {

    in_stream[0].write(test_config::TWord());
    ActorStatus actor_status =
        window_selector.step(in_stream, out_stream, out_shift_stream);
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
  test_config::TWord data;
  while (out_stream[0].read_nb(data))
    ;

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