#include "DequantQuant.hpp"
#include "StreamingMaxPool.hpp"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

using TInputWord = std::array<test_config::TInput, test_config::CH_PAR>;
static constexpr size_t FW_EXPAND =
    test_config::FW + (test_config::W_PAR - 1) * test_config::STRIDE_W;
static constexpr size_t OUT_HEIGHT =
    1 + (test_config::IN_HEIGHT + test_config::PAD_T + test_config::PAD_B -
         test_config::DIL_H * (test_config::FH - 1) - 1) /
            test_config::STRIDE_H;
static constexpr size_t OUT_WIDTH =
    1 + (test_config::IN_WIDTH + test_config::PAD_L + test_config::PAD_R -
         test_config::DIL_W * (test_config::FW - 1) - 1) /
            test_config::STRIDE_W;
using TOutputWord = TInputWord;

void wrap_run(hls::stream<TInputWord> i_data[test_config::FH * FW_EXPAND],
              hls::stream<TInputWord> o_data[test_config::W_PAR]) {
  // Wrapper for synthesis.
  StreamingMaxPool<TInputWord, test_config::TInput, test_config::OUT_CH,
                   OUT_HEIGHT, OUT_WIDTH,
                   test_config::FH, test_config::FW, test_config::STRIDE_H,
                   test_config::STRIDE_W, test_config::CH_PAR,
                   test_config::W_PAR>
      pool;
  pool.run<0>(i_data, o_data);
}

bool test_run() {

  // Prepare input and output streams
  hls::stream<TInputWord> in_stream[test_config::FH * FW_EXPAND];
  hls::stream<TOutputWord> out_stream[test_config::W_PAR];

  // Fill input streams with test data
  for (size_t h = 0; h < OUT_HEIGHT; h++) {
    for (size_t w = 0; w < OUT_WIDTH; w += test_config::W_PAR) {
      for (size_t i_ich = 0; i_ich < test_config::OUT_CH;
           i_ich += test_config::CH_PAR) {
        for (size_t fh = 0; fh < test_config::FH; fh++) {
          for (size_t fw = 0; fw < FW_EXPAND; fw++) {

            size_t input_index_h =
                (h * test_config::STRIDE_H) - test_config::PAD_T + fh;
            size_t input_index_w =
                (w * test_config::STRIDE_W) - test_config::PAD_L + fw;

            TInputWord input_data;
            for (size_t i_ch_par = 0; i_ch_par < test_config::CH_PAR;
                 i_ch_par++) {
              if (input_index_h < 0 || input_index_h >= test_config::IN_HEIGHT ||
                  input_index_w < 0 || input_index_w >= test_config::IN_WIDTH) {
                input_data[i_ch_par] = LimitsImpl<test_config::TInput>::min(); // Padding with zeros
              } else {
                input_data[i_ch_par] =
                    test_config::input_tensor[0][i_ich + i_ch_par]
                                             [input_index_h][input_index_w];
              }
            }
            in_stream[fh * FW_EXPAND + fw].write(input_data);
          }
        }
      }
    }
  }

  // Run pooling
  wrap_run(in_stream, out_stream);

  // Read and check output
  bool flag = true;
  for (size_t h = 0; h < OUT_HEIGHT; h++) {
    for (size_t w = 0; w < OUT_WIDTH; w += test_config::W_PAR) {
      for (size_t i_och = 0; i_och < test_config::OUT_CH;
           i_och += test_config::CH_PAR) {
        for (size_t i_w_par = 0; i_w_par < test_config::W_PAR; i_w_par++) {
          TOutputWord output_data = out_stream[i_w_par].read();
          for (size_t i_och_par = 0; i_och_par < test_config::CH_PAR;
               i_och_par++) {

            // Check if the output data matches the expected result
            if (output_data[i_och_par] !=
                test_config::output_tensor[0][i_och + i_och_par][h][w + i_w_par]
                                          ) {
              std::cerr << "Output mismatch at (" << h << ", " << w + i_w_par
                        << ", " << i_och + i_och_par
                        << "): " << output_data[i_och_par] << " != "
                        << test_config::output_tensor[0][i_och + i_och_par][h]
                                                     [w + i_w_par]
                        << std::endl;
              return false;
            }
          }
        }
      }
    }
  }

  // Ensure all streams are empty
  for (size_t i = 0; i < test_config::FH * FW_EXPAND; i++) {
    if (!in_stream[i].empty()) {
      std::cout << "Input stream not empty after run." << std::endl;
      flag = false;
    }
  }

  for (size_t i = 0; i < test_config::W_PAR; i++) {
    if (!out_stream[i].empty()) {
      std::cout << "Output stream not empty after run." << std::endl;
      flag = false;
    }
  }

  std::cout << "Test run completed." << std::endl;
  return flag;
}

bool test_step() {

  static constexpr size_t expectedII =
      OUT_HEIGHT * OUT_WIDTH * test_config::OUT_CH /
      (test_config::CH_PAR * test_config::W_PAR);

  // Create input and output streams
  hls::stream<TInputWord> i_data[test_config::FH * FW_EXPAND];
  hls::stream<TOutputWord> o_data[test_config::W_PAR];

  // Run the global average pooling
  StreamingMaxPool<TInputWord, test_config::TInput,
                   test_config::OUT_CH, OUT_HEIGHT,
                   OUT_WIDTH, test_config::FH, test_config::FW,
                   test_config::STRIDE_H, test_config::STRIDE_W,
                   test_config::CH_PAR, test_config::W_PAR>
      pool;
  pool.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {

    // Provide dummy input data to keep the pipeline busy
    for (size_t i = 0; i < test_config::FH * FW_EXPAND; i++) {
      i_data[i].write(TInputWord());
    }

    ActorStatus actor_status = pool.step(i_data, o_data);
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
  TOutputWord output_struct;
  while (o_data[0].read_nb(output_struct))
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
