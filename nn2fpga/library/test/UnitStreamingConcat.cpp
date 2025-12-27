#include "StreamingConcat.hpp"
#include "DequantQuant.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

using TInputWord = std::array<test_config::TInput, test_config::CH_PAR>;
using TOutputWord = std::array<test_config::TOutput, test_config::CH_PAR>;

void wrap_run(hls::stream<TInputWord> i_data0[test_config::W_PAR],
              hls::stream<TInputWord> i_data1[test_config::W_PAR],
              hls::stream<TOutputWord> o_data[test_config::W_PAR]) {
  StreamingConcat<TInputWord, test_config::TInput, TOutputWord,
                  test_config::TOutput, test_config::Quantizer,
                  test_config::IN_HEIGHT, test_config::IN_WIDTH,
                  test_config::IN_CH_A, test_config::IN_CH_B,
                  test_config::W_PAR, test_config::CH_PAR>
      concat;
  concat.run<0>(i_data0, i_data1, o_data);
}

bool test_run() {
  hls::stream<TInputWord> i_data0[test_config::W_PAR];
  hls::stream<TInputWord> i_data1[test_config::W_PAR];
  hls::stream<TOutputWord> o_data[test_config::W_PAR];

  // Streaming input tensors.
  for (size_t h = 0; h < test_config::IN_HEIGHT; h++) {
    for (size_t w = 0; w < test_config::IN_WIDTH; w += test_config::W_PAR) {
      for (size_t i_ich = 0; i_ich < test_config::IN_CH_A;
           i_ich += test_config::CH_PAR) {
        for (size_t i_w_par = 0; i_w_par < test_config::W_PAR; i_w_par++) {
          TInputWord input_data;
          for (size_t i_ch_par = 0; i_ch_par < test_config::CH_PAR;
               i_ch_par++) {
            input_data[i_ch_par] =
                test_config::input_tensor0[0][i_ich + i_ch_par][h][w + i_w_par];
          }
          i_data0[i_w_par].write(input_data);
        }
      }
    }
  }
  for (size_t h = 0; h < test_config::IN_HEIGHT; h++) {
    for (size_t w = 0; w < test_config::IN_WIDTH; w += test_config::W_PAR) {
      for (size_t i_ich = 0; i_ich < test_config::IN_CH_B;
           i_ich += test_config::CH_PAR) {
        for (size_t i_w_par = 0; i_w_par < test_config::W_PAR; i_w_par++) {
          TInputWord input_data;
          for (size_t i_ch_par = 0; i_ch_par < test_config::CH_PAR;
               i_ch_par++) {
            input_data[i_ch_par] =
                test_config::input_tensor1[0][i_ich + i_ch_par][h][w + i_w_par];
          }
          i_data1[i_w_par].write(input_data);
        }
      }
    }
  }
    // Run the operator
  wrap_run(i_data0, i_data1, o_data);

  // Check output
  bool flag = true;

  // Check output tensor.
  for (size_t h = 0; h < test_config::IN_HEIGHT; h++) {
    for (size_t w = 0; w < test_config::IN_WIDTH; w += test_config::W_PAR) {
      for (size_t i_ich = 0; i_ich < test_config::IN_CH_A + test_config::IN_CH_B;
           i_ich += test_config::CH_PAR) {
        for (size_t i_w_par = 0; i_w_par < test_config::W_PAR; i_w_par++) {
          TOutputWord data = o_data[i_w_par].read();
          bool cmp;
          for (size_t i_ch_par = 0; i_ch_par < test_config::CH_PAR;
               i_ch_par++) {
            cmp = (data[i_ch_par] ==
                   test_config::output_tensor[0][i_ich + i_ch_par][h]
                                             [w + i_w_par]);
            if (!cmp) {
              std::cout << "Mismatch at index (h=" << h << ", w=" << w
                        << ", ich=" << i_ich << ", w_par=" << i_w_par
                        << ", ch_par=" << i_ch_par
                        << "). got: " << data[i_ch_par] << ", expected: "
                        << test_config::output_tensor[0][i_ich + i_ch_par][h]
                                                     [w + i_w_par]
                        << std::endl;
            }
            flag &= cmp;
          }
        }
      }
    }
  }

  // Empty shift output stream
  for (size_t i = 0; i < test_config::W_PAR; i++) {
    if (!o_data[i].empty()) {
      flag = false;
      std::cout << "Output stream " << i << " not empty after reading."
                << std::endl;
    }
  }

  return flag;
}

bool test_step() {

  static constexpr size_t expectedII =
      test_config::IN_HEIGHT * test_config::IN_WIDTH *
      (test_config::IN_CH_A + test_config::IN_CH_B) /
      (test_config::CH_PAR * test_config::W_PAR);

  // Create input and output streams
  hls::stream<TInputWord> i_data0[test_config::W_PAR];
  hls::stream<TInputWord> i_data1[test_config::W_PAR];
  hls::stream<TOutputWord> o_data[test_config::W_PAR];
  StreamingConcat<TInputWord, test_config::TInput, TOutputWord,
                  test_config::TOutput, test_config::Quantizer,
                  test_config::IN_HEIGHT, test_config::IN_WIDTH,
                  test_config::IN_CH_A, test_config::IN_CH_B,
                  test_config::W_PAR, test_config::CH_PAR>
      concat;
  concat.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {

    // Dummy input data
    for (size_t i = 0; i < test_config::W_PAR; i++){
      i_data0[i].write(TInputWord());
      i_data1[i].write(TInputWord());
    }

    ActorStatus actor_status =
        concat.step(i_data0, i_data1, o_data);
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
  for (size_t i = 0; i < test_config::W_PAR; i++) {
    TOutputWord data;
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