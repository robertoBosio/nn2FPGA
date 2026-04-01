#include "YoloAttention/Transpose.hpp"
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

void wrap_run(hls::stream<test_config::TWord> input_data[1],
              hls::stream<test_config::TWord> output_data[1]) {
  TransposeRowCol<test_config::TWord, test_config::IN_HEIGHT,
                  test_config::IN_WIDTH, test_config::IN_CH>
      transpose;
  transpose.run<0>(input_data, output_data);
}

bool test_run() {

  // Prepare input and output streams
  hls::stream<test_config::TWord> in_stream[1];
  hls::stream<test_config::TWord> out_stream[1];

  // Fill input streams with test data in NWCH format
  for (size_t i_w = 0; i_w < test_config::IN_WIDTH; i_w += 1) {
    for (size_t i_ch = 0; i_ch < test_config::IN_CH; i_ch++) {
      for (size_t i_h = 0; i_h < test_config::IN_HEIGHT; i_h++) {
        test_config::TWord input_word;
        input_word[0] = test_config::tensor[0][i_ch][i_h][i_w];
        in_stream[0].write(input_word);
      }
    }
  }
  // Run the operator
  wrap_run(in_stream, out_stream);

  // Check output tensor in NHCW format
  bool flag = true;
  for (size_t i_h = 0; i_h < test_config::IN_HEIGHT; i_h++) {
    for (size_t i_ch = 0; i_ch < test_config::IN_CH; i_ch++) {
      for (size_t i_w = 0; i_w < test_config::IN_WIDTH; i_w += 1) {
        test_config::TWord output_word = out_stream[0].read();
        bool cmp = output_word[0] == test_config::tensor[0][i_ch][i_h][i_w];
        if (!cmp) {
          std::cout << "Mismatch at index (i_h=" << i_h << ", i_w=" << i_w
                    << ", i_ch=" << i_ch << "): " << output_word[0]
                    << " != " << test_config::tensor[0][i_ch][i_h][i_w]
                    << std::endl;
        }
        flag &= cmp;
      }
    }
  }

  return flag;
}

bool test_step() {

  static constexpr size_t expectedII =
      test_config::IN_HEIGHT * test_config::IN_WIDTH * test_config::IN_CH * 2;

  // Prepare input and output streams
  hls::stream<test_config::TWord> in_stream[1];
  hls::stream<test_config::TWord> out_stream[1];

  TransposeRowCol<test_config::TWord, test_config::IN_HEIGHT,
                  test_config::IN_WIDTH, test_config::IN_CH>
      transpose;
  transpose.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    // Provide dummy input data to keep the pipeline busy
    in_stream[0].write(test_config::TWord());

    ActorStatus actor_status = transpose.step(in_stream, out_stream);
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
  test_config::TWord output_struct;
  while (out_stream[0].read_nb(output_struct))
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