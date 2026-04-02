#include "YoloAttention/ReshapeV.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include "utils/CSDFG_utils.hpp"
#include "DequantQuant.hpp"

void wrap_run(
    hls::stream<test_config::TInputWord> input_data[2],
    hls::stream<test_config::TOutputWord>
        output_data[1]) {
  ReshapeV<test_config::TInputWord, test_config::TInput,
           test_config::TOutputWord, test_config::TOutput,
           test_config::Quantizer, test_config::IN_HEADS, test_config::IN_DIM,
           test_config::IN_SEQ, test_config::OUT_HEIGHT, test_config::OUT_WIDTH,
           test_config::OUT_CH, test_config::REDUCE_PAR>
      reshape_v;
  reshape_v.run<0>(input_data, output_data);
}

bool test_run(){

  // Prepare input and output streams
  hls::stream<test_config::TInputWord> in_stream[2];
  hls::stream<test_config::TOutputWord> out_stream[1];

  // Fill input streams with test data
  for (size_t i_seq = 0; i_seq < test_config::IN_SEQ; i_seq++) {
    for (size_t i_head = 0; i_head < test_config::IN_HEADS; i_head += 1) {
      for (size_t i_dim = 0; i_dim < test_config::IN_DIM; i_dim += test_config::REDUCE_PAR) {
        test_config::TInputWord input_word;
        for (size_t reduce_par = 0; reduce_par < test_config::REDUCE_PAR;
             reduce_par++) {
          input_word[reduce_par] =
              test_config::input_tensor[0][i_head][i_dim + reduce_par][i_seq];
        }
        in_stream[i_head].write(input_word);
      }
    }
  }

  // Run the operator
  wrap_run(in_stream, out_stream);

  // Check output tensor
  bool flag = true;
  for (size_t i_h = 0; i_h < test_config::OUT_HEIGHT; i_h++) {
    for (size_t i_w = 0; i_w < test_config::OUT_WIDTH; i_w++) {
      for (size_t i_ch = 0; i_ch < test_config::OUT_CH; i_ch += test_config::REDUCE_PAR) {
        test_config::TOutputWord output_word = out_stream[0].read();
        for (size_t reduce_par = 0; reduce_par < test_config::REDUCE_PAR;
             reduce_par++) {
          bool cmp0 =
              (output_word[reduce_par] ==
               test_config::output_tensor[0][i_ch + reduce_par][i_h][i_w]);
          if (!cmp0) {
            std::cout
                << "Mismatch at output index (i_h=" << i_h << ", i_w=" << i_w
                << ", i_ch=" << i_ch + reduce_par << "): "
                << output_word[reduce_par] << " != "
                << test_config::output_tensor[0][i_ch + reduce_par][i_h][i_w]
                << std::endl;
          }
          flag &= cmp0;
        }
      }
    }
  }

  return flag;
}

bool test_step() {

  static constexpr size_t expectedII =
      test_config::OUT_HEIGHT * test_config::OUT_WIDTH * test_config::OUT_CH /
      test_config::REDUCE_PAR;

  // Prepare input and output streams
  hls::stream<test_config::TInputWord> in_stream[2];
  hls::stream<test_config::TOutputWord> out_stream[1];

  ReshapeV<test_config::TInputWord, test_config::TInput,
           test_config::TOutputWord, test_config::TOutput,
           test_config::Quantizer, test_config::IN_HEADS, test_config::IN_DIM,
           test_config::IN_SEQ, test_config::OUT_HEIGHT, test_config::OUT_WIDTH,
           test_config::OUT_CH, test_config::REDUCE_PAR>
      reshape_v;
  reshape_v.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    // Provide dummy input data to keep the pipeline busy
    for (size_t i_heads_par = 0; i_heads_par < 2; i_heads_par++) {
      test_config::TInputWord input_struct;
      in_stream[i_heads_par].write(input_struct);
    }

    ActorStatus actor_status = reshape_v.step(in_stream, out_stream);
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
  test_config::TOutputWord output_struct;
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