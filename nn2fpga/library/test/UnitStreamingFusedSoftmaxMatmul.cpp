#include "DequantQuant.hpp"
#include "StreamingFusedSoftmaxMatmul.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

using TQKInputWord = std::array<test_config::TQKInput, test_config::REDUCE_PAR>;
using TVInputWord = std::array<test_config::TVInput, test_config::REDUCE_PAR>;
using TOutputWord = std::array<test_config::TOutput, test_config::REDUCE_PAR>;

void wrap_run(hls::stream<TQKInputWord> qk_data[1],
              hls::stream<TVInputWord> v_data[1],
              const test_config::TLUT LUTmem[test_config::LUT_SIZE],
              hls::stream<test_config::TOutput> o_data[1]) {
  StreamingFusedSoftmaxMatmul<
      TQKInputWord, test_config::TQKInput, TVInputWord, test_config::TVInput,
      TOutputWord, test_config::TOutput,
      test_config::TLUT, test_config::TNum, test_config::TDen,
      test_config::TDiv, test_config::Quantizer, test_config::LUT_SIZE,
      test_config::DIM_HEADS, test_config::DIM_V,
      test_config::DIM_SEQ, test_config::REDUCE_PAR>
      streaming_fusedsoftmaxmatmul;
  streaming_fusedsoftmaxmatmul.run<0>(qk_data, v_data, LUTmem, o_data);
}

bool test_run() {

  // Prepare input and output streams
  hls::stream<TQKInputWord> qk_stream[1];
  hls::stream<TVInputWord> v_stream[1];
  hls::stream<test_config::TOutput> out_stream[1];

  // Streaming qk in NHCW order
  for (size_t i_h = 0; i_h < test_config::DIM_SEQ; i_h++) {
    for (size_t i_head = 0; i_head < test_config::DIM_HEADS; i_head++) {
      for (size_t i_red = 0; i_red < test_config::DIM_SEQ;
           i_red += test_config::REDUCE_PAR) {
        TQKInputWord input_word;
        for (size_t red_par = 0; red_par < test_config::REDUCE_PAR; red_par++) {
          input_word[red_par] =
              test_config::qk_tensor[0][i_head][i_h][i_red + red_par];
        }
        qk_stream[0].write(input_word);
      }
    }
  }

  // Streaming v in NHCW order
  for (size_t i_head = 0; i_head < test_config::DIM_HEADS; i_head++) {
    for (size_t i_h = 0; i_h < test_config::DIM_V; i_h++) {
      for (size_t i_red = 0; i_red < test_config::DIM_SEQ;
           i_red += test_config::REDUCE_PAR) {
        TVInputWord input_word;
        for (size_t red_par = 0; red_par < test_config::REDUCE_PAR; red_par++) {
          input_word[red_par] =
              test_config::v_tensor[0][i_head][i_h][i_red + red_par];
        }
        v_stream[0].write(input_word);
      }
    }
  }

  // Run the operator
  wrap_run(qk_stream, v_stream, test_config::LUTmem, out_stream);

  // Check output in NWCH order
  bool flag = true;
  for (size_t i_seq = 0; i_seq < test_config::DIM_SEQ; i_seq++) {
    for (size_t i_head = 0; i_head < test_config::DIM_HEADS; i_head++) {
      for (size_t i_v = 0; i_v < test_config::DIM_V;
           i_v += test_config::REDUCE_PAR) {
        for (size_t red_par = 0; red_par < test_config::REDUCE_PAR; red_par++) {
          test_config::TOutput output_word = out_stream[0].read();
          bool cmp = (std::abs(output_word -
                          test_config::output_tensor[0][i_head][i_v + red_par]
                                                    [i_seq]) <= 1);
          if (!cmp) {
            std::cout
                << "Mismatch at index (i_head=" << i_head << ", i_seq=" << i_seq
                << ", dim=" << i_v + red_par << "): " << output_word
                << " != "
                << test_config::output_tensor[0][i_head][i_v + red_par][i_seq]
                << std::endl;
          }
          flag &= cmp;
        }
      }
    }
  }
  std::cout << "Output comparison " << (flag ? "passed" : "failed")
            << std::endl;

  return flag;
}

bool test_step() {

  static constexpr size_t expectedII =
      test_config::DIM_SEQ * test_config::DIM_V * test_config::DIM_HEADS *
      test_config::DIM_SEQ / test_config::REDUCE_PAR;

  // Prepare input and output streams
  hls::stream<TQKInputWord> qk_stream[1];
  hls::stream<TVInputWord> v_stream[1];
  hls::stream<test_config::TOutput> out_stream[1];

  StreamingFusedSoftmaxMatmul<
      TQKInputWord, test_config::TQKInput, TVInputWord, test_config::TVInput,
      TOutputWord, test_config::TOutput, test_config::TLUT, test_config::TNum,
      test_config::TDen, test_config::TDiv, test_config::Quantizer,
      test_config::LUT_SIZE, test_config::DIM_HEADS, test_config::DIM_V,
      test_config::DIM_SEQ, test_config::REDUCE_PAR>
      streaming_fusedsoftmaxmatmul;
  streaming_fusedsoftmaxmatmul.step_init(test_config::PIPELINE_DEPTH);
  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    // Provide dummy input data to keep the pipeline busy
    qk_stream[0].write(TQKInputWord());
    v_stream[0].write(TVInputWord());

    ActorStatus actor_status =
        streaming_fusedsoftmaxmatmul.step(qk_stream, v_stream, test_config::LUTmem, out_stream);
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
  test_config::TOutput output_struct;
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