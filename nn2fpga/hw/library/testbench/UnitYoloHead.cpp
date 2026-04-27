#include "DequantQuant.hpp"
#include "StreamingSoftmax.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

using TInputWord = std::array<test_config::TInput, test_config::REDUCE_PAR>;
using TOutputWord = std::array<test_config::TOutput, test_config::REDUCE_PAR>;

void wrap_run(hls::stream<TInputWord> i_data[test_config::LINE_PAR],
              const test_config::TLUT LUTmem[test_config::LUT_SIZE],
              hls::stream<TOutputWord> o_data[test_config::LINE_PAR]) {
  StreamingSoftmax<
      TInputWord, test_config::TInput, TOutputWord, test_config::TOutput,
      test_config::TLUT, test_config::TAcc, test_config::TDiv,
      test_config::Quantizer, test_config::LUT_SIZE,
      test_config::OUT_HEIGHT * test_config::OUT_WIDTH, test_config::OUT_CH,
      test_config::LINE_PAR, test_config::REDUCE_PAR>
      streaming_softmax;
  streaming_softmax.run<0>(i_data, LUTmem, o_data);
}

bool test_run() {

  // Prepare input and output streams
  hls::stream<TInputWord> in_stream[test_config::LINE_PAR];
  hls::stream<TOutputWord> out_stream[test_config::LINE_PAR];

  // Fill input streams with test data
  for (size_t i_h = 0; i_h < test_config::IN_HEIGHT; i_h++) {
    for (size_t i_w = 0; i_w < test_config::IN_WIDTH;
         i_w += test_config::LINE_PAR) {
      for (size_t ch = 0; ch < test_config::IN_CH;
           ch += test_config::REDUCE_PAR) {
        for (size_t i_line_par = 0; i_line_par < test_config::LINE_PAR;
             i_line_par++) {
          TInputWord input_word;
          for (size_t i_red_par = 0; i_red_par < test_config::REDUCE_PAR;
               i_red_par++) {
            input_word[i_red_par] =
                test_config::input_tensor[0][ch + i_red_par][i_h]
                                         [i_w + i_line_par];
          }
          in_stream[i_line_par].write(input_word);
        }
      }
    }
  }

  // Run the operator
  wrap_run(in_stream, test_config::LUTmem, out_stream);

  // Check output
  bool flag = true;
  for (size_t i_w = 0; i_w < test_config::OUT_WIDTH;
       i_w += test_config::LINE_PAR) {
    for (size_t i_h = 0; i_h < test_config::OUT_HEIGHT; i_h++) {
      for (size_t ch = 0; ch < test_config::OUT_CH;
           ch += test_config::REDUCE_PAR) {
        for (size_t i_line_par = 0; i_line_par < test_config::LINE_PAR;
             i_line_par++) {
          TOutputWord output_word = out_stream[i_line_par].read();
          for (size_t i_red_par = 0; i_red_par < test_config::REDUCE_PAR;
               i_red_par++) {
            bool cmp = abs(output_word[i_red_par] -
                           test_config::output_tensor[0][ch + i_red_par][i_h]
                                                     [i_w + i_line_par]) < 2;
            if (!cmp) {
              std::cout << "Mismatch at index (i_h=" << i_h
                        << ", i_w=" << i_w + i_line_par
                        << ", ch=" << ch + i_red_par
                        << "): " << output_word[i_red_par] << " != "
                        << test_config::output_tensor[0][ch + i_red_par][i_h]
                                                     [i_w + i_line_par]
                        << std::endl;
            }
            flag &= cmp;
          }
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
      test_config::IN_HEIGHT * test_config::IN_WIDTH * test_config::IN_CH * 3 /
      (test_config::LINE_PAR * test_config::REDUCE_PAR);

  // Prepare input and output streams
  hls::stream<TInputWord> in_stream[test_config::LINE_PAR];
  hls::stream<TOutputWord> out_stream[test_config::LINE_PAR];

  StreamingSoftmax<
      TInputWord, test_config::TInput, TOutputWord, test_config::TOutput,
      test_config::TLUT, test_config::TAcc, test_config::TDiv,
      test_config::Quantizer, test_config::LUT_SIZE,
      test_config::OUT_HEIGHT * test_config::OUT_WIDTH, test_config::OUT_CH,
      test_config::LINE_PAR, test_config::REDUCE_PAR>
      streaming_softmax;
  streaming_softmax.step_init(test_config::PIPELINE_DEPTH);
  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    // Provide dummy input data to keep the pipeline busy
    for (size_t i_line_par = 0; i_line_par < test_config::LINE_PAR;
         i_line_par++) {
      TInputWord input_struct;
      for (size_t i_red_par = 0; i_red_par < test_config::REDUCE_PAR;
           i_red_par++) {
        input_struct[i_red_par] = 0; // Dummy data
      }
      in_stream[i_line_par].write(input_struct);
    }

    ActorStatus actor_status =
        streaming_softmax.step(in_stream, test_config::LUTmem, out_stream);
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
  for (size_t i_line_par = 0; i_line_par < test_config::LINE_PAR;
       i_line_par++) {
    TOutputWord output_struct;
    while (out_stream[i_line_par].read_nb(output_struct))
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