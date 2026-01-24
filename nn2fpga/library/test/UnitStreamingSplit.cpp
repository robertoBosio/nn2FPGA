#include "StreamingSplit.hpp"
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
    hls::stream<test_config::TInputWord> input_data_stream[test_config::W_PAR],
    hls::stream<test_config::TOutputWord> output_data_stream0[test_config::W_PAR],
    hls::stream<test_config::TOutputWord> output_data_stream1[test_config::W_PAR]) {
  test_config::StreamingSplit streaming_split;
  streaming_split.run<0>(input_data_stream, output_data_stream0,
                         output_data_stream1);
}

bool test_run(){

  // Prepare input and output streams
  hls::stream<test_config::TInputWord> in_stream[test_config::W_PAR];
  hls::stream<test_config::TOutputWord> out_stream0[test_config::W_PAR];
  hls::stream<test_config::TOutputWord> out_stream1[test_config::W_PAR];

  // Fill input streams with test data
  for (size_t i_h = 0; i_h < test_config::IN_HEIGHT; i_h++) {
    for (size_t i_w = 0; i_w < test_config::IN_WIDTH;
         i_w += test_config::W_PAR) {
      for (size_t i_ch = 0; i_ch < test_config::IN_CH;
           i_ch += test_config::CH_PAR) {
        for (size_t w_par = 0; w_par < test_config::W_PAR; w_par++) {
          test_config::TInputWord input_word;
          for (size_t ch_par = 0; ch_par < test_config::CH_PAR; ch_par++) {
            input_word[ch_par] =
                test_config::input_tensor[0][i_ch + ch_par][i_h][i_w + w_par];
          }
          in_stream[w_par].write(input_word);
        }
      }
    }
  }

  // Run the operator
  wrap_run(in_stream, out_stream0, out_stream1);

  // Check first tensor
  bool flag = true;
  for (size_t o_h = 0; o_h < test_config::OUT_HEIGHT0; o_h++){
    for (size_t o_w = 0; o_w < test_config::OUT_WIDTH0; o_w += test_config::W_PAR){
      for (size_t o_ch = 0; o_ch < test_config::OUT_CH0; o_ch += test_config::CH_PAR){
        for (size_t w_par = 0; w_par < test_config::W_PAR; w_par++){
          test_config::TOutputWord output_word0 = out_stream0[w_par].read();
          for (size_t ch_par = 0; ch_par < test_config::CH_PAR; ch_par++){
            bool cmp0 = (output_word0[ch_par] ==
                         test_config::output_tensor0[0][o_ch + ch_par][o_h][o_w + w_par]);
            if (!cmp0){
              std::cout << "Mismatch at output0 index (o_h=" << o_h << ", o_w=" << o_w
                        << ", o_ch=" << o_ch << ", w_par=" << w_par
                        << ", ch_par=" << ch_par << "): " << output_word0[ch_par]
                        << " != "
                        << test_config::output_tensor0[0][o_ch + ch_par][o_h][o_w + w_par]
                        << std::endl;
            }
            flag &= cmp0;
          }
        }
      }
    }
  }

  // Check second tensor
  for (size_t o_h = 0; o_h < test_config::OUT_HEIGHT1; o_h++){
    for (size_t o_w = 0; o_w < test_config::OUT_WIDTH1; o_w += test_config::W_PAR){
      for (size_t o_ch = 0; o_ch < test_config::OUT_CH1; o_ch += test_config::CH_PAR){
        for (size_t w_par = 0; w_par < test_config::W_PAR; w_par++){
          test_config::TOutputWord output_word1 = out_stream1[w_par].read();
          for (size_t ch_par = 0; ch_par < test_config::CH_PAR; ch_par++){
            bool cmp1 = (output_word1[ch_par] ==
                         test_config::output_tensor1[0][o_ch + ch_par][o_h][o_w + w_par]);
            if (!cmp1){
              std::cout << "Mismatch at output1 index (o_h=" << o_h << ", o_w=" << o_w
                        << ", o_ch=" << o_ch << ", w_par=" << w_par
                        << ", ch_par=" << ch_par << "): " << output_word1[ch_par]
                        << " != "
                        << test_config::output_tensor1[0][o_ch + ch_par][o_h][o_w + w_par]
                        << std::endl;
            }
            flag &= cmp1;
          }
        }
      }
    }
  }

  return flag;
}

bool test_step() {

  static constexpr size_t expectedII =
      test_config::IN_HEIGHT * test_config::IN_WIDTH * test_config::IN_CH /
      (test_config::W_PAR * test_config::CH_PAR);

  // Prepare input and output streams
  hls::stream<test_config::TInputWord> in_stream[test_config::W_PAR];
  hls::stream<test_config::TOutputWord> out_stream0[test_config::W_PAR];
  hls::stream<test_config::TOutputWord> out_stream1[test_config::W_PAR];

  test_config::StreamingSplit streaming_split;
  streaming_split.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    // Provide dummy input data to keep the pipeline busy
    for (size_t i_w_par = 0; i_w_par < test_config::W_PAR; i_w_par++) {
      test_config::TInputWord input_struct;
      in_stream[i_w_par].write(input_struct);
    }

    ActorStatus actor_status = streaming_split.step(in_stream, out_stream0, out_stream1);
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
  for (size_t w_par = 0; w_par < test_config::W_PAR; w_par++) {
    test_config::TOutputWord output_struct;
    while (out_stream0[w_par].read_nb(output_struct))
      ;
  }

  for (size_t w_par = 0; w_par < test_config::W_PAR; w_par++) {
    test_config::TOutputWord output_struct;
    while (out_stream1[w_par].read_nb(output_struct))
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