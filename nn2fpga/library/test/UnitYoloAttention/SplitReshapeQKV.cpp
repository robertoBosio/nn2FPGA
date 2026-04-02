#include "YoloAttention/SplitReshapeQKV.hpp"
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
    hls::stream<test_config::TInputWord> input_data[1],
    hls::stream<test_config::TOutputWord> output_data_q[2],
    hls::stream<test_config::TOutputWord> output_data_k[2],
    hls::stream<test_config::TOutputWord>
        output_data_v[2]) {
  SplitReshapeQKV<test_config::TInputWord, test_config::TInput,
               test_config::TOutputWord, test_config::TOutput,
               test_config::Quantizer, test_config::IN_HEIGHT,
               test_config::IN_WIDTH, test_config::IN_CH,
               test_config::REDUCE_PAR>
      split_reshape;
  split_reshape.run<0>(input_data, output_data_q, output_data_k, output_data_v);
}

bool test_run(){

  // Prepare input and output streams
  hls::stream<test_config::TInputWord> in_stream[1];
  hls::stream<test_config::TOutputWord> out_streamq[2];
  hls::stream<test_config::TOutputWord> out_streamk[2];
  hls::stream<test_config::TOutputWord> out_streamv[2];

  // Fill input streams with test data
  for (size_t i_h = 0; i_h < test_config::IN_HEIGHT; i_h++) {
    for (size_t i_w = 0; i_w < test_config::IN_WIDTH;
         i_w += 1) {
      for (size_t i_ch = 0; i_ch < test_config::IN_CH;
           i_ch += test_config::REDUCE_PAR) {
        for (size_t w_par = 0; w_par < 1; w_par++) {
          test_config::TInputWord input_word;
          for (size_t ch_par = 0; ch_par < test_config::REDUCE_PAR; ch_par++) {
            input_word[ch_par] =
                test_config::input_tensor[0][i_ch + ch_par][i_h][i_w + w_par];
          }
          in_stream[w_par].write(input_word);
        }
      }
    }
  }

  // Run the operator
  wrap_run(in_stream, out_streamq, out_streamk, out_streamv);

  // Check q tensor
  bool flag = true;
  for (size_t o_seq = 0; o_seq < 400; o_seq++) {
    for (size_t o_heads = 0; o_heads < 2; o_heads += 1) {
      for (size_t o_q = 0; o_q < 32; o_q += test_config::REDUCE_PAR) {
        test_config::TOutputWord output_word0 = out_streamq[o_heads].read();
        for (size_t reduce_par = 0; reduce_par < test_config::REDUCE_PAR;
             reduce_par++) {
          bool cmp0 =
              (output_word0[reduce_par] ==
               test_config::q_tensor[0][o_heads][o_q + reduce_par][o_seq]);
          if (!cmp0) {
            std::cout
                << "Mismatch at output0 index (o_heads=" << o_heads
                << ", o_q=" << o_q + reduce_par << ", o_seq=" << o_seq
                << "): " << output_word0[reduce_par] << " != "
                << test_config::q_tensor[0][o_heads][o_q + reduce_par][o_seq]
                << std::endl;
          }
          flag &= cmp0;
        }
      }
    }
  }

  // Check k tensor
  for (size_t o_seq = 0; o_seq < 400; o_seq++) {
    for (size_t o_heads = 0; o_heads < 2; o_heads += 1) {
      for (size_t o_k = 0; o_k < 32; o_k += test_config::REDUCE_PAR) {
        test_config::TOutputWord output_word0 = out_streamk[o_heads].read();
        for (size_t reduce_par = 0; reduce_par < test_config::REDUCE_PAR;
             reduce_par++) {
          bool cmp0 =
              (output_word0[reduce_par] ==
               test_config::k_tensor[0][o_heads][o_k + reduce_par][o_seq]);
          if (!cmp0) {
            std::cout
                << "Mismatch at output0 index (o_heads=" << o_heads
                << ", o_k=" << o_k + reduce_par << ", o_seq=" << o_seq
                << "): " << output_word0[reduce_par] << " != "
                << test_config::k_tensor[0][o_heads][o_k + reduce_par][o_seq]
                << std::endl;
          }
          flag &= cmp0;
        }
      }
    }
  }

  // Check v tensor
  for (size_t o_seq = 0; o_seq < 400; o_seq++) {
    for (size_t o_heads = 0; o_heads < 2; o_heads += 1) {
      for (size_t o_v = 0; o_v < 64; o_v += test_config::REDUCE_PAR) {
        test_config::TOutputWord output_word0 = out_streamv[o_heads].read();
        for (size_t reduce_par = 0; reduce_par < test_config::REDUCE_PAR;
             reduce_par++) {
          bool cmp0 =
              (output_word0[reduce_par] ==
               test_config::v_tensor[0][o_heads][o_v + reduce_par][o_seq]);
          if (!cmp0) {
            std::cout
                << "Mismatch at output0 index (o_heads=" << o_heads
                << ", o_v=" << o_v + reduce_par << ", o_seq=" << o_seq
                << "): " << output_word0[reduce_par] << " != "
                << test_config::v_tensor[0][o_heads][o_v + reduce_par][o_seq]
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
      test_config::IN_HEIGHT * test_config::IN_WIDTH * test_config::IN_CH /
      test_config::REDUCE_PAR;

  // Prepare input and output streams
  hls::stream<test_config::TInputWord> in_stream[1];
  hls::stream<test_config::TOutputWord> out_streamq[2];
  hls::stream<test_config::TOutputWord> out_streamk[2];
  hls::stream<test_config::TOutputWord> out_streamv[2];

  SplitReshapeQKV<test_config::TInputWord, test_config::TInput,
               test_config::TOutputWord, test_config::TOutput,
               test_config::Quantizer, test_config::IN_HEIGHT,
               test_config::IN_WIDTH, test_config::IN_CH,
               test_config::REDUCE_PAR>
      split_reshape;
  split_reshape.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    // Provide dummy input data to keep the pipeline busy
    for (size_t i_heads_par = 0; i_heads_par < 1; i_heads_par++) {
      test_config::TInputWord input_struct;
      in_stream[i_heads_par].write(input_struct);
    }

    ActorStatus actor_status = split_reshape.step(in_stream, out_streamq, out_streamk, out_streamv);
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
  for (size_t heads_par = 0; heads_par < 2; heads_par++) {
    test_config::TOutputWord output_struct;
    while (out_streamq[heads_par].read_nb(output_struct))
      ;
  }

  for (size_t heads_par = 0; heads_par < 2; heads_par++) {
    test_config::TOutputWord output_struct;
    while (out_streamk[heads_par].read_nb(output_struct))
      ;
  }

  for (size_t heads_par = 0; heads_par < 2; heads_par++) {
    test_config::TOutputWord output_struct;
    while (out_streamv[heads_par].read_nb(output_struct))
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