#include "YoloAttention/QKMatMul.hpp"
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

void wrap_run(hls::stream<test_config::TQInputWord> i_dataq[test_config::DIM_HEADS],
              hls::stream<test_config::TKInputWord> i_datak[test_config::DIM_HEADS],
              hls::stream<test_config::TOutputWord> o_data[test_config::DIM_HEADS]) {
  QKMatMul<test_config::TQInputWord, test_config::TQInput,
           test_config::TKInputWord, test_config::TKInput,
           test_config::TOutputWord, test_config::TOutput, test_config::TAcc,
           test_config::Quantizer, test_config::DIM_HEADS, test_config::DIM_Q,
           test_config::DIM_K, test_config::DIM_SEQ, test_config::REDUCE_PAR>
      matmul;
  matmul.run<0>(i_dataq, i_datak, o_data);
}

bool test_run() {
  hls::stream<test_config::TQInputWord> i_dataq[test_config::DIM_HEADS];
  hls::stream<test_config::TKInputWord> i_datak[test_config::DIM_HEADS];
  hls::stream<test_config::TOutputWord> o_data[test_config::DIM_HEADS];

  // Streaming q tensor
  for (size_t i_qrow = 0; i_qrow < test_config::DIM_Q; i_qrow++) {
    for (size_t i_heads = 0; i_heads < test_config::DIM_HEADS; i_heads++) {
      for (size_t i_reduce = 0;
           i_reduce < test_config::DIM_SEQ / test_config::REDUCE_PAR;
           i_reduce++) {
        test_config::TQInputWord q_word;
        for (size_t i_reduce_par = 0; i_reduce_par < test_config::REDUCE_PAR;
             i_reduce_par++) {
          q_word[i_reduce_par] =
              test_config::q_tensor[0][i_heads]
                                   [i_reduce * test_config::REDUCE_PAR +
                                    i_reduce_par][i_qrow];
        }
        i_dataq[i_heads].write(q_word);
      }
    }
  }

  // Streaming k tensor
  for (size_t i_kcol = 0; i_kcol < test_config::DIM_K; i_kcol++) {
    for (size_t i_heads = 0; i_heads < test_config::DIM_HEADS; i_heads++) {
      for (size_t i_reduce = 0;
           i_reduce < test_config::DIM_SEQ / test_config::REDUCE_PAR;
           i_reduce++) {
        test_config::TKInputWord k_word;
        for (size_t i_reduce_par = 0; i_reduce_par < test_config::REDUCE_PAR;
             i_reduce_par++) {
          k_word[i_reduce_par] =
              test_config::k_tensor[0][i_heads]
                                   [i_reduce * test_config::REDUCE_PAR +
                                    i_reduce_par][i_kcol];
        }
        i_datak[i_heads].write(k_word);
      }
    }
  }

  // Run the operator
  wrap_run(i_dataq, i_datak, o_data);

  // Check output
  bool flag = true;

  // Check output tensor.
  for (size_t i_qrow = 0; i_qrow < test_config::DIM_Q; i_qrow++) {
    for (size_t i_kcol = 0; i_kcol < test_config::DIM_K; i_kcol++) {
      for (size_t i_heads = 0; i_heads < test_config::DIM_HEADS; i_heads++) {
        test_config::TOutputWord data = o_data[i_heads].read();
        bool cmp;
        cmp =
            (data[0] == test_config::output_tensor[0][i_heads][i_qrow][i_kcol]);
        if (!cmp) {
          std::cout << "Mismatch at index (qrow=" << i_qrow
                    << ", kcol=" << i_kcol << ", head=" << i_heads
                    << ", reduce_par=" << 0 << "). got: " << data[0]
                    << ", expected: "
                    << test_config::output_tensor[0][i_heads][i_qrow][i_kcol]
                    << std::endl;
        }
        flag &= cmp;
      }
    }
  }

  // Empty shift output stream
  for (size_t i = 0; i < test_config::DIM_HEADS; i++) {
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
      test_config::DIM_HEADS * test_config::DIM_Q * test_config::DIM_K * test_config::DIM_SEQ /
      (test_config::REDUCE_PAR * test_config::HEADS_PAR);

  // Create input and output streams
  hls::stream<test_config::TQInputWord> i_dataq[test_config::DIM_HEADS];
  hls::stream<test_config::TKInputWord> i_datak[test_config::DIM_HEADS];
  hls::stream<test_config::TOutputWord> o_data[test_config::DIM_HEADS];
  QKMatMul<test_config::TQInputWord, test_config::TQInput,
           test_config::TKInputWord, test_config::TKInput,
           test_config::TOutputWord, test_config::TOutput, test_config::TAcc,
           test_config::Quantizer, test_config::DIM_HEADS, test_config::DIM_Q,
           test_config::DIM_K, test_config::DIM_SEQ, test_config::REDUCE_PAR>
      matmul;
  matmul.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {

    // Dummy input data
    for (size_t i = 0; i < test_config::DIM_HEADS; i++){
      i_dataq[i].write(test_config::TQInputWord());
      i_datak[i].write(test_config::TKInputWord());
    }

    ActorStatus actor_status =
        matmul.step(i_dataq, i_datak, o_data);
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
  for (size_t i = 0; i < test_config::DIM_HEADS; i++) {
    test_config::TOutputWord data;
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