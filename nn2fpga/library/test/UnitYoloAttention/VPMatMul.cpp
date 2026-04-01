#include "YoloAttention/VPMatMul.hpp"
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

void wrap_run(hls::stream<test_config::TVInputWord> i_datav[test_config::DIM_HEADS],
              hls::stream<test_config::TPInputWord> i_datap[test_config::DIM_HEADS],
              hls::stream<test_config::TOutputWord> o_data[test_config::DIM_HEADS]) {
  VPMatMul<test_config::TVInputWord, test_config::TVInput,
           test_config::TPInputWord, test_config::TPInput,
           test_config::TOutputWord, test_config::TOutput, test_config::TAcc,
           test_config::Quantizer, test_config::DIM_HEADS, test_config::DIM_V,
           test_config::DIM_P, test_config::DIM_SEQ, test_config::REDUCE_PAR>
      matmul;
  matmul.run<0>(i_datav, i_datap, o_data);
}

bool test_run() {
  hls::stream<test_config::TVInputWord> i_datav[test_config::DIM_HEADS];
  hls::stream<test_config::TPInputWord> i_datap[test_config::DIM_HEADS];
  hls::stream<test_config::TOutputWord> o_data[test_config::DIM_HEADS];

  // Streaming v tensor
  for (size_t i_vrow = 0; i_vrow < test_config::DIM_V; i_vrow++) {
    for (size_t i_heads = 0; i_heads < test_config::DIM_HEADS; i_heads++) {
      for (size_t i_reduce = 0;
           i_reduce < test_config::DIM_SEQ / test_config::REDUCE_PAR;
           i_reduce++) {
        test_config::TVInputWord v_word;
        for (size_t i_reduce_par = 0; i_reduce_par < test_config::REDUCE_PAR;
             i_reduce_par++) {
          v_word[i_reduce_par] =
              test_config::v_tensor[0][i_heads][i_vrow]
                                   [i_reduce * test_config::REDUCE_PAR +
                                    i_reduce_par];
        }
        i_datav[i_heads].write(v_word);
      }
    }
  }

  // Streaming p tensor
  for (size_t i_pcol = 0; i_pcol < test_config::DIM_P; i_pcol++) {
    for (size_t i_heads = 0; i_heads < test_config::DIM_HEADS; i_heads++) {
      for (size_t i_reduce = 0;
           i_reduce < test_config::DIM_SEQ / test_config::REDUCE_PAR;
           i_reduce++) {
        test_config::TPInputWord p_word;
        for (size_t i_reduce_par = 0; i_reduce_par < test_config::REDUCE_PAR;
             i_reduce_par++) {
          p_word[i_reduce_par] =
              test_config::p_tensor[0][i_heads][i_pcol]
                                   [i_reduce * test_config::REDUCE_PAR +
                                    i_reduce_par];
        }
        i_datap[i_heads].write(p_word);
      }
    }
  }

  // Run the operator
  wrap_run(i_datav, i_datap, o_data);

  // Check output
  bool flag = true;

  // Check output tensor.
  for (size_t i_pcol = 0; i_pcol < test_config::DIM_P; i_pcol++) {
    for (size_t i_vrow = 0; i_vrow < test_config::DIM_V; i_vrow++) {
      for (size_t i_heads = 0; i_heads < test_config::DIM_HEADS; i_heads++) {
        test_config::TOutputWord data = o_data[i_heads].read();
        bool cmp;
        cmp =
            (data[0] == test_config::output_tensor[0][i_heads][i_vrow][i_pcol]);
        if (!cmp) {
          std::cout << "Mismatch at index (vrow=" << i_vrow
                    << ", pcol=" << i_pcol << ", head=" << i_heads
                    << ", reduce_par=" << 0 << "). got: " << data[0]
                    << ", expected: "
                    << test_config::output_tensor[0][i_heads][i_vrow][i_pcol]
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
      test_config::DIM_HEADS * test_config::DIM_V * test_config::DIM_P *
      test_config::DIM_SEQ / (test_config::REDUCE_PAR);

  // Create input and output streams
  hls::stream<test_config::TVInputWord> i_datav[test_config::DIM_HEADS];
  hls::stream<test_config::TPInputWord> i_datap[test_config::DIM_HEADS];
  hls::stream<test_config::TOutputWord> o_data[test_config::DIM_HEADS];
  VPMatMul<test_config::TVInputWord, test_config::TVInput,
           test_config::TPInputWord, test_config::TPInput,
           test_config::TOutputWord, test_config::TOutput, test_config::TAcc,
           test_config::Quantizer, test_config::DIM_HEADS, test_config::DIM_V,
           test_config::DIM_P, test_config::DIM_SEQ, test_config::REDUCE_PAR>
      matmul;
  matmul.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {

    // Dummy input data
    for (size_t i = 0; i < test_config::DIM_HEADS; i++){
      i_datav[i].write(test_config::TVInputWord());
      i_datap[i].write(test_config::TPInputWord());
    }

    ActorStatus actor_status =
        matmul.step(i_datav, i_datap, o_data);
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