#include "DequantQuant.hpp"
#include "StreamingAdd.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

using TInputWordA = std::array<test_config::TInputA, test_config::DIM2_UNROLL>;
using TInputWordB = std::array<test_config::TInputB, test_config::DIM2_UNROLL>;
using TOutputWord = std::array<test_config::TOutput, test_config::DIM2_UNROLL>;

void wrap_run(hls::stream<TInputWordA> i_data0[test_config::DIM1_UNROLL],
              hls::stream<TInputWordB> i_data1[test_config::DIM1_UNROLL],
              hls::stream<TOutputWord> o_data[test_config::DIM1_UNROLL]) {
  StreamingAdd<TInputWordA, test_config::TInputA, TInputWordB,
               test_config::TInputB, TOutputWord, test_config::TOutput,
               test_config::TAcc, test_config::Activation,
               test_config::Quantizer, test_config::AlignA, test_config::AlignB,
               test_config::DIM0, test_config::DIM1, test_config::DIM2,
               test_config::DIM1_UNROLL, test_config::DIM2_UNROLL>
      add;
  add.run<0>(i_data0, i_data1, o_data);
}

bool test_run() {
  hls::stream<TInputWordA> i_data0[test_config::DIM1_UNROLL];
  hls::stream<TInputWordB> i_data1[test_config::DIM1_UNROLL];
  hls::stream<TOutputWord> o_data[test_config::DIM1_UNROLL];

  // Streaming input tensors.
  for (size_t i_dim0 = 0; i_dim0 < test_config::DIM0; i_dim0++) {
    for (size_t i_dim1 = 0; i_dim1 < test_config::DIM1;
         i_dim1 += test_config::DIM1_UNROLL) {
      for (size_t i_dim2 = 0; i_dim2 < test_config::DIM2;
           i_dim2 += test_config::DIM2_UNROLL) {
        for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
             i_dim1_par++) {
          TInputWordA input_data0;
          TInputWordB input_data1;
          for (size_t i_dim2_par = 0; i_dim2_par < test_config::DIM2_UNROLL;
               i_dim2_par++) {
            input_data0[i_dim2_par] =
                test_config::input_tensor0[0][i_dim2 + i_dim2_par][i_dim0]
                                          [i_dim1 + i_dim1_par];
            input_data1[i_dim2_par] =
                test_config::input_tensor1[0][i_dim2 + i_dim2_par][i_dim0]
                                          [i_dim1 + i_dim1_par];
          }
          i_data0[i_dim1_par].write(input_data0);
          i_data1[i_dim1_par].write(input_data1);
        }
      }
    }
  }
  // Run the operator
  wrap_run(i_data0, i_data1, o_data);

  // Check output
  bool flag = true;

  // Check output tensor.
  for (size_t i_dim0 = 0; i_dim0 < test_config::DIM0; i_dim0++) {
    for (size_t i_dim1 = 0; i_dim1 < test_config::DIM1;
         i_dim1 += test_config::DIM1_UNROLL) {
      for (size_t i_dim2 = 0; i_dim2 < test_config::DIM2;
           i_dim2 += test_config::DIM2_UNROLL) {
        for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
             i_dim1_par++) {
          TOutputWord data = o_data[i_dim1_par].read();
          bool cmp;
          for (size_t i_dim2_par = 0; i_dim2_par < test_config::DIM2_UNROLL;
               i_dim2_par++) {
            cmp = (data[i_dim2_par] ==
                   test_config::output_tensor[0][i_dim2 + i_dim2_par][i_dim0]
                                             [i_dim1 + i_dim1_par]);
            if (!cmp) {
              std::cout
                  << "Mismatch at index (h=" << i_dim0 << ", w=" << i_dim1
                  << ", ich=" << i_dim2 << ", w_par=" << i_dim1_par
                  << ", ch_par=" << i_dim2_par << "). got: " << data[i_dim2_par]
                  << ", expected: "
                  << test_config::output_tensor[0][i_dim2 + i_dim2_par][i_dim0]
                                               [i_dim1 + i_dim1_par]
                  << std::endl;
            }
            flag &= cmp;
          }
        }
      }
    }
  }

  // Empty shift output stream
  for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
       i_dim1_par++) {
    if (!o_data[i_dim1_par].empty()) {
      flag = false;
      std::cout << "Output stream " << i_dim1_par << " not empty after reading."
                << std::endl;
    }
  }

  return flag;
}

bool test_step() {

  static constexpr size_t expectedII =
      test_config::DIM0 * test_config::DIM1 * test_config::DIM2 /
      (test_config::DIM2_UNROLL * test_config::DIM1_UNROLL);

  // Create input and output streams
  hls::stream<TInputWordA> i_data0[test_config::DIM1_UNROLL];
  hls::stream<TInputWordB> i_data1[test_config::DIM1_UNROLL];
  hls::stream<TOutputWord> o_data[test_config::DIM1_UNROLL];
  StreamingAdd<TInputWordA, test_config::TInputA, TInputWordB,
               test_config::TInputB, TOutputWord, test_config::TOutput,
               test_config::TAcc, test_config::Activation,
               test_config::Quantizer, test_config::AlignA, test_config::AlignB,
               test_config::DIM0, test_config::DIM1, test_config::DIM2,
               test_config::DIM1_UNROLL, test_config::DIM2_UNROLL>
      add;
  add.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {

    // Dummy input data
    for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
         i_dim1_par++) {
      i_data0[i_dim1_par].write(TInputWordA());
      i_data1[i_dim1_par].write(TInputWordB());
    }

    ActorStatus actor_status = add.step(i_data0, i_data1, o_data);
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
  for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
       i_dim1_par++) {
    TOutputWord data;
    while (o_data[i_dim1_par].read_nb(data))
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