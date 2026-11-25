#ifdef NN2FPGA_CATAPULT
#include "catapult/StreamingReLU.hpp"
#include <ac_channel.h>

template <typename T>
using TestStream = ac_channel<T>;
#else
#include "vitis/DequantQuant.hpp"
#include "vitis/StreamingReLU.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "vitis/utils/CSDFG_utils.hpp"

template <typename T>
using TestStream = hls::stream<T>;
#endif

#include <array>
#include <cassert>
#include <iostream>
#ifndef NN2FPGA_CATAPULT
#include <unordered_map>
#endif

#include "test_config.hpp"

using TInputWord = std::array<test_config::TInput, test_config::DIM2_UNROLL>;
using TOutputWord = TInputWord;

#ifdef NN2FPGA_CATAPULT
#pragma hls_design top
#endif
void wrap_run(TestStream<TInputWord> i_data[test_config::DIM1_UNROLL],
              TestStream<TOutputWord> o_data[test_config::DIM1_UNROLL]) {
  StreamingReLU<TInputWord, test_config::TInput, TOutputWord,
                test_config::TOutput, test_config::Quantizer, test_config::DIM0,
                test_config::DIM1, test_config::DIM2, test_config::DIM1_UNROLL,
                test_config::DIM2_UNROLL>
      streaming_relu;
  streaming_relu.run<0>(i_data, o_data);
}

bool test_run() {

  // Prepare input and output streams
  TestStream<TInputWord> in_stream[test_config::DIM1_UNROLL];
  TestStream<TOutputWord> out_stream[test_config::DIM1_UNROLL];

  // Fill input streams with test data
  for (size_t i_dim0 = 0; i_dim0 < test_config::DIM0; i_dim0++) {
    for (size_t i_dim1 = 0; i_dim1 < test_config::DIM1;
         i_dim1 += test_config::DIM1_UNROLL) {
      for (size_t i_dim2 = 0; i_dim2 < test_config::DIM2;
           i_dim2 += test_config::DIM2_UNROLL) {
        for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
             i_dim1_par++) {
          TInputWord input_word;
          for (size_t i_dim2_par = 0; i_dim2_par < test_config::DIM2_UNROLL;
               i_dim2_par++) {
            input_word[i_dim2_par] =
                test_config::input_tensor[0][i_dim2 + i_dim2_par][i_dim0]
                                         [i_dim1 + i_dim1_par];
          }
          in_stream[i_dim1_par].write(input_word);
        }
      }
    }
  }

  // Run the operator
  wrap_run(in_stream, out_stream);

  // Check output
  bool flag = true;
  for (size_t i_dim0 = 0; i_dim0 < test_config::DIM0; i_dim0++) {
    for (size_t i_dim1 = 0; i_dim1 < test_config::DIM1;
         i_dim1 += test_config::DIM1_UNROLL) {
      for (size_t i_dim2 = 0; i_dim2 < test_config::DIM2;
           i_dim2 += test_config::DIM2_UNROLL) {
        for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
             i_dim1_par++) {
          TOutputWord output_word = out_stream[i_dim1_par].read();
          for (size_t i_dim2_par = 0; i_dim2_par < test_config::DIM2_UNROLL;
               i_dim2_par++) {
            bool cmp =
                (output_word[i_dim2_par] ==
                 test_config::output_tensor[0][i_dim2 + i_dim2_par][i_dim0]
                                           [i_dim1 + i_dim1_par]);
            if (!cmp) {
              std::cout
                  << "Mismatch at index (i_dim0=" << i_dim0
                  << ", i_dim1=" << i_dim1 << ", i_dim2=" << i_dim2
                  << ", i_dim1_par=" << i_dim1_par
                  << ", i_dim2_par=" << i_dim2_par
                  << "): " << output_word[i_dim2_par] << " != "
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

  return flag;
}

#ifndef NN2FPGA_CATAPULT
bool test_step() {

  static constexpr size_t expectedII =
      test_config::DIM0 * test_config::DIM1 * test_config::DIM2 /
      (test_config::DIM1_UNROLL * test_config::DIM2_UNROLL);

  // Prepare input and output streams
  hls::stream<TInputWord> in_stream[test_config::DIM1_UNROLL];
  hls::stream<TOutputWord> out_stream[test_config::DIM1_UNROLL];

  StreamingReLU<TInputWord, test_config::TInput, TOutputWord,
                test_config::TOutput, test_config::Quantizer, test_config::DIM0,
                test_config::DIM1, test_config::DIM2, test_config::DIM1_UNROLL,
                test_config::DIM2_UNROLL>
      streaming_relu;
  streaming_relu.step_init(test_config::PIPELINE_DEPTH);
  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    // Provide dummy input data to keep the pipeline busy
    for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
         i_dim1_par++) {
      TInputWord input_struct;
      in_stream[i_dim1_par].write(input_struct);
    }

    ActorStatus actor_status = streaming_relu.step(in_stream, out_stream);
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
    TOutputWord output_struct;
    while (out_stream[i_dim1_par].read_nb(output_struct))
      ;
  }

  bool flag = (II == expectedII);
  std::cout << "Expected II: " << expectedII << ", Measured II: " << II
            << std::endl;
  return flag;
}
#endif

int main(int argc, char **argv) {

  bool all_passed = true;

  all_passed &= test_run();

#ifndef NN2FPGA_CATAPULT
  // Testing the pipeline with csim only, as it is only relevant for fifo depth
  // estimations.
  if (argc > 1 && std::string(argv[1]) == "csim") {
    all_passed &= test_step();
  }
#endif

  if (!all_passed) {
    std::cout << "Failed." << std::endl;
  } else {
    std::cout << "Passed." << std::endl;
  }

  return all_passed ? 0 : 1;
}
