#include "DequantQuant.hpp"
#include "StreamingUpsample.hpp"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

using TInputWord = std::array<test_config::TInput, test_config::DIM2_UNROLL>;
using TOutputWord = std::array<test_config::TOutput, test_config::DIM2_UNROLL>;
static constexpr size_t OUT_DIM0 =
    test_config::IN_DIM0 * test_config::SCALE_FACTOR;
static constexpr size_t OUT_DIM1 =
    test_config::IN_DIM1 * test_config::SCALE_FACTOR;

void wrap_run(hls::stream<TInputWord> i_data[test_config::IN_DIM1_UNROLL],
              hls::stream<TOutputWord> o_data[test_config::OUT_DIM1_UNROLL]) {
  // Wrapper for synthesis.
  StreamingUpsampleDim01<TInputWord, TOutputWord, test_config::Quantizer,
                    test_config::IN_DIM0, test_config::IN_DIM1,
                    test_config::IN_DIM2, OUT_DIM0, OUT_DIM1,
                    test_config::SCALE_FACTOR, test_config::IN_DIM1_UNROLL,
                    test_config::OUT_DIM1_UNROLL, test_config::DIM2_UNROLL>
      upsample;
  upsample.run<0>(i_data, o_data);
}

bool test_run() {

  // Prepare input and output streams
  hls::stream<TInputWord> in_stream[test_config::IN_DIM1_UNROLL];
  hls::stream<TOutputWord> out_stream[test_config::OUT_DIM1_UNROLL];

  // Fill input streams with test data
  for (size_t i_dim0 = 0; i_dim0 < test_config::IN_DIM0; i_dim0++) {
    for (size_t i_dim1 = 0; i_dim1 < test_config::IN_DIM1;
         i_dim1 += test_config::IN_DIM1_UNROLL) {
      for (size_t i_dim2 = 0; i_dim2 < test_config::IN_DIM2;
           i_dim2 += test_config::DIM2_UNROLL) {
        for (size_t i_dim1_par = 0; i_dim1_par < test_config::IN_DIM1_UNROLL;
             i_dim1_par++) {
          TInputWord input_data;
          for (size_t i_dim2_par = 0; i_dim2_par < test_config::DIM2_UNROLL;
               i_dim2_par++) {
            input_data[i_dim2_par] =
                test_config::input_tensor[0][i_dim2 + i_dim2_par][i_dim0]
                                         [i_dim1 + i_dim1_par];
          }
          in_stream[i_dim1_par].write(input_data);
        }
      }
    }
  }

  // Run pooling
  wrap_run(in_stream, out_stream);

  // Read and check output
  bool flag = true;
  for (size_t i_dim0 = 0; i_dim0 < OUT_DIM0; i_dim0++) {
    for (size_t i_dim1 = 0; i_dim1 < OUT_DIM1;
         i_dim1 += test_config::OUT_DIM1_UNROLL) {
      for (size_t i_dim2 = 0; i_dim2 < test_config::OUT_DIM2;
           i_dim2 += test_config::DIM2_UNROLL) {
        for (size_t i_dim1_par = 0; i_dim1_par < test_config::OUT_DIM1_UNROLL;
             i_dim1_par++) {
          TOutputWord output_data = out_stream[i_dim1_par].read();
          for (size_t i_dim2_par = 0; i_dim2_par < test_config::DIM2_UNROLL;
               i_dim2_par++) {

            // Check if the output data matches the expected result
            if (output_data[i_dim2_par] !=
                test_config::output_tensor[0][i_dim2 + i_dim2_par][i_dim0]
                                          [i_dim1 + i_dim1_par]) {
              std::cerr
                  << "Output mismatch at (" << i_dim0 << ", "
                  << i_dim1 + i_dim1_par << ", " << i_dim2 + i_dim2_par
                  << "): " << output_data[i_dim2_par] << " != "
                  << test_config::output_tensor[0][i_dim2 + i_dim2_par][i_dim0]
                                               [i_dim1 + i_dim1_par]
                  << std::endl;
              return false;
            }
          }
        }
      }
    }
  }

  // Ensure all streams are empty
  for (size_t i = 0; i < test_config::IN_DIM1_UNROLL; i++) {
    if (!in_stream[i].empty()) {
      std::cout << "Input stream not empty after run." << std::endl;
      flag = false;
    }
  }

  for (size_t i = 0; i < test_config::OUT_DIM1_UNROLL; i++) {
    if (!out_stream[i].empty()) {
      std::cout << "Output stream not empty after run." << std::endl;
      flag = false;
    }
  }

  std::cout << "Test run completed." << std::endl;
  return flag;
}

bool test_step() {

  static constexpr size_t expectedII =
      OUT_DIM0 * OUT_DIM1 * test_config::OUT_DIM2 /
      (test_config::DIM2_UNROLL * test_config::OUT_DIM1_UNROLL);

  // Create input and output streams
  hls::stream<TInputWord> i_data[test_config::IN_DIM1_UNROLL];
  hls::stream<TOutputWord> o_data[test_config::OUT_DIM1_UNROLL];

  // Create StreamingUpsample instance and initialize step state
  StreamingUpsampleDim01<TInputWord, TOutputWord, test_config::Quantizer,
                    test_config::IN_DIM0, test_config::IN_DIM1,
                    test_config::IN_DIM2, OUT_DIM0, OUT_DIM1,
                    test_config::SCALE_FACTOR, test_config::IN_DIM1_UNROLL,
                    test_config::OUT_DIM1_UNROLL, test_config::DIM2_UNROLL>
      upsample;
  upsample.step_init(expectedII);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {

    // Provide dummy input data to keep the pipeline busy
    for (size_t i = 0; i < test_config::IN_DIM1_UNROLL; i++) {
      i_data[i].write(TInputWord());
    }

    ActorStatus actor_status = upsample.step(i_data, o_data);
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
  for (size_t i = 0; i < test_config::OUT_DIM1_UNROLL; i++) {
    while (!o_data[i].empty()) {
      o_data[i].read();
    }
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
