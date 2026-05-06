#include "DequantQuant.hpp"
#include "StreamingSplit.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

void wrap_run(hls::stream<test_config::TInputWord>
                  input_data_stream[test_config::DIM1_UNROLL],
              hls::stream<test_config::TOutputWord>
                  output_data_stream0[test_config::DIM1_UNROLL],
              hls::stream<test_config::TOutputWord>
                  output_data_stream1[test_config::DIM1_UNROLL]) {
  test_config::StreamingSplit streaming_split;
  streaming_split.run<0>(input_data_stream, output_data_stream0,
                         output_data_stream1);
}

bool test_run() {

  // Prepare input and output streams
  hls::stream<test_config::TInputWord> in_stream[test_config::DIM1_UNROLL];
  hls::stream<test_config::TOutputWord> out_stream0[test_config::DIM1_UNROLL];
  hls::stream<test_config::TOutputWord> out_stream1[test_config::DIM1_UNROLL];

  // Fill input streams with test data
  for (size_t i_dim0 = 0; i_dim0 < test_config::IN_DIM0; i_dim0++) {
    for (size_t i_dim1 = 0; i_dim1 < test_config::IN_DIM1;
         i_dim1 += test_config::DIM1_UNROLL) {
      for (size_t i_dim2 = 0; i_dim2 < test_config::IN_DIM2;
           i_dim2 += test_config::DIM2_UNROLL) {
        for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
             i_dim1_par++) {
          test_config::TInputWord input_word;
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
  wrap_run(in_stream, out_stream0, out_stream1);

  // Check first tensor
  bool flag = true;
  for (size_t i_dim0 = 0; i_dim0 < test_config::OUT_DIM0_A; i_dim0++) {
    for (size_t i_dim1 = 0; i_dim1 < test_config::OUT_DIM1_A;
         i_dim1 += test_config::DIM1_UNROLL) {
      for (size_t i_dim2 = 0; i_dim2 < test_config::OUT_DIM2_A;
           i_dim2 += test_config::DIM2_UNROLL) {
        for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
             i_dim1_par++) {
          test_config::TOutputWord output_word0 =
              out_stream0[i_dim1_par].read();
          for (size_t i_dim2_par = 0; i_dim2_par < test_config::DIM2_UNROLL;
               i_dim2_par++) {
            bool cmp0 =
                (output_word0[i_dim2_par] ==
                 test_config::output_tensor0[0][i_dim2 + i_dim2_par][i_dim0]
                                            [i_dim1 + i_dim1_par]);
            if (!cmp0) {
              std::cout
                  << "Mismatch at output0 index (i_dim0=" << i_dim0
                  << ", i_dim1=" << i_dim1 << ", i_dim2=" << i_dim2
                  << ", i_dim1_par=" << i_dim1_par
                  << ", i_dim2_par=" << i_dim2_par
                  << "): " << output_word0[i_dim2_par] << " != "
                  << test_config::output_tensor0[0][i_dim2 + i_dim2_par][i_dim0]
                                                [i_dim1 + i_dim1_par]
                  << std::endl;
            }
            flag &= cmp0;
          }
        }
      }
    }
  }

  // Check second tensor
  for (size_t i_dim0 = 0; i_dim0 < test_config::OUT_DIM0_B; i_dim0++) {
    for (size_t i_dim1 = 0; i_dim1 < test_config::OUT_DIM1_B;
         i_dim1 += test_config::DIM1_UNROLL) {
      for (size_t i_dim2 = 0; i_dim2 < test_config::OUT_DIM2_B;
           i_dim2 += test_config::DIM2_UNROLL) {
        for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
             i_dim1_par++) {
          test_config::TOutputWord output_word1 =
              out_stream1[i_dim1_par].read();
          for (size_t i_dim2_par = 0; i_dim2_par < test_config::DIM2_UNROLL;
               i_dim2_par++) {
            bool cmp1 =
                (output_word1[i_dim2_par] ==
                 test_config::output_tensor1[0][i_dim2 + i_dim2_par][i_dim0]
                                            [i_dim1 + i_dim1_par]);
            if (!cmp1) {
              std::cout
                  << "Mismatch at output1 index (i_dim0=" << i_dim0
                  << ", i_dim1=" << i_dim1 << ", i_dim2=" << i_dim2
                  << ", i_dim1_par=" << i_dim1_par
                  << ", i_dim2_par=" << i_dim2_par
                  << "): " << output_word1[i_dim2_par] << " != "
                  << test_config::output_tensor1[0][i_dim2 + i_dim2_par][i_dim0]
                                                [i_dim1 + i_dim1_par]
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
      test_config::IN_DIM0 * test_config::IN_DIM1 * test_config::IN_DIM2 /
      (test_config::DIM1_UNROLL * test_config::DIM2_UNROLL);

  // Prepare input and output streams
  hls::stream<test_config::TInputWord> in_stream[test_config::DIM1_UNROLL];
  hls::stream<test_config::TOutputWord> out_stream0[test_config::DIM1_UNROLL];
  hls::stream<test_config::TOutputWord> out_stream1[test_config::DIM1_UNROLL];

  test_config::StreamingSplit streaming_split;
  streaming_split.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    // Provide dummy input data to keep the pipeline busy
    for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL; i_dim1_par++) {
      test_config::TInputWord input_struct;
      in_stream[i_dim1_par].write(input_struct);
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
  for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL; i_dim1_par++) {
    test_config::TOutputWord output_struct;
    while (out_stream0[i_dim1_par].read_nb(output_struct))
      ;
  }

  for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL; i_dim1_par++) {
    test_config::TOutputWord output_struct;
    while (out_stream1[i_dim1_par].read_nb(output_struct))
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