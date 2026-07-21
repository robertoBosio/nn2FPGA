#include "BandwidthAdjust.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include "utils/CSDFG_utils.hpp"

void wrap_run(
    hls::stream<test_config::TInputWord> input_data_stream[test_config::IN_DIM1_UNROLL],
    hls::stream<test_config::TOutputWord> output_data_stream[test_config::OUT_DIM1_UNROLL]) {
  test_config::BandwidthAdjust bandwidth_adjust;
  bandwidth_adjust.run<0>(input_data_stream, output_data_stream);
}

bool test_run(){

  // Prepare input and output streams
  hls::stream<test_config::TInputWord> in_stream[test_config::IN_DIM1_UNROLL];
  hls::stream<test_config::TOutputWord> out_stream[test_config::OUT_DIM1_UNROLL];

  // Fill input streams with test data
  for (size_t i_dim01 = 0; i_dim01 < test_config::IN_DIM0 * test_config::IN_DIM1;
       i_dim01 += test_config::IN_DIM1_UNROLL) {
    for (size_t i_dim2 = 0; i_dim2 < test_config::IN_DIM2; i_dim2 += test_config::IN_DIM2_UNROLL) {
      for (size_t i_dim1_par = 0; i_dim1_par < test_config::IN_DIM1_UNROLL; i_dim1_par++) {
        test_config::TInputWord input_word;
        for (size_t i_dim2_par = 0; i_dim2_par < test_config::IN_DIM2_UNROLL; i_dim2_par++) {
          input_word[i_dim2_par] = (i_dim01 + i_dim1_par) * test_config::IN_DIM2 + i_dim2 + i_dim2_par;
        }
        in_stream[i_dim1_par].write(input_word);
      }
    }
  }

  // Run the operator
  wrap_run(in_stream, out_stream);

  // Check output
  bool flag = true;
  for (size_t i_dim01 = 0; i_dim01 < test_config::IN_DIM0 * test_config::IN_DIM1;
       i_dim01 += test_config::OUT_DIM1_UNROLL) {
    for (size_t i_dim2 = 0; i_dim2 < test_config::IN_DIM2; i_dim2 += test_config::OUT_DIM2_UNROLL) {
      for (size_t i_dim1_par = 0; i_dim1_par < test_config::OUT_DIM1_UNROLL; i_dim1_par++) {
        test_config::TOutputWord output_word = out_stream[i_dim1_par].read();
        for (size_t i_dim2_par = 0; i_dim2_par < test_config::OUT_DIM2_UNROLL; i_dim2_par++) {
          bool cmp = (output_word[i_dim2_par] ==
                      (i_dim01 + i_dim1_par) * test_config::IN_DIM2 + i_dim2 + i_dim2_par);
          if (!cmp) {
            std::cout << "Mismatch at index (i_dim01=" << i_dim01 << ", i_dim2=" << i_dim2
                      << ", i_dim1_par=" << i_dim1_par << ", i_dim2_par=" << i_dim2_par
                      << "): " << output_word[i_dim2_par] << " != "
                      << (i_dim01 + i_dim1_par) * test_config::IN_DIM2 + i_dim2 + i_dim2_par
                      << std::endl;
          }
          flag &= cmp;
        }
      }
    }
  }

  return flag;
}

bool test_step() {

  static constexpr size_t expectedII =
      test_config::IN_DIM0 * test_config::IN_DIM1 * test_config::IN_DIM2 /
      (std::min(test_config::IN_DIM1_UNROLL, test_config::OUT_DIM1_UNROLL) *
       std::min(test_config::IN_DIM2_UNROLL, test_config::OUT_DIM2_UNROLL));

  // Prepare input and output streams
  hls::stream<test_config::TInputWord> in_stream[test_config::IN_DIM1_UNROLL];
  hls::stream<test_config::TOutputWord> out_stream[test_config::OUT_DIM1_UNROLL];

  test_config::BandwidthAdjust bandwidth_adjust;
  bandwidth_adjust.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    // Provide dummy input data to keep the pipeline busy
    for (size_t i_dim1_par = 0; i_dim1_par < test_config::IN_DIM1_UNROLL; i_dim1_par++) {
      test_config::TInputWord input_struct;
      in_stream[i_dim1_par].write(input_struct);
    }

    ActorStatus actor_status = bandwidth_adjust.step(in_stream, out_stream);
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
  for (size_t i_dim1_par = 0; i_dim1_par < test_config::OUT_DIM1_UNROLL; i_dim1_par++) {
    test_config::TOutputWord output_struct;
    while (out_stream[i_dim1_par].read_nb(output_struct))
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