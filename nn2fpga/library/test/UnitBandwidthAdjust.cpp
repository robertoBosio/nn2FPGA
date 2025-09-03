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
    hls::stream<test_config::TInputWord> input_data_stream[test_config::IN_W_PAR],
    hls::stream<test_config::TOutputWord> output_data_stream[test_config::OUT_W_PAR]) {
  test_config::BandwidthAdjust bandwidth_adjust;
  bandwidth_adjust.run(input_data_stream, output_data_stream);
}

template <class BandwidthAdjust>
bool test_run(){

  // Prepare input and output streams
  hls::stream<test_config::TInputWord> in_stream[test_config::IN_W_PAR];
  hls::stream<test_config::TOutputWord> out_stream[test_config::OUT_W_PAR];

  // Fill input streams with test data
  for (size_t i = 0; i < test_config::IN_HEIGHT * test_config::IN_WIDTH;
       i += test_config::IN_W_PAR) {
    for (size_t ch = 0; ch < test_config::IN_CH; ch += test_config::IN_CH_PAR) {
      for (size_t w_par = 0; w_par < test_config::IN_W_PAR; w_par++) {
        test_config::TInputWord input_word;
        for (size_t ch_par = 0; ch_par < test_config::IN_CH_PAR; ch_par++) {
          input_word[ch_par] = (i + w_par) * test_config::IN_CH + ch + ch_par;
        }
        in_stream[w_par].write(input_word);
      }
    }
  }

  // Run the operator
  wrap_run(in_stream, out_stream);

  // Check output
  bool flag = true;
  for (size_t i = 0; i < test_config::IN_HEIGHT * test_config::IN_WIDTH;
       i += test_config::OUT_W_PAR) {
    for (size_t ch = 0; ch < test_config::IN_CH; ch += test_config::OUT_CH_PAR) {
      for (size_t w_par = 0; w_par < test_config::OUT_W_PAR; w_par++) {
        test_config::TOutputWord output_word = out_stream[w_par].read();
        for (size_t ch_par = 0; ch_par < test_config::OUT_CH_PAR; ch_par++) {
          bool cmp = (output_word[ch_par] ==
                      (i + w_par) * test_config::IN_CH + ch + ch_par);
          if (!cmp) {
            std::cout << "Mismatch at index (i=" << i << ", ch=" << ch
                      << ", w_par=" << w_par << ", ch_par=" << ch_par
                      << "): " << output_word[ch_par] << " != "
                      << (i + w_par) * test_config::IN_CH + ch + ch_par
                      << std::endl;
          }
          flag &= cmp;
        }
      }
    }
  }

  return flag;
}

template <class BandwidthAdjust> bool test_step() {

  static constexpr size_t expectedII =
      test_config::IN_HEIGHT * test_config::IN_WIDTH * test_config::IN_CH /
      (std::min(test_config::IN_W_PAR, test_config::OUT_W_PAR) *
       std::min(test_config::IN_CH_PAR, test_config::OUT_CH_PAR));

  // Prepare input and output streams
  hls::stream<test_config::TInputWord> in_stream[test_config::IN_W_PAR];
  hls::stream<test_config::TOutputWord> out_stream[test_config::OUT_W_PAR];

  BandwidthAdjust bandwidth_adjust(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    // Provide dummy input data to keep the pipeline busy
    for (size_t i_w_par = 0; i_w_par < test_config::IN_W_PAR; i_w_par++) {
      test_config::TInputWord input_struct;
      in_stream[i_w_par].write(input_struct);
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
  for (size_t w_par = 0; w_par < test_config::OUT_W_PAR; w_par++) {
    test_config::TOutputWord output_struct;
    while (out_stream[w_par].read_nb(output_struct))
      ;
  }

  bool flag = (II == expectedII);
  std::cout << "Expected II: " << expectedII << ", Measured II: " << II
            << std::endl;
  return flag;
}

int main(int argc, char **argv) {

  bool all_passed = true;

  all_passed &= test_run<test_config::BandwidthAdjust>();

  // Testing the pipeline with csim only, as it is only relevant for fifo depth
  // estimations
  if (argc > 1 && std::string(argv[1]) == "csim") {
    all_passed &= test_step<test_config::BandwidthAdjust>();
  }

  if (!all_passed) {
    std::cout << "Failed." << std::endl;
  } else {
    std::cout << "Passed." << std::endl;
  }

  return all_passed ? 0 : 1;
}