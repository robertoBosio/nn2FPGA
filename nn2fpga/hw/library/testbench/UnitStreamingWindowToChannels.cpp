#include "StreamingWindowToChannels.hpp"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr size_t IN_DIM2 =
    test_config::OUT_DIM2 / (test_config::FH * test_config::FW);
static constexpr size_t CHANNEL_GROUPS = IN_DIM2 / test_config::DIM2_UNROLL;
static constexpr size_t FW_EXPAND =
    test_config::FW + (test_config::DIM1_UNROLL - 1) * test_config::STRIDE_W;
static constexpr size_t WINDOW_STREAMS = test_config::FH * FW_EXPAND;

void wrap_run(
    hls::stream<test_config::TInputWord> i_data[WINDOW_STREAMS],
    hls::stream<test_config::TOutputWord>
        o_data[test_config::DIM1_UNROLL]) {
  StreamingWindowToChannels<
      test_config::TInputWord, test_config::TInput, test_config::TOutputWord,
      test_config::TOutput, test_config::Quantizer, test_config::OUT_DIM0,
      test_config::OUT_DIM1, test_config::OUT_DIM2, test_config::FH,
      test_config::FW, test_config::STRIDE_W, test_config::DIM1_UNROLL,
      test_config::DIM2_UNROLL>
      window_to_channels;
  window_to_channels.run<0>(i_data, o_data);
}

bool test_run() {
  hls::stream<test_config::TInputWord> i_data[WINDOW_STREAMS];
  hls::stream<test_config::TOutputWord> o_data[test_config::DIM1_UNROLL];

  for (size_t out_h = 0; out_h < test_config::OUT_DIM0; out_h++) {
    for (size_t out_w = 0; out_w < test_config::OUT_DIM1;
         out_w += test_config::DIM1_UNROLL) {
      const size_t width_group = out_w / test_config::DIM1_UNROLL;
      for (size_t channel_group = 0; channel_group < CHANNEL_GROUPS;
           channel_group++) {
        for (size_t stream_index = 0; stream_index < WINDOW_STREAMS;
             stream_index++) {
          test_config::TInputWord input_word;
          for (size_t channel_lane = 0;
               channel_lane < test_config::DIM2_UNROLL; channel_lane++) {
            input_word[channel_lane] =
                test_config::input_tensor[out_h][width_group][channel_group]
                                         [stream_index][channel_lane];
          }
          i_data[stream_index].write(input_word);
        }
      }
    }
  }

  wrap_run(i_data, o_data);
  bool passed = true;

  for (size_t out_h = 0; out_h < test_config::OUT_DIM0; out_h++) {
    for (size_t out_w = 0; out_w < test_config::OUT_DIM1;
         out_w += test_config::DIM1_UNROLL) {
      for (size_t output_channel = 0; output_channel < test_config::OUT_DIM2;
           output_channel += test_config::DIM2_UNROLL) {
        for (size_t width_lane = 0; width_lane < test_config::DIM1_UNROLL;
             width_lane++) {
          test_config::TOutputWord output_word = o_data[width_lane].read();
          for (size_t channel_lane = 0;
               channel_lane < test_config::DIM2_UNROLL; channel_lane++) {
            const bool match =
                output_word[channel_lane] ==
                test_config::output_tensor[out_h]
                                          [out_w + width_lane]
                                          [output_channel + channel_lane];
            if (!match) {
              std::cout << "Mismatch at output (h=" << out_h
                        << ", w=" << out_w + width_lane
                        << ", channel=" << output_channel + channel_lane
                        << "): got " << output_word[channel_lane]
                        << ", expected "
                        << test_config::output_tensor[out_h]
                                                      [out_w + width_lane]
                                                      [output_channel + channel_lane]
                        << std::endl;
            }
            passed &= match;
          }
        }
      }
    }
  }

  for (size_t stream_index = 0; stream_index < WINDOW_STREAMS;
       stream_index++) {
    if (!i_data[stream_index].empty()) {
      std::cout << "Input stream " << stream_index
                << " not empty after run." << std::endl;
      passed = false;
    }
  }

  return passed;
}

bool test_step() {
  static constexpr size_t expectedII =
      test_config::OUT_DIM0 * (test_config::OUT_DIM1 / test_config::DIM1_UNROLL) *
      (test_config::OUT_DIM2 / test_config::DIM2_UNROLL);

  hls::stream<test_config::TInputWord> i_data[WINDOW_STREAMS];
  hls::stream<test_config::TOutputWord> o_data[test_config::DIM1_UNROLL];

  StreamingWindowToChannels<
      test_config::TInputWord, test_config::TInput, test_config::TOutputWord,
      test_config::TOutput, test_config::Quantizer, test_config::OUT_DIM0,
      test_config::OUT_DIM1, test_config::OUT_DIM2, test_config::FH,
      test_config::FW, test_config::STRIDE_W, test_config::DIM1_UNROLL,
      test_config::DIM2_UNROLL>
      window_to_channels;
  window_to_channels.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    // Supply every possible window stream so the actor is never input-stalled.
    for (size_t stream_index = 0; stream_index < WINDOW_STREAMS;
         stream_index++) {
      i_data[stream_index].write(test_config::TInputWord());
    }

    ActorStatus actor_status = window_to_channels.step(i_data, o_data);
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

    clock_cycles++;
    assert(clock_cycles < 10 * expectedII);
  }

  for (size_t width_lane = 0; width_lane < test_config::DIM1_UNROLL;
       width_lane++) {
    test_config::TOutputWord output_word;
    while (o_data[width_lane].read_nb(output_word))
      ;
  }

  const bool passed = II == expectedII;
  std::cout << "Expected II: " << expectedII << ", Measured II: " << II
            << std::endl;
  return passed;
}

int main(int argc, char **argv) {
  bool all_passed = true;
  all_passed &= test_run();

  // Testing the pipeline with csim only, as it is only relevant for FIFO depth
  // estimations.
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
