#include "StreamingConverter.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

using namespace test_config;

using TInputWord = std::array<test_config::TInput, test_config::CH_PAR>;
using TOutputWord = std::array<test_config::TOutput, test_config::W_PAR>;

// For this unit test, we build a StreamingConverter instance with the same
// template as in config
void wrap_run(hls::stream<TInputWord> in_data[test_config::W_PAR],
              hls::stream<TOutputWord> out_data[test_config::CH_PAR]) {
  StreamingConverter<TInputWord, test_config::TInput, TOutputWord,
                     test_config::TOutput, test_config::IN_HEIGHT,
                     test_config::IN_WIDTH, test_config::IN_CH,
                     test_config::W_PAR, test_config::CH_PAR>
      conv;
  conv.run<0>(in_data, out_data);
}

bool test_run() {
  // streams
  hls::stream<TInputWord> in_data[test_config::W_PAR];
  hls::stream<TOutputWord> out_data[test_config::CH_PAR];

  for (int r = 0; r < test_config::IN_HEIGHT; ++r) {
    for (int w = 0; w < (int)(test_config::IN_WIDTH / test_config::W_PAR);
         ++w) {
      for (int ch = 0; ch < (int)(test_config::IN_CH / test_config::CH_PAR);
           ++ch) {
        for (int w_i = 0; w_i < (int)test_config::W_PAR; ++w_i) {
          TInputWord word{};
          for (int ch_i = 0; ch_i < (int)test_config::CH_PAR; ++ch_i) {
            int val = r * test_config::IN_WIDTH * test_config::IN_CH;
            val += (w + w_i) * test_config::IN_CH;
            val += (ch * test_config::CH_PAR + ch_i);
            word[ch_i] = (TInput)val;
          }
          in_data[w_i].write(word);
        }
      }
    }
  }

  // Run DUT
  wrap_run(in_data, out_data);

  // Read outputs and check mapping:
  // We expect out_data[ch_i] to carry all values for that channel lane across
  // width streams
  bool ok = true;

  for (int r = 0; r < (int)test_config::IN_HEIGHT; ++r) {
    for (int w = 0; w < (int)(test_config::IN_WIDTH / test_config::W_PAR);
         ++w) {
      for (int ch = 0; ch < (int)(test_config::IN_CH / test_config::CH_PAR);
           ++ch) {
        for (int ch_i = 0; ch_i < (int)test_config::CH_PAR; ++ch_i) {
          // One output word from stream ch_i
          TOutputWord out_word = out_data[ch_i].read();
          for (int w_i = 0; w_i < (int)test_config::W_PAR; ++w_i) {
            int expected = r * test_config::IN_WIDTH * test_config::IN_CH;
            expected += (w + w_i) * test_config::IN_CH;
            expected += ch * test_config::CH_PAR + ch_i;
            int got = (int)out_word[w_i];
            if (got != expected) {
              std::cout << "Mismatch at (r=" << r << ", w=" << w
                        << ", ch=" << ch << ", w_i=" << w_i << "). got " << got
                        << ", expected " << expected << "\n";
              ok = false;
            }
          }
        }
      }
    }
  }

  // Ensure no extra outputs
  for (int s = 0; s < (int)test_config::CH_PAR; ++s) {
    if (!out_data[s].empty()) {
      std::cout << "Extra data in out_data[" << s << "] after reads.\n";
      ok = false;
    }
  }

  return ok;
}

// Optional: step-based test to verify schedule (similar to your CSDFG tests)
bool test_step() {
  hls::stream<TInputWord> in_data[test_config::W_PAR];
  hls::stream<TOutputWord> out_data[test_config::CH_PAR];

  StreamingConverter<TInputWord, test_config::TInput, TOutputWord,
                     test_config::TOutput, test_config::IN_HEIGHT,
                     test_config::IN_WIDTH, test_config::IN_CH,
                     test_config::W_PAR, test_config::CH_PAR>
      conv;
  conv.step_init(test_config::PIPELINE_DEPTH);

  // Expected total firings in one full sweep of (k, wg, chg)
  static constexpr size_t expectedII =
      (test_config::IN_CH / test_config::CH_PAR) *
      (test_config::IN_WIDTH / test_config::W_PAR) * test_config::IN_HEIGHT;

  // For step-based testing, we feed dummy data on every call and count that
  // step() fires expectedFirings times before returning to (0,0,0).
  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;

  while (true) {
    // Feed dummy data to all input streams
    for (int w_i = 0; w_i < (int)test_config::W_PAR; ++w_i) {
      TInputWord dummy{};
      in_data[w_i].write(dummy);
    }

    ActorStatus st = conv.step(in_data, out_data);

    // Update state and check for cycles
    std::vector<ActorStatus> actor_statuses = {st};
    std::vector<size_t> channel_quantities = {0};
    current_state = CSDFGState(actor_statuses, channel_quantities);

    if (visited_states.find(current_state) != visited_states.end()) {
      II = clock_cycles - visited_states[current_state];
      break;
    }
    visited_states.emplace(current_state, clock_cycles);
    clock_cycles++;
    assert(clock_cycles < 10 * expectedII);
  }

  // Flush the output stream
  for (int s = 0; s < (int)test_config::CH_PAR; ++s) {
    while (!out_data[s].empty()) {
      out_data[s].read();
    }
  }

  bool ok = (II == expectedII);
  std::cout << "test_step: expected II = " << expectedII
            << ", observed = " << II << "\n";
  return ok;
}

int main(int argc, char **argv) {
  bool all_ok = true;

  all_ok &= test_run();

  if (argc > 1 && std::string(argv[1]) == "csim") {
    all_ok &= test_step();
  }

  if (!all_ok) {
    std::cout << "Failed\n";
    return 1;
  } else {
    std::cout << "Passed\n";
    return 0;
  }
}