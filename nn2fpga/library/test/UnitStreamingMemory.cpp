#include "DequantQuant.hpp"
#include "StreamingMemory.hpp"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include <array>
#include <cassert>
#include <iostream>

using TOutputWord =
    std::array<test_config::TOutput,
               test_config::IN_CH_PAR * test_config::OUT_CH_PAR>;
using TInputWord = std::array<test_config::TInput, 1>;
static constexpr size_t TIMES =
    test_config::OUT_HEIGHT * test_config::OUT_WIDTH / test_config::W_PAR;
static constexpr size_t CH_GROUPS =
    test_config::OUT_CH * test_config::IN_CH /
    (test_config::OUT_CH_PAR * test_config::IN_CH_PAR);
static constexpr size_t FHW = test_config::FH * test_config::FW;
static constexpr size_t PAR = test_config::IN_CH_PAR * test_config::OUT_CH_PAR;

void wrap_run(hls::stream<TInputWord> i_shift_data[1],
              hls::stream<TOutputWord> o_data[FHW],
              hls::stream<TInputWord> o_shift_data[1]) {
  StreamingMemory<TInputWord, test_config::TOutput, TOutputWord,
                  test_config::DATA_PER_WORD, test_config::DATA_TO_SHIFT, TIMES,
                  test_config::OUT_CH, test_config::IN_CH, test_config::FW,
                  test_config::FH, test_config::OUT_CH_PAR,
                  test_config::IN_CH_PAR>
      mem;
  if (test_config::DATA_TO_SHIFT > 0) {
    mem.run(i_shift_data, o_data, o_shift_data);
  } else {
    mem.run(i_shift_data, o_data);
  }
}

bool test_run() {

  // Create input and output streams
  hls::stream<TInputWord> i_shift_data[1];
  hls::stream<TInputWord> o_shift_data[1];
  hls::stream<TOutputWord> o_data[FHW];

  for (const auto &word : test_config::packed_weights) {
    TInputWord word_array;
    word_array[0] = word;
    i_shift_data[0].write(word_array);
  }

  // Run the operator
  wrap_run(i_shift_data, o_data, o_shift_data);

  // Check output
  bool flag = true;
  for (size_t i_times = 0; i_times < TIMES; i_times++) {
    for (size_t i_ch_groups = 0; i_ch_groups < CH_GROUPS; i_ch_groups++) {
      for (size_t i_fhw = 0; i_fhw < FHW; i_fhw++) {
        TOutputWord actual_word = o_data[i_fhw].read();
        for (size_t i_par = 0; i_par < PAR; i_par++) {
          test_config::TOutput actual_data = actual_word[i_par];
          test_config::TOutput expected_data =
              test_config::weight_tensor[i_ch_groups][i_par][i_fhw];
          if (actual_data != expected_data) {
            flag = false;
            std::cout << "Mismatch at (" << i_ch_groups << ", " << i_par << ", "
                      << i_fhw << "): "
                      << "Expected: " << expected_data << ", "
                      << "Actual: " << actual_data << std::endl;
          }
        }
      }
    }
  }

  // Ensure all output streams are empty
  for (size_t i_fhw = 0; i_fhw < FHW; i_fhw++) {
    if (!o_data[i_fhw].empty()) {
      flag = false;
      std::cout << "Output stream " << i_fhw
                << " not empty after reading all data." << std::endl;
    }
  }

  // Ensure input stream is empty
  if (!i_shift_data[0].empty()) {
    flag = false;
    std::cout << "Input stream not empty after processing." << std::endl;
  }

  // Ensure data shifted correctly.
  if (o_shift_data[0].size() != test_config::DATA_TO_SHIFT) {
    flag = false;
    std::cout << "Shifted data size mismatch. Expected: "
              << test_config::DATA_TO_SHIFT << ", Actual: "
              << o_shift_data[0].size() << std::endl;
  }
  return flag;
}

bool test_step(){
  // Each group produces output every cycle.
  constexpr size_t expectedII = TIMES * CH_GROUPS;

  // Create input and output streams
  hls::stream<TInputWord> i_shift_data[1];
  hls::stream<TInputWord> o_shift_data[1];
  hls::stream<TOutputWord> o_data[FHW];

  // Instantiate the operator
  StreamingMemory<TInputWord, test_config::TOutput, TOutputWord,
                  test_config::DATA_PER_WORD, test_config::DATA_TO_SHIFT, TIMES,
                  test_config::OUT_CH, test_config::IN_CH, test_config::FW,
                  test_config::FH, test_config::OUT_CH_PAR,
                  test_config::IN_CH_PAR>
      mem(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    ActorStatus actor_status;
    if (test_config::DATA_TO_SHIFT > 0) {
      actor_status = mem.step(i_shift_data, o_data, o_shift_data);
    } else {
      actor_status = mem.step(i_shift_data, o_data);
    }
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
  for (size_t i_fhw = 0; i_fhw < FHW; ++i_fhw) {
    while (!o_data[i_fhw].empty()) {
      TOutputWord output_word = o_data[i_fhw].read();
      (void)output_word; // Suppress unused variable warning
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