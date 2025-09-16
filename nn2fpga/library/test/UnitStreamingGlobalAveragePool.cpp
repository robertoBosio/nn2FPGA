#include "DequantQuant.hpp"
#include "StreamingGlobalAveragePool.hpp"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

using TInputWord = std::array<test_config::TInput, test_config::OUT_CH_PAR>;
using TOutputWord = std::array<test_config::TOutput, test_config::OUT_CH_PAR>;

void wrap_run(hls::stream<TInputWord> i_data[1],
              hls::stream<TOutputWord> o_data[1]) {
  // Wrapper for synthesis.
  StreamingGlobalAveragePool<TInputWord, test_config::TInput, TOutputWord,
                             test_config::TOutput, test_config::TAcc,
                             test_config::TDiv, test_config::Quantizer,
                             test_config::IN_HEIGHT, test_config::IN_WIDTH,
                             test_config::OUT_CH, test_config::OUT_CH_PAR>
      pool;
  pool.run(i_data, o_data);
}

bool test_run() {

  // Prepare input and output streams
  hls::stream<TInputWord> in_stream[1];
  hls::stream<TOutputWord> out_stream[1];

  // Prepare input data: fill every channel with 1, expect sum = 4 (2x2 window)
  for (size_t ih = 0; ih < test_config::IN_HEIGHT; ih++) {
    for (size_t iw = 0; iw < test_config::IN_WIDTH; iw++) {
      for (size_t c = 0; c < test_config::OUT_CH; c += test_config::OUT_CH_PAR) {
        TInputWord input_struct;
        for (size_t j = 0; j < test_config::OUT_CH_PAR; j++) {
          input_struct[j] = test_config::input_tensor[0][c + j][ih][iw];
        }
        in_stream[0].write(input_struct);
      }
    }
  }

  // Run pooling
  wrap_run(in_stream, out_stream);

  // Read and check output
  bool flag = true;
  for (size_t c = 0; c < test_config::OUT_CH; c += test_config::OUT_CH_PAR) {
    TOutputWord output_struct;
    out_stream[0].read(output_struct);
    for (size_t j = 0; j < test_config::OUT_CH_PAR; j++) {
      // Each channel should have the average value of 1
      bool condition =
          (output_struct[j] == test_config::output_tensor[0][c + j][0][0]);
      if (!condition) {
        std::cout << "Mismatch at channel " << (c + j) << ": got "
                  << output_struct[j] << ", expected "
                  << test_config::output_tensor[0][c + j][0][0] << std::endl;
      }
      flag &= condition;
    }
  }

  // Ensure all streams are empty
  if (!in_stream[0].empty()) {
    std::cout << "Input stream not empty after run." << std::endl;
    flag = false;
  }

  if (!out_stream[0].empty()) {
    std::cout << "Output stream not empty after run." << std::endl;
    flag = false;
  }

  return flag;
}

bool test_step() {

  static constexpr size_t expectedII =
      test_config::IN_HEIGHT * test_config::IN_WIDTH * test_config::OUT_CH /
      (test_config::OUT_CH_PAR);

  // Create input and output streams
  hls::stream<TInputWord> i_data[1];
  hls::stream<TOutputWord> o_data[1];

  // Run the global average pooling
  StreamingGlobalAveragePool<TInputWord, test_config::TInput, TOutputWord,
                             test_config::TOutput, test_config::TAcc,
                             test_config::TDiv, test_config::Quantizer,
                             test_config::IN_HEIGHT, test_config::IN_WIDTH,
                             test_config::OUT_CH, test_config::OUT_CH_PAR>
      pool(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {

    // Provide dummy input data to keep the pipeline busy
    i_data[0].write(TInputWord());

    ActorStatus actor_status = pool.step(i_data, o_data);
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
  TOutputWord output_struct;
  while (o_data[0].read_nb(output_struct))
    ;

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
