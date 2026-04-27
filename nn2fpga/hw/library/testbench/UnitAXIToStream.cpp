#include "AXIToStream.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

using TOutputWord = std::array<test_config::TOutput, test_config::DIM2_UNROLL>;

void wrap_run(hls::stream<test_config::TInputWord> &in_stream,
              hls::stream<TOutputWord> out_stream[test_config::DIM1_UNROLL]) {
#pragma HLS INTERFACE axis port = in_stream

  AXIToStream<test_config::TInputWord, test_config::TInput, TOutputWord,
              test_config::TOutput, test_config::Quantizer,
              test_config::DATA_PER_WORD, test_config::DIM0, test_config::DIM1,
              test_config::DIM2, test_config::DIM1_UNROLL,
              test_config::DIM2_UNROLL>
      producer;
  producer.run<0>(in_stream, out_stream);
}

bool test_run() {

  // Prepare input and output streams
  hls::stream<test_config::TInputWord> in_stream;
  hls::stream<TOutputWord> out_stream[test_config::DIM1_UNROLL];

  // Prepare input data: fill every pixel with a counter following HWC format
  size_t data_in_word = 0;
  test_config::TInputWord input_axi_struct;
  for (size_t i_dim01 = 0; i_dim01 < test_config::DIM0 * test_config::DIM1;
       i_dim01++) {
    for (size_t i_dim2 = 0; i_dim2 < test_config::DIM2; i_dim2++) {
      // Fill the input structure with the value 1 for each channel
      input_axi_struct.data.range(
          test_config::TOutput::width * (data_in_word + 1) - 1,
          test_config::TOutput::width * data_in_word) =
          i_dim01 * test_config::DIM2 + i_dim2;
      data_in_word++;

      if (data_in_word >= test_config::DATA_PER_WORD) {
        // If we have filled the current word, write it to the stream
        if (i_dim01 == test_config::DIM0 * test_config::DIM1 - 1 &&
            i_dim2 == test_config::DIM2 - 1) {
          // If this is the last input, set the last signal
          input_axi_struct.last = true;
        } else {
          input_axi_struct.last = false;
        }
        in_stream.write(input_axi_struct);

        data_in_word = 0;
        input_axi_struct.data = 0; // Reset for next word
      }
    }
  }

  // Run producer
  wrap_run(in_stream, out_stream);

  // Read and check output
  bool flag = true;
  for (size_t i_dim01 = 0; i_dim01 < test_config::DIM0 * test_config::DIM1;
       i_dim01 += test_config::DIM1_UNROLL) {
    for (size_t i_dim2 = 0; i_dim2 < test_config::DIM2;
         i_dim2 += test_config::DIM2_UNROLL) {
      for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL; i_dim1_par++) {
        TOutputWord out_word = out_stream[i_dim1_par].read();
        for (size_t i_dim2_par = 0; i_dim2_par < test_config::DIM2_UNROLL; i_dim2_par++) {
          // Each channel should have the average value of 1
          flag &= (out_word[i_dim2_par] ==
                   (i_dim01 + i_dim1_par) * test_config::DIM2 + i_dim2 + i_dim2_par);
        }
      }
    }
  }

  return flag;
}

  bool test_step() {
    // This function tests the step() method of AXIToStream.

    static constexpr size_t expectedII =
        test_config::DIM0 * test_config::DIM1 * test_config::DIM2 /
        (test_config::DIM2_UNROLL * test_config::DIM1_UNROLL);

    // Instantiate the operator
    AXIToStream<test_config::TInputWord, test_config::TInput, TOutputWord,
                test_config::TOutput, test_config::Quantizer,
                test_config::DATA_PER_WORD, test_config::DIM0,
                test_config::DIM1, test_config::DIM2, test_config::DIM1_UNROLL,
                test_config::DIM2_UNROLL>
        producer;
    producer.step_init(test_config::PIPELINE_DEPTH);

    // Prepare input and output streams
    hls::stream<test_config::TInputWord> in_stream;
    hls::stream<TOutputWord> out_stream[test_config::DIM1_UNROLL];

    std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
    CSDFGState current_state;
    size_t clock_cycles = 0;
    size_t II = 0;
    while (true) {
      test_config::TInputWord input_data;
      in_stream.write(input_data);
      ActorStatus actor_status = producer.step(in_stream, out_stream);
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
    for (size_t  i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL; i_dim1_par++) {
      TOutputWord out_word;
      while (out_stream[i_dim1_par].read_nb(out_word))
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

    // Testing the pipeline with csim only, as it is only relevant for fifo
    // depth estimations
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