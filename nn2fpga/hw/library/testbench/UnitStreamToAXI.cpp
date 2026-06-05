#include "StreamToAXI.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include "utils/CSDFG_utils.hpp"
#include "utils/HLS_utils.hpp"
#include "test_config.hpp"

using TInputWord = std::array<test_config::TInput, test_config::DIM2_UNROLL>;

void wrap_run(hls::stream<TInputWord> input_data_stream[test_config::DIM1_UNROLL],
              hls::stream<test_config::TOutputWord> &output_data_stream) {
#pragma HLS INTERFACE axis port = output_data_stream
  // Wrapper function to call the run() method of StreamToAXI, for synthesis.
  StreamToAXI<TInputWord, test_config::TInput, test_config::TOutputWord,
              test_config::Quantizer, test_config::ITER,
              test_config::DATA_PER_WORD, test_config::DIM0,
              test_config::DIM1, test_config::DIM2, test_config::DIM1_UNROLL,
              test_config::DIM2_UNROLL>
      consumer;
  consumer.run<0>(input_data_stream, output_data_stream);
}

bool test_run() {
  // This function tests the run() method of StreamToAXI.

  // Prepare input and output streams
  hls::stream<TInputWord> in_stream[test_config::DIM1_UNROLL];
  hls::stream<test_config::TOutputWord> out_stream;

  for (size_t i_dim01 = 0; i_dim01 < test_config::DIM0 * test_config::DIM1;
       i_dim01 += test_config::DIM1_UNROLL) {
    for (size_t i_dim2 = 0; i_dim2 < test_config::DIM2;
         i_dim2 += test_config::DIM2_UNROLL) {
      for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
           i_dim1_par++) {
        TInputWord input_word;
        for (size_t i_dim2_par = 0; i_dim2_par < test_config::DIM2_UNROLL;
             i_dim2_par++) {
          size_t index = (i_dim01 + i_dim1_par) * test_config::DIM2 + (i_dim2 + i_dim2_par);
          input_word[i_dim2_par] = test_config::TInput(index);
        }
        in_stream[i_dim1_par].write(input_word);
      }
    }
  }

  // Run consumer
  wrap_run(in_stream, out_stream);

  // Read and check output
  bool flag = true;
  size_t data_in_word = 0;
  size_t output_word_index = 0;
  constexpr size_t total_elems = test_config::DIM0 * test_config::DIM1 *
                                 test_config::DIM2;
  constexpr size_t output_words =
      (total_elems + test_config::DATA_PER_WORD - 1) /
      test_config::DATA_PER_WORD;
  constexpr size_t last_word_elems =
      total_elems - ((output_words - 1) * test_config::DATA_PER_WORD);
  constexpr size_t last_word_bytes =
      last_word_elems * data_width_v<test_config::TInput> / 8;
  test_config::TOutputWord output_word;
  for (size_t i_dim01 = 0; i_dim01 < test_config::DIM0 * test_config::DIM1;
       i_dim01++) {
    for (size_t i_dim2 = 0; i_dim2 < test_config::DIM2; i_dim2++) {
      if (data_in_word == 0) {
        // Read the output structure from the stream
        output_word = out_stream.read();
        bool expected_last = output_word_index == output_words - 1;
        decltype(output_word.keep) expected_keep = 0;
        if (expected_last) {
          for (size_t i = 0; i < last_word_bytes; i++) {
            expected_keep[i] = 1;
          }
        } else {
          expected_keep = ~expected_keep;
        }

        flag &= (output_word.last == expected_last);
        flag &= (output_word.keep == expected_keep);
        if (output_word.last != expected_last || output_word.keep != expected_keep) {
          std::cout << "AXI metadata mismatch at word " << output_word_index
                    << " Expected last: " << expected_last
                    << ", got: " << output_word.last
                    << " Expected keep: " << expected_keep
                    << ", got: " << output_word.keep << std::endl;
        }
        output_word_index++;
      }

      constexpr size_t W = data_width_v<test_config::TInput>;

      ap_uint<W> bits_read = output_word.data.range(
          W * (data_in_word + 1) - 1, W * data_in_word);

      ap_uint<W> expected_bits = get_raw_bits(
          test_config::TInput(i_dim01 * test_config::DIM2 + i_dim2));
      flag &= (bits_read == expected_bits);

      if (!flag) {
        std::cout << "Mismatch at (i,c)=(" << i_dim01 << "," << i_dim2 << ")"
                  << " Expected: " << expected_bits << ", Got: " << bits_read
                  << std::endl;
      }
      data_in_word++;

      if (data_in_word >= test_config::DATA_PER_WORD) {
        data_in_word = 0;
      }
    }
  }

  flag &= (output_word_index == output_words);
  flag &= out_stream.empty();

  return flag;
}

bool test_step() {
  // This function tests the step() method of StreamToAXI

  size_t expectedII = test_config::DIM0 * test_config::DIM1 *
                      test_config::DIM2 /
                      (test_config::DIM2_UNROLL * test_config::DIM1_UNROLL);
  if ((test_config::DIM0 * test_config::DIM1 * test_config::DIM2) %
          (test_config::DATA_PER_WORD) !=
      0) {
    expectedII += 1;
  }

  // Instantiate the operator
  StreamToAXI<TInputWord, test_config::TInput, test_config::TOutputWord,
              test_config::Quantizer, test_config::ITER,
              test_config::DATA_PER_WORD, test_config::DIM0, test_config::DIM1,
              test_config::DIM2, test_config::DIM1_UNROLL,
              test_config::DIM2_UNROLL>
      consumer;
  consumer.step_init(test_config::PIPELINE_DEPTH);

  // Prepare input and output streams
  hls::stream<TInputWord> in_stream[test_config::DIM1_UNROLL];
  hls::stream<test_config::TOutputWord> out_stream;

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    TInputWord input_data;
    for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL; i_dim1_par++) {
      in_stream[i_dim1_par].write(input_data);
    }
    ActorStatus actor_status = consumer.step(in_stream, out_stream);
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
  test_config::TOutputWord output_word;
  while (out_stream.read_nb(output_word))
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
