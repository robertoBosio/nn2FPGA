#ifdef NN2FPGA_CATAPULT
#include "catapult/Axis.hpp"
#include "catapult/DequantQuant.hpp"
#include "catapult/StreamToAXI.hpp"
#include <ac_channel.h>
#include <mc_scverify.h>
template <typename T> using TestStream = ac_channel<T>;
#else
#include "vitis/StreamToAXI.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "vitis/utils/CSDFG_utils.hpp"
#include "vitis/utils/HLS_utils.hpp"
template <typename T> using TestStream = hls::stream<T>;
#endif
#include "test_config.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <string>
#ifndef NN2FPGA_CATAPULT
#include <unordered_map>
#include <vector>
#endif

using TInputWord = std::array<test_config::TInput, test_config::DIM2_UNROLL>;

#ifdef NN2FPGA_CATAPULT
#pragma hls_design top
#endif
void wrap_run(TestStream<TInputWord> input_data_stream[test_config::DIM1_UNROLL],
              TestStream<test_config::TOutputWord> &output_data_stream) {
#ifndef NN2FPGA_CATAPULT
#pragma HLS INTERFACE axis port = output_data_stream
#endif
  StreamToAXIS<TInputWord, test_config::TInput, test_config::TOutputWord,
              test_config::Quantizer, test_config::ITER,
              test_config::DATA_PER_WORD, test_config::DIM0,
              test_config::DIM1, test_config::DIM2, test_config::DIM1_UNROLL,
              test_config::DIM2_UNROLL>
      consumer;
  consumer.run<0>(input_data_stream, output_data_stream);
}

bool test_run() {
  TestStream<TInputWord> in_stream[test_config::DIM1_UNROLL];
  TestStream<test_config::TOutputWord> out_stream;

  for (size_t i_dim01 = 0; i_dim01 < test_config::DIM0 * test_config::DIM1;
       i_dim01 += test_config::DIM1_UNROLL) {
    for (size_t i_dim2 = 0; i_dim2 < test_config::DIM2;
         i_dim2 += test_config::DIM2_UNROLL) {
      for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
           i_dim1_par++) {
        TInputWord input_word;
        for (size_t i_dim2_par = 0; i_dim2_par < test_config::DIM2_UNROLL;
             i_dim2_par++) {
          const size_t index =
              (i_dim01 + i_dim1_par) * test_config::DIM2 + i_dim2 + i_dim2_par;
          input_word[i_dim2_par] = test_config::TInput(index);
        }
        in_stream[i_dim1_par].write(input_word);
      }
    }
  }

#ifdef NN2FPGA_CATAPULT
  CCS_DESIGN(wrap_run)(in_stream, out_stream);
#else
  wrap_run(in_stream, out_stream);
#endif

  bool flag = true;
  size_t data_in_word = 0;
  size_t output_word_index = 0;
  constexpr size_t total_elems =
      test_config::DIM0 * test_config::DIM1 * test_config::DIM2;
  constexpr size_t output_words =
      (total_elems + test_config::DATA_PER_WORD - 1) /
      test_config::DATA_PER_WORD;
  constexpr size_t last_word_elems =
      total_elems - ((output_words - 1) * test_config::DATA_PER_WORD);
#ifdef NN2FPGA_CATAPULT
  constexpr size_t input_width = test_config::TInput::width;
#else
  constexpr size_t input_width = data_width_v<test_config::TInput>;
#endif
  constexpr size_t last_word_bytes =
      last_word_elems * input_width / 8;
  test_config::TOutputWord output_word;
  for (size_t i_dim01 = 0; i_dim01 < test_config::DIM0 * test_config::DIM1;
       i_dim01++) {
    for (size_t i_dim2 = 0; i_dim2 < test_config::DIM2; i_dim2++) {
      if (data_in_word == 0) {
        output_word = out_stream.read();
        const bool expected_last = output_word_index == output_words - 1;
        using TKeep = decltype(output_word.keep);
        TKeep expected_keep = expected_last ? TKeep(0) : TKeep(-1);
        if (expected_last) {
          for (size_t i = 0; i < last_word_bytes; i++) {
            expected_keep[static_cast<unsigned>(i)] = 1;
          }
        }

        flag &= (output_word.last == expected_last);
        flag &= (output_word.keep == expected_keep);
        if (output_word.last != expected_last ||
            output_word.keep != expected_keep) {
          std::cout << "AXI metadata mismatch at word " << output_word_index
                    << " Expected last: " << expected_last
                    << ", got: " << output_word.last
                    << " Expected keep: " << expected_keep
                    << ", got: " << output_word.keep << std::endl;
        }
        output_word_index++;
      }

#ifdef NN2FPGA_CATAPULT
      const test_config::TInput value_read = test_config::TInput(
          output_word.data.template slc<test_config::TInput::width>(
              static_cast<unsigned>(data_in_word * test_config::TInput::width)));
      const test_config::TInput expected_value =
          test_config::TInput(i_dim01 * test_config::DIM2 + i_dim2);
      const bool match = value_read == expected_value;
#else
      constexpr size_t W = data_width_v<test_config::TInput>;
      const ap_uint<W> bits_read = output_word.data.range(
          W * (data_in_word + 1) - 1, W * data_in_word);
      const ap_uint<W> expected_bits = get_raw_bits(
          test_config::TInput(i_dim01 * test_config::DIM2 + i_dim2));
      const bool match = bits_read == expected_bits;
#endif

      flag &= match;
      if (!match) {
        std::cout << "Mismatch at (i,c)=(" << i_dim01 << "," << i_dim2
                  << ")" << std::endl;
      }
      data_in_word++;
      if (data_in_word >= test_config::DATA_PER_WORD) {
        data_in_word = 0;
      }
    }
  }

  flag &= (output_word_index == output_words);
  for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
       i_dim1_par++) {
    flag &= in_stream[i_dim1_par].empty();
  }
  flag &= out_stream.empty();
  return flag;
}

#ifndef NN2FPGA_CATAPULT
bool test_maxi_run() {
  constexpr size_t total_elems =
      test_config::DIM0 * test_config::DIM1 * test_config::DIM2;
  constexpr size_t output_words =
      (total_elems + test_config::DATA_PER_WORD - 1) /
      test_config::DATA_PER_WORD;
  using TMAXIWord = ap_uint<test_config::AXI_DATAWIDTH>;

  hls::stream<TInputWord> in_stream[test_config::DIM1_UNROLL];
  TMAXIWord output_words_maxi[output_words] = {};
  for (size_t i_dim01 = 0; i_dim01 < test_config::DIM0 * test_config::DIM1;
       i_dim01 += test_config::DIM1_UNROLL) {
    for (size_t i_dim2 = 0; i_dim2 < test_config::DIM2;
         i_dim2 += test_config::DIM2_UNROLL) {
      for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
           i_dim1_par++) {
        TInputWord input_word;
        for (size_t i_dim2_par = 0; i_dim2_par < test_config::DIM2_UNROLL;
             i_dim2_par++) {
          const size_t index =
              (i_dim01 + i_dim1_par) * test_config::DIM2 + i_dim2 + i_dim2_par;
          input_word[i_dim2_par] = test_config::TInput(index);
        }
        in_stream[i_dim1_par].write(input_word);
      }
    }
  }

  StreamToMAXI<TInputWord, test_config::TInput, TMAXIWord,
               test_config::Quantizer, test_config::ITER,
               test_config::DATA_PER_WORD, test_config::DIM0,
               test_config::DIM1, test_config::DIM2, test_config::DIM1_UNROLL,
               test_config::DIM2_UNROLL>
      consumer;
  consumer.run<0>(in_stream, output_words_maxi);

  bool flag = true;
  constexpr size_t W = data_width_v<test_config::TInput>;
  for (size_t i = 0; i < total_elems; i++) {
    const size_t word = i / test_config::DATA_PER_WORD;
    const size_t lane = i % test_config::DATA_PER_WORD;
    const ap_uint<W> bits_read =
        output_words_maxi[word].range(W * (lane + 1) - 1, W * lane);
    const ap_uint<W> expected_bits = get_raw_bits(test_config::TInput(i));
    flag &= (bits_read == expected_bits);
  }
  return flag;
}

bool test_step() {
  size_t expectedII = test_config::DIM0 * test_config::DIM1 * test_config::DIM2 /
                      (test_config::DIM2_UNROLL * test_config::DIM1_UNROLL);
  if ((test_config::DIM0 * test_config::DIM1 * test_config::DIM2) %
          test_config::DATA_PER_WORD !=
      0) {
    expectedII += 1;
  }

  StreamToAXIS<TInputWord, test_config::TInput, test_config::TOutputWord,
              test_config::Quantizer, test_config::ITER,
              test_config::DATA_PER_WORD, test_config::DIM0, test_config::DIM1,
              test_config::DIM2, test_config::DIM1_UNROLL,
              test_config::DIM2_UNROLL>
      consumer;
  consumer.step_init(test_config::PIPELINE_DEPTH);

  hls::stream<TInputWord> in_stream[test_config::DIM1_UNROLL];
  hls::stream<test_config::TOutputWord> out_stream;
  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    TInputWord input_data;
    for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
         i_dim1_par++) {
      in_stream[i_dim1_par].write(input_data);
    }
    const ActorStatus actor_status = consumer.step(in_stream, out_stream);
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

  test_config::TOutputWord output_word;
  while (out_stream.read_nb(output_word))
    ;

  const bool flag = (II == expectedII);
  std::cout << "Expected II: " << expectedII << ", Measured II: " << II
            << std::endl;
  return flag;
}
#endif

#ifdef NN2FPGA_CATAPULT
CCS_MAIN(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif
  bool all_passed = test_run();

#ifndef NN2FPGA_CATAPULT
  all_passed &= test_maxi_run();
  if (argc > 1 && std::string(argv[1]) == "csim") {
    all_passed &= test_step();
  }
#endif

  std::cout << (all_passed ? "Passed." : "Failed.") << std::endl;
#ifdef NN2FPGA_CATAPULT
  CCS_RETURN(all_passed ? 0 : 1);
#else
  return all_passed ? 0 : 1;
#endif
}
