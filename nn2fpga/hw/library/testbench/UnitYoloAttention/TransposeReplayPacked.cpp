#include "YoloAttention/Transpose.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

void wrap_run(hls::stream<test_config::TInputWord> input_data[test_config::REDUCE_PAR],
              hls::stream<test_config::TOutputWord> output_data[1]) {
  TransposeRowColReplayPacked<
      test_config::TInputWord,
      test_config::TOutputWord,
      test_config::IN_HEIGHT,
      test_config::IN_WIDTH,
      test_config::DIM_P,
      test_config::REDUCE_PAR>
      transpose;
  transpose.run<0>(input_data, output_data);
}

bool test_run() {
  hls::stream<test_config::TInputWord> in_stream[test_config::REDUCE_PAR];
  hls::stream<test_config::TOutputWord> out_stream[1];

  for (size_t i_seq_group = 0;
       i_seq_group < test_config::IN_WIDTH / test_config::REDUCE_PAR;
       i_seq_group++) {
    for (size_t i_vrow = 0; i_vrow < test_config::IN_HEIGHT; i_vrow++) {
      for (size_t i_reduce = 0; i_reduce < test_config::REDUCE_PAR;
           i_reduce++) {
        test_config::TInputWord input_word;
        size_t i_seq = i_seq_group * test_config::REDUCE_PAR + i_reduce;
        input_word[0] = test_config::tensor[0][i_vrow][i_seq];
        in_stream[i_reduce].write(input_word);
      }
    }
  }

  wrap_run(in_stream, out_stream);

  bool flag = true;
  for (size_t i_pcol = 0; i_pcol < test_config::DIM_P; i_pcol++) {
    for (size_t i_vrow = 0; i_vrow < test_config::IN_HEIGHT; i_vrow++) {
      for (size_t i_seq_group = 0;
           i_seq_group < test_config::IN_WIDTH / test_config::REDUCE_PAR;
           i_seq_group++) {
        test_config::TOutputWord output_word = out_stream[0].read();
        for (size_t i_reduce = 0; i_reduce < test_config::REDUCE_PAR;
             i_reduce++) {
          size_t i_seq = i_seq_group * test_config::REDUCE_PAR + i_reduce;
          bool cmp = output_word[i_reduce] == test_config::tensor[0][i_vrow][i_seq];
          if (!cmp) {
            std::cout << "Mismatch at index (pcol=" << i_pcol
                      << ", vrow=" << i_vrow
                      << ", seq=" << i_seq
                      << ", reduce=" << i_reduce << "): "
                      << output_word[i_reduce] << " != "
                      << test_config::tensor[0][i_vrow][i_seq] << std::endl;
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
      (test_config::DIM_P + 1) * test_config::IN_HEIGHT *
          (test_config::IN_WIDTH / test_config::REDUCE_PAR);

  hls::stream<test_config::TInputWord> in_stream[test_config::REDUCE_PAR];
  hls::stream<test_config::TOutputWord> out_stream[1];

  TransposeRowColReplayPacked<
      test_config::TInputWord,
      test_config::TOutputWord,
      test_config::IN_HEIGHT,
      test_config::IN_WIDTH,
      test_config::DIM_P,
      test_config::REDUCE_PAR>
      transpose;
  transpose.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    for (size_t i_reduce = 0; i_reduce < test_config::REDUCE_PAR;
         i_reduce++) {
      in_stream[i_reduce].write(test_config::TInputWord());
    }

    ActorStatus actor_status = transpose.step(in_stream, out_stream);
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
  while (out_stream[0].read_nb(output_word))
    ;

  bool flag = (II == expectedII);
  std::cout << "Expected II: " << expectedII << ", Measured II: " << II
            << std::endl;
  return flag;
}

int main(int argc, char **argv) {
  bool all_passed = true;

  all_passed &= test_run();

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
