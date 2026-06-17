#include "StreamingCircularLineBuffer.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include <array>
#include <iostream>

static constexpr size_t OUT_HEIGHT =
    1 + (test_config::IN_HEIGHT + test_config::PAD_T + test_config::PAD_B -
         test_config::DIL_H * (test_config::FH - 1) - 1) /
            test_config::STRIDE_H;
static constexpr size_t OUT_WIDTH =
    1 + (test_config::IN_WIDTH + test_config::PAD_L + test_config::PAD_R -
         test_config::DIL_W * (test_config::FW - 1) - 1) /
            test_config::STRIDE_W;
static constexpr size_t FW_EXPAND =
    test_config::FW + (test_config::W_PAR - 1) * test_config::STRIDE_W;
static constexpr size_t PADDED_HEIGHT =
    test_config::IN_HEIGHT + test_config::PAD_T + test_config::PAD_B;
static constexpr size_t PADDED_WIDTH =
    test_config::IN_WIDTH + test_config::PAD_L + test_config::PAD_R;
static constexpr size_t PADDED_WORDS_PER_ROW = PADDED_WIDTH / test_config::W_PAR;
static constexpr size_t CH_GROUPS = test_config::IN_CH / test_config::CH_PAR;

void wrap_run(
    hls::stream<test_config::TWord> i_data[test_config::W_PAR],
    hls::stream<test_config::TWord> o_data[test_config::FH * FW_EXPAND]) {
  StreamingCircularLineBuffer<
      test_config::TWord, test_config::TData, test_config::IN_HEIGHT,
      test_config::IN_WIDTH, test_config::IN_CH, test_config::FH,
      test_config::FW, test_config::STRIDE_H, test_config::STRIDE_W,
      test_config::DIL_H, test_config::DIL_W, test_config::PAD_T,
      test_config::PAD_L, test_config::PAD_B, test_config::PAD_R,
      test_config::W_PAR, test_config::CH_PAR, test_config::PAD_VALUE>
      linebuffer;

  linebuffer.run<0>(i_data, o_data);
}

void wrap_step(
    hls::stream<test_config::TWord> i_data[test_config::W_PAR],
    hls::stream<test_config::TWord> o_data[test_config::FH * FW_EXPAND]) {
  StreamingCircularLineBuffer<
      test_config::TWord, test_config::TData, test_config::IN_HEIGHT,
      test_config::IN_WIDTH, test_config::IN_CH, test_config::FH,
      test_config::FW, test_config::STRIDE_H, test_config::STRIDE_W,
      test_config::DIL_H, test_config::DIL_W, test_config::PAD_T,
      test_config::PAD_L, test_config::PAD_B, test_config::PAD_R,
      test_config::W_PAR, test_config::CH_PAR, test_config::PAD_VALUE>
      linebuffer;

  linebuffer.step_init(1);
  for (size_t i = 0; i < PADDED_HEIGHT * PADDED_WORDS_PER_ROW * CH_GROUPS;
       i++) {
    linebuffer.step(i_data, o_data);
  }
}

static void fill_input_streams(
    hls::stream<test_config::TWord> i_data[test_config::W_PAR]) {
  for (size_t h = 0; h < test_config::IN_HEIGHT; h++) {
    for (size_t w = 0; w < test_config::IN_WIDTH; w += test_config::W_PAR) {
      for (size_t ch = 0; ch < test_config::IN_CH;
           ch += test_config::CH_PAR) {
        for (size_t w_par = 0; w_par < test_config::W_PAR; w_par++) {
          test_config::TWord input_word;
          for (size_t ch_par = 0; ch_par < test_config::CH_PAR; ch_par++) {
            input_word[ch_par] =
                test_config::input_tensor[0][ch + ch_par][h][w + w_par];
          }
          i_data[w_par].write(input_word);
        }
      }
    }
  }
}

static bool check_output_streams(
    hls::stream<test_config::TWord> o_data[test_config::FH * FW_EXPAND]) {
  bool passed = true;

  for (size_t out_h = 0; out_h < OUT_HEIGHT; out_h++) {
    for (size_t out_w = 0; out_w < OUT_WIDTH; out_w += test_config::W_PAR) {
      for (size_t ch = 0; ch < test_config::IN_CH;
           ch += test_config::CH_PAR) {
        for (size_t fh = 0; fh < test_config::FH; fh++) {
          for (size_t fw = 0; fw < FW_EXPAND; fw++) {
            const int in_h = static_cast<int>(out_h * test_config::STRIDE_H + fh) -
                             test_config::PAD_T;
            const int in_w = static_cast<int>(out_w * test_config::STRIDE_W + fw) -
                             test_config::PAD_L;
            test_config::TWord data = o_data[fh * FW_EXPAND + fw].read();

            for (size_t ch_par = 0; ch_par < test_config::CH_PAR; ch_par++) {
              const bool is_within_tensor = in_h >= 0 &&
                                            in_h < test_config::IN_HEIGHT &&
                                            in_w >= 0 &&
                                            in_w < test_config::IN_WIDTH;
              const test_config::TData expected = is_within_tensor
                  ? test_config::input_tensor[0][ch + ch_par][in_h][in_w]
                  : test_config::TData(test_config::PAD_VALUE);
              const bool match = data[ch_par] == expected;
              if (!match) {
                std::cout << "Mismatch at output window (out_h=" << out_h
                          << ", out_w=" << out_w << ", ch=" << ch
                          << ", fh=" << fh << ", fw=" << fw
                          << ", ch_par=" << ch_par << "). got: "
                          << data[ch_par] << ", expected: " << expected
                          << std::endl;
              }
              passed &= match;
            }
          }
        }
      }
    }
  }

  for (size_t i = 0; i < test_config::FH * FW_EXPAND; i++) {
    if (!o_data[i].empty()) {
      std::cout << "Output stream " << i << " not empty after checking."
                << std::endl;
      passed = false;
    }
  }

  return passed;
}

bool test_run() {
  hls::stream<test_config::TWord> i_data[test_config::W_PAR];
  hls::stream<test_config::TWord> o_data[test_config::FH * FW_EXPAND];

  fill_input_streams(i_data);
  wrap_run(i_data, o_data);
  return check_output_streams(o_data);
}

bool test_step() {
  static constexpr size_t expectedII =
      PADDED_HEIGHT * PADDED_WORDS_PER_ROW * CH_GROUPS;

  hls::stream<test_config::TWord> i_data[test_config::W_PAR];
  hls::stream<test_config::TWord> o_data[test_config::FH * FW_EXPAND];

  StreamingCircularLineBuffer<
      test_config::TWord, test_config::TData, test_config::IN_HEIGHT,
      test_config::IN_WIDTH, test_config::IN_CH, test_config::FH,
      test_config::FW, test_config::STRIDE_H, test_config::STRIDE_W,
      test_config::DIL_H, test_config::DIL_W, test_config::PAD_T,
      test_config::PAD_L, test_config::PAD_B, test_config::PAD_R,
      test_config::W_PAR, test_config::CH_PAR, test_config::PAD_VALUE>
      linebuffer;
  linebuffer.step_init(test_config::PIPELINE_DEPTH);

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;

  while (true) {
    for (size_t i_w_par = 0; i_w_par < test_config::W_PAR; i_w_par++) {
      test_config::TWord input_word;
      i_data[i_w_par].write(input_word);
    }

    ActorStatus actor_status = linebuffer.step(i_data, o_data);

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

    for (size_t i = 0; i < test_config::FH * FW_EXPAND; i++) {
      test_config::TWord output_word;
      while (o_data[i].read_nb(output_word))
        ;
    }

    clock_cycles++;
    assert(clock_cycles < 10 * expectedII);
  }

  bool flag = (II == expectedII);
  std::cout << "Expected II: " << expectedII << ", Measured II: " << II
            << std::endl;
  return flag;
}

int main(int argc, char **argv) {
  bool passed = true;
  passed &= test_run();

  if (argc > 1 && std::string(argv[1]) == "csim") {
    passed &= test_step();
  }

  if (passed) {
    std::cout << "Passed." << std::endl;
  } else {
    std::cout << "Failed." << std::endl;
  }

  return passed ? 0 : 1;
}
