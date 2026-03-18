#include "StreamingMatMul.hpp"
#include "DequantQuant.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

using TInputWordA = std::array<test_config::TInputA, test_config::W_PAR>;
using TInputWordB = std::array<test_config::TInputB, test_config::W_PAR>;
using TOutputWord = ap_axis<test_config::CH_PAR * test_config::OUTPUT_DATAWIDTH, 0, 0, 0>;

void wrap_run(hls::stream<TInputWordA> in_data_A[test_config::CH_PAR],
              hls::stream<TInputWordB> in_data_B[test_config::CH_PAR],
              hls::stream<TOutputWord>& mat_out) {
  StreamingMatMul<TInputWordA, test_config::TInputA,
                  TInputWordB, test_config::TInputB,
                  TOutputWord, test_config::TOutput,
                  test_config::TAcc, test_config::Quantizer,
                  test_config::IN_HEIGHT_A, test_config::IN_WIDTH_A,
                  test_config::IN_WIDTH_B,
                  test_config::IN_CH_A, test_config::W_PAR,
                  test_config::CH_PAR>
      matmul;
  matmul.run<0>(in_data_A, in_data_B, mat_out);
}

bool test_run() {
  hls::stream<TInputWordA> in_data_A[test_config::CH_PAR];
  hls::stream<TInputWordB> in_data_B[test_config::CH_PAR];
  hls::stream<TOutputWord> mat_out;

  // Stream A: read at j==0, order: r → k (step W_PAR) → i_ch
  for (int r = 0; r < (int)test_config::IN_HEIGHT_A; r++) {
    for (int k = 0; k < (int)test_config::IN_WIDTH_A; k += test_config::W_PAR) {
     
      for (int ch = 0; ch < (int)test_config::IN_CH_A; ch += test_config::CH_PAR) {
        
        for (int i_ch = 0; i_ch < (int)test_config::CH_PAR; i_ch++) {
          TInputWordA pktA;
          for (int i_w = 0; i_w < (int)test_config::W_PAR; i_w++){
            pktA[i_w] = test_config::input_tensor0[0][ch + i_ch][r][k + i_w];
          }
          in_data_A[i_ch].write(pktA);
        }
      }
    }
  }

  // Stream B: read at r==0, order: j → k (step W_PAR) → i_ch
  // local_B[j][i_ch][k] = B[0][i_ch][k][j]  ← k/j transposed
  for (int j = 0; j < (int)test_config::IN_WIDTH_B; j++) {
    for (int k = 0; k < (int)test_config::IN_WIDTH_A; k += test_config::W_PAR) {
      
      for (int ch = 0; ch < (int)test_config::IN_CH_A; ch += test_config::CH_PAR) {
       
        for (int i_ch = 0; i_ch < (int)test_config::CH_PAR; i_ch++) {
           TInputWordB pktB;
          for (int i_w = 0; i_w < (int)test_config::W_PAR; i_w++){
            pktB[i_w] = test_config::input_tensor1[0][ch + i_ch][k + i_w][j];
          }
          in_data_B[i_ch].write(pktB);
        }
      }
    }
  }

  wrap_run(in_data_A, in_data_B, mat_out);

  // Read output: IN_CH/CH_PAR packets per (r, j)
  bool flag = true;
  for (int r = 0; r < (int)test_config::IN_HEIGHT_A; r++) {
    for (int j = 0; j < (int)test_config::IN_WIDTH_B; j++) {
      for (int ch = 0; ch < (int)test_config::IN_CH_A; ch += test_config::CH_PAR) {
        TOutputWord pkt = mat_out.read();
        for (int i_ch = 0; i_ch < (int)test_config::CH_PAR; i_ch++) {
          test_config::TOutput got = pkt.data.range(
              (i_ch * test_config::OUTPUT_DATAWIDTH) + test_config::OUTPUT_DATAWIDTH - 1,
               i_ch * test_config::OUTPUT_DATAWIDTH);
          test_config::TOutput expected =
              test_config::output_tensor[0][ch + i_ch][r][j];
          bool cmp = (got == expected);
          if (!cmp)
            std::cout << "Mismatch at (r=" << r << ", j=" << j
                      << ", ch=" << ch + i_ch
                      << "). got: " << got
                      << ", expected: " << expected << std::endl;
          flag &= cmp;
        }
      }
    }
  }
  if (!mat_out.empty()) {
    flag = false;
    std::cout << "Output stream not empty after reading." << std::endl;
  }
  return flag;
}

bool test_step() {
  // One firing per (r, j, k_step) — no pipeline depth needed
  static constexpr size_t expectedII =
      test_config::IN_HEIGHT_A * test_config::IN_WIDTH_B * (test_config::IN_WIDTH_A / test_config::W_PAR) *(test_config::IN_CH_A / test_config::CH_PAR);

  hls::stream<TInputWordA> in_data_A[test_config::CH_PAR];
  hls::stream<TInputWordB> in_data_B[test_config::CH_PAR];
  hls::stream<TOutputWord> mat_out;

  StreamingMatMul<TInputWordA, test_config::TInputA,
                  TInputWordB, test_config::TInputB,
                  TOutputWord, test_config::TOutput,
                  test_config::TAcc, test_config::Quantizer,
                  test_config::IN_HEIGHT_A, test_config::IN_WIDTH_A,
                  test_config::IN_WIDTH_B,
                  test_config::IN_CH_A, test_config::W_PAR,
                  test_config::CH_PAR>
                  matmul;
  matmul.step_init();   

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;

  while (true) {
    // Feed dummy data to all IN_CH streams
    for (int i_ch = 0; i_ch < (int)test_config::CH_PAR; i_ch++) {
      in_data_A[i_ch].write(TInputWordA());
      in_data_B[i_ch].write(TInputWordB());
    }

    ActorStatus actor_status = matmul.step(in_data_A, in_data_B, mat_out);

    std::vector<ActorStatus> actor_statuses = {actor_status};
    std::vector<size_t>      channel_quantities = {0};
    current_state = CSDFGState(actor_statuses, channel_quantities);

    if (visited_states.find(current_state) != visited_states.end()) {
      II = clock_cycles - visited_states[current_state];
      break;
    }
    visited_states.emplace(current_state, clock_cycles);
    clock_cycles++;
    assert(clock_cycles < 10 * expectedII);
  }

  // Flush output stream
  TOutputWord out_val;
  while (mat_out.read_nb(out_val));

  bool flag = (II == expectedII);
  std::cout << "Expected II: " << expectedII
            << ", Measured II: " << II << std::endl;
  return flag;
}

int main(int argc, char **argv) {
  bool all_passed = true;

  all_passed &= test_run();

  //test_step only in csim — used for FIFO depth estimation
  if (argc > 1 && std::string(argv[1]) == "csim") {
    all_passed &= test_step();
  }

  if (!all_passed){
    std::cout << "Failed." << std::endl;
  } else {
    std::cout << "Passed." << std::endl;
  }

  return all_passed ? 0 : 1;
}
