#include "utils/dynfifo_utils.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "ap_float.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

void wrap_run(hls::stream<test_config::TWord> in_stream[test_config::DIM1_UNROLL],
              test_config::TAXIWord ddr_buffer_read[test_config::DIM],
              test_config::TAXIWord ddr_buffer_write[test_config::DIM],
              hls::stream<test_config::TWord> out_stream[test_config::DIM1_UNROLL]) {
#pragma HLS INTERFACE axis port = in_stream
#pragma HLS INTERFACE axis port = out_stream
#pragma HLS INTERFACE m_axi port = ddr_buffer_read bundle = gmem0 depth = test_config::DIM
#pragma HLS INTERFACE m_axi port = ddr_buffer_write bundle = gmem1 depth = test_config::DIM
#pragma HLS STABLE variable = ddr_buffer_read
#pragma HLS STABLE variable = ddr_buffer_write
#pragma HLS ALIAS ports = ddr_buffer_read, ddr_buffer_write distance = 0
#pragma HLS INTERFACE mode = ap_ctrl_chain port = return
#pragma HLS DATAFLOW

    DDRstream<test_config::TWord, test_config::TData, test_config::TAXIWord,
              test_config::DIM, test_config::BURST_SIZE, test_config::AXIWORD_PAR,
              test_config::PAR, test_config::DEPTH> ddr_stream;

    ddr_stream.run<0>(in_stream, ddr_buffer_read, ddr_buffer_write, out_stream);
}

bool test_run() {

    // Prepare input and output streams
    hls::stream<test_config::TWord> in_stream[test_config::DIM1_UNROLL];
    hls::stream<test_config::TWord> out_stream[test_config::DIM1_UNROLL];

    // Prepare DDR buffers
    test_config::TAXIWord ddr_buffer[test_config::DIM];

    // Fill the input stream with test data
    for (size_t i_dim0 = 0; i_dim0 < test_config::DIM0; ++i_dim0) {
      for (size_t i_dim1 = 0; i_dim1 < test_config::DIM1;
           i_dim1 += test_config::DIM1_UNROLL) {
        for (size_t i_dim2 = 0; i_dim2 < test_config::DIM2;
             i_dim2 += test_config::DIM2_UNROLL) {
          for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
               ++i_dim1_par) {
            test_config::TWord word;
            for (size_t i_dim2_par = 0; i_dim2_par < test_config::DIM2_UNROLL;
                 ++i_dim2_par) {
              word[i_dim2_par] =
                  test_config::input_tensor[0][i_dim2 + i_dim2_par][i_dim0]
                                           [i_dim1 + i_dim1_par];
            }
            in_stream[i_dim1_par].write(word);
          }
        }
      }
    }

    // Run the DDRstream module
    wrap_run(in_stream, ddr_buffer, ddr_buffer, out_stream);

    // Verify the output data
    bool all_passed = true;
    for (size_t i_dim0 = 0; i_dim0 < test_config::DIM0; ++i_dim0) {
      for (size_t i_dim1 = 0; i_dim1 < test_config::DIM1;
           i_dim1 += test_config::DIM1_UNROLL) {
        for (size_t i_dim2 = 0; i_dim2 < test_config::DIM2;
             i_dim2 += test_config::DIM2_UNROLL) {
          for (size_t i_dim1_par = 0; i_dim1_par < test_config::DIM1_UNROLL;
               ++i_dim1_par) {
            test_config::TWord expected_word;
            test_config::TWord actual_word = out_stream[i_dim1_par].read();
            for (size_t i_dim2_par = 0; i_dim2_par < test_config::DIM2_UNROLL;
                 ++i_dim2_par) {
              expected_word[i_dim2_par] =
                  test_config::input_tensor[0][i_dim2 + i_dim2_par][i_dim0]
                                           [i_dim1 + i_dim1_par];
              if (actual_word[i_dim2_par] != expected_word[i_dim2_par]) {
                all_passed = false;
                std::cout << "Mismatch at dim0=" << i_dim0
                          << ", dim1=" << (i_dim1 + i_dim1_par)
                          << ", dim2=" << (i_dim2 + i_dim2_par) << ": expected "
                          << (float)expected_word[i_dim2_par] << ", got "
                          << (float)actual_word[i_dim2_par] << std::endl;
              }
            }
          }
        }
      }
    }

    return all_passed;
}

int main() {
    bool passed = test_run();
    if (passed) {
        std::cout << "Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Failed" << std::endl;
        return 1;
    }
}