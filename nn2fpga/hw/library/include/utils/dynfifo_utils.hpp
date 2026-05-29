#pragma once
#include "ap_float.h"
#include "ap_int.h"
#include "etc/autopilot_ssdm_op.h"
#include "hls_fence.h"
#include "hls_stream.h"
#include "utils/HLS_utils.hpp"
#include <cstddef>
#include <cstdint>

/**
 * DDRstream: A utility class to have an hls stream backed by DDR memory.
 *
 * @tparam TWord: The data type of the stream (e.g., std::array<ap_uint<8>, 16>)
 * @tparam TData: The data type of the individual elements in the stream (e.g., ap_uint<8>)
 * @tparam TAXIWord: The data type of the packed data in DDR (e.g.,
 * ap_uint<128>)
 * @tparam DIM: The total number of packed words to be transferred (the size of
 * the tensor)
 * @tparam BURST_SIZE: The number of packed words to transfer in one burst (must
 * divide DIM)
 * @tparam AXIWORD_PAR: The number of TWord in one TAXIWord (e.g., 16 for a
 * ap_uint<128> AXI word containing 16 std::array<ap_uint<8>, 1> elements)
 * @tparam DIM2_UNROLL: The number of parallel elements in one TWord (e.g., 16 for
 * std::array<ap_uint<8>, 16>)
 * @tparam DEPTH: The depth of the FIFO to be substituted by DDR.
 *
 * The class provides a dataflow architecture with 4 main stages:
 * 1. pack_data: Reads from the input stream, packs the data into TAXIWord
 * format, and writes to an intermediate stream.
 * 2. write_to_ddr: Reads packed data from the intermediate stream and writes
 *    it to DDR in bursts.
 * 3. read_from_ddr: Reads packed data from DDR in bursts and writes it to
 *    another intermediate stream.
 * 4. unpack_data: Reads packed data from the intermediate stream, unpacks it,
 *    and writes it to the output stream.
 */
template <typename TWord, typename TData, typename TAXIWord, size_t DIM,
          size_t BURST_SIZE, size_t AXIWORD_PAR, size_t DIM2_UNROLL, size_t DEPTH>
class DDRstream {

  static_assert(DIM2_UNROLL != 0 && AXIWORD_PAR != 0 &&
                    AXIWORD_PAR % DIM2_UNROLL == 0,
                "AXIWORD_PAR must be a non-zero multiple of DIM2_UNROLL");
  static_assert(BURST_SIZE != 0 && DIM % BURST_SIZE == 0,
                "DIM must be a multiple of BURST_SIZE");

  constexpr static size_t width = data_width_v<TData>;

  void pack_data(hls::stream<TWord> input_stream[1],
                 hls::stream<TAXIWord> &packed_data) {
    TAXIWord temp_data;
    for (size_t i = 0; i < DIM; ++i) {
      for (size_t j = 0; j < AXIWORD_PAR / DIM2_UNROLL; ++j) {
#pragma HLS loop_flatten
#pragma HLS pipeline II = 1
        TWord input_data = input_stream[0].read();
        for (size_t k = 0; k < DIM2_UNROLL; ++k) {
#pragma HLS unroll
          temp_data.range((j * DIM2_UNROLL + k + 1) * width - 1,
                          (j * DIM2_UNROLL + k) * width) = get_raw_bits(input_data[k]);
        }
        if (j == (AXIWORD_PAR / DIM2_UNROLL) - 1) {
          packed_data.write(temp_data);
        }
      }
    }
  }

  void write_to_ddr(hls::stream<TAXIWord> &input_stream,
                    TAXIWord ddr_buffer[DIM], hls::stream<bool> &valid_stream) {

    for (size_t i = 0; i < DIM / BURST_SIZE; ++i) {
    WRITE_TO_DDR_LOOP:
      for (size_t j = 0; j < BURST_SIZE; ++j) {
#pragma HLS pipeline II = 1
        TAXIWord input_data = input_stream.read();
        ddr_buffer[i * BURST_SIZE + j] = input_data;
      }
      hls::fence<6>({ddr_buffer}, {valid_stream});
      valid_stream.write(true);
    }
  }

  void read_from_ddr(hls::stream<bool> &valid_stream, TAXIWord ddr_buffer[DIM],
                     hls::stream<TAXIWord> &output_stream) {
    TAXIWord output_data;
    for (size_t i = 0; i < DIM / BURST_SIZE; ++i) {
#pragma HLS loop_flatten off
      bool valid = valid_stream.read();
      (void)valid;
      hls::fence({valid_stream}, {ddr_buffer});
    READ_FROM_DDR_LOOP:
      for (size_t j = 0; j < BURST_SIZE; ++j) {
#pragma HLS pipeline II = 1
        output_data = ddr_buffer[i * BURST_SIZE + j];
        output_stream.write(output_data);
      }
    }
  }

  void unpack_data(hls::stream<TAXIWord> &input_stream,
                   hls::stream<TWord> output_stream[1]) {
    TAXIWord input_data;
    for (size_t i = 0; i < DIM; ++i) {
      TWord output_data;
      for (size_t j = 0; j < AXIWORD_PAR / DIM2_UNROLL; ++j) {
#pragma HLS loop_flatten
#pragma HLS pipeline II = 1
        if (j == 0) {
          input_data = input_stream.read();
        }
        for (size_t k = 0; k < DIM2_UNROLL; ++k) {
#pragma HLS unroll
          output_data[k] = set_raw_bits<TData>(input_data.range(
              (j * DIM2_UNROLL + k + 1) * width - 1, (j * DIM2_UNROLL + k) * width));
        }
        output_stream[0].write(output_data);
      }
    }
  }

public:
  template <size_t HLS_ARG>
  void run(hls::stream<TWord> input_stream[1], TAXIWord ddr_buffer_read[DIM],
           TAXIWord ddr_buffer_write[DIM],
           hls::stream<TWord> output_stream[1]) {
#pragma HLS dataflow
#pragma HLS inline
    constexpr size_t valid_depth =
        (DEPTH / (BURST_SIZE * AXIWORD_PAR / DIM2_UNROLL)) + 1;
    hls::stream<bool, valid_depth> valid_stream;
    hls::stream<TAXIWord, BURST_SIZE + 1> packed_input_stream;
    hls::stream<TAXIWord, BURST_SIZE + 1> packed_output_stream;
    pack_data(input_stream, packed_input_stream);
    write_to_ddr(packed_input_stream, ddr_buffer_write, valid_stream);
    read_from_ddr(valid_stream, ddr_buffer_read, packed_output_stream);
    unpack_data(packed_output_stream, output_stream);
  }
};
