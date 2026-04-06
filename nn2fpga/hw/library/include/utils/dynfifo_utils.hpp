#pragma once
#include "hls_fence.h"
#include <cstddef>
#include <cstdint>
#include "etc/autopilot_ssdm_op.h"

template <typename TWord, typename TData, size_t DIM, size_t CHUNK_SIZE,
          size_t PAR, size_t DEPTH>
class DDRstream {

  static_assert(DIM % PAR == 0, "DIM must be a multiple of PAR");
  static_assert(DIM % CHUNK_SIZE == 0, "DIM must be a multiple of CHUNK_SIZE");
  static_assert(CHUNK_SIZE % PAR == 0, "CHUNK_SIZE must be a multiple of PAR");
  using packed_data_t = std::array<TData, CHUNK_SIZE>;

  void pack_data(hls::stream<TWord> input_stream[1],
                 hls::stream<packed_data_t> &packed_data) {
    packed_data_t temp_data;
    for (size_t i = 0; i < DIM / CHUNK_SIZE; ++i) {
      for (size_t j = 0; j < CHUNK_SIZE / PAR; ++j) {
#pragma HLS loop_flatten
#pragma HLS pipeline II = 1
        TWord input_data = input_stream[0].read();
        for (size_t k = 0; k < PAR; ++k) {
#pragma HLS unroll
          temp_data[j * PAR + k] = input_data[k];
        }
        if (j == (CHUNK_SIZE / PAR) - 1) {
          packed_data.write(temp_data);
        }
      }
    }
  }

  void write_to_ddr(hls::stream<packed_data_t> &input_stream,
                    TData ddr_buffer[DIM], hls::stream<bool> &valid_stream) {

    for (size_t i = 0; i < DIM / CHUNK_SIZE; ++i) {
      packed_data_t input_data = input_stream.read();
      for (size_t k = 0; k < CHUNK_SIZE; ++k) {
#pragma HLS loop_flatten
#pragma HLS pipeline II = 1
        ddr_buffer[i * CHUNK_SIZE + k] = input_data[k];
      }
      hls::fence(ddr_buffer, valid_stream);
      // ap_wait();
      valid_stream.write(true);
    }
  }

  void read_from_ddr(hls::stream<bool> &valid_stream, TData ddr_buffer[DIM],
                     hls::stream<packed_data_t> &output_stream) {
    packed_data_t output_data;
    for (size_t i = 0; i < DIM / CHUNK_SIZE; ++i) {
      bool valid = valid_stream.read();
      (void)valid; // Suppress unused variable warning
      hls::fence<2>(valid_stream, ddr_buffer);
      // ap_wait();
      for (size_t k = 0; k < CHUNK_SIZE; ++k) {
#pragma HLS loop_flatten
#pragma HLS pipeline II = 1
        output_data[k] = ddr_buffer[i * CHUNK_SIZE + k];
      }
      output_stream.write(output_data);
    }
  }

  void unpack_data(hls::stream<packed_data_t> &input_stream,
                   hls::stream<TWord> output_stream[1]) {
    packed_data_t input_data;
    for (size_t i = 0; i < DIM / CHUNK_SIZE; ++i) {
      TWord output_data;
      for (size_t j = 0; j < CHUNK_SIZE / PAR; ++j) {
#pragma HLS loop_flatten
#pragma HLS pipeline II = 1
        if (j == 0) {
          input_data = input_stream.read();
        }
        for (size_t k = 0; k < PAR; ++k) {
#pragma HLS unroll
          output_data[k] = input_data[j * PAR + k];
        }
        output_stream[0].write(output_data);
      }
    }
  }

public:
  template <size_t HLS_ARG>
  void run(hls::stream<TWord> input_stream[1], TData ddr_buffer_read[DIM],
           TData ddr_buffer_write[DIM], hls::stream<TWord> output_stream[1]) {
#pragma HLS dataflow
#pragma HLS inline
    hls::stream<bool, DEPTH> valid_stream;
    hls::stream<packed_data_t, 2> packed_input_stream;
    hls::stream<packed_data_t, 2> packed_output_stream;
    pack_data(input_stream, packed_input_stream);
    write_to_ddr(packed_input_stream, ddr_buffer_write, valid_stream);
    read_from_ddr(valid_stream, ddr_buffer_read, packed_output_stream);
    unpack_data(packed_output_stream, output_stream);
  }
};