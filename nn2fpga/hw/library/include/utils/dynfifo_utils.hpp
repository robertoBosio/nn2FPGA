#pragma once
#include "hls_fence.h"
#include <cstddef>
#include <cstdint>
#include "etc/autopilot_ssdm_op.h"

template <typename TWord, typename TData, size_t DIM, size_t BURST_SIZE, size_t WORD_BYTES,
          size_t PAR, size_t DEPTH>
class DDRstream {

  static_assert(DIM % PAR == 0, "DIM must be a multiple of PAR");
  static_assert(WORD_BYTES % PAR == 0, "CHUNK_SIZE must be a multiple of PAR");

  void pack_data(hls::stream<TWord> input_stream[1],
                 hls::stream<TData> &packed_data) {
    TData temp_data;
    for (size_t i = 0; i < DIM; ++i) {
      for (size_t j = 0; j < WORD_BYTES / PAR; ++j) {
#pragma HLS loop_flatten
#pragma HLS pipeline II = 1
        TWord input_data = input_stream[0].read();
        for (size_t k = 0; k < PAR; ++k) {
#pragma HLS unroll
          temp_data.range((j * PAR + k + 1) * 8 - 1, (j * PAR + k) * 8) =
              input_data[k];
        }
        if (j == (WORD_BYTES / PAR) - 1) {
          packed_data.write(temp_data);
        }
      }
    }
  }

  void write_to_ddr(hls::stream<TData> &input_stream,
                    TData ddr_buffer[DIM], hls::stream<bool> &valid_stream,
                    hls::stream<bool> &start_stream) {

    for (size_t i = 0; i < DIM / BURST_SIZE; ++i) {
      for (size_t j = 0; j < BURST_SIZE; ++j) {
        TData input_data = input_stream.read();
        ddr_buffer[i * BURST_SIZE + j] = input_data;
      }
      hls::fence(ddr_buffer, valid_stream);
      valid_stream.write(true);
      if (i == 0) {
        start_stream.write(true); // Signal the end of the first write operation
      }
    }
  }

  void read_from_ddr(hls::stream<bool> &start_stream,
                     hls::stream<bool> &valid_stream, TData ddr_buffer[DIM],
                     hls::stream<TData> &output_stream) {
    TData output_data;
    start_stream.read(); // Wait for the signal to start reading
    for (size_t i = 0; i < DIM / BURST_SIZE; ++i) {
      bool valid = valid_stream.read();
      hls::fence(valid_stream, ddr_buffer);
      (void)valid; 
      for (size_t j = 0; j < BURST_SIZE; ++j) {
        output_data = ddr_buffer[i * BURST_SIZE + j];
        output_stream.write(output_data);
      }
    }
  }

  void unpack_data(hls::stream<TData> &input_stream,
                   hls::stream<TWord> output_stream[1]) {
    TData input_data;
    for (size_t i = 0; i < DIM; ++i) {
      TWord output_data;
      for (size_t j = 0; j < WORD_BYTES / PAR; ++j) {
#pragma HLS loop_flatten
#pragma HLS pipeline II = 1
        if (j == 0) {
          input_data = input_stream.read();
        }
        for (size_t k = 0; k < PAR; ++k) {
#pragma HLS unroll
          output_data[k] = input_data.range((j * PAR + k + 1) * 8 - 1, (j * PAR + k) * 8);
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
    hls::stream<bool, (DEPTH / (BURST_SIZE * WORD_BYTES / PAR)) + 1> valid_stream;
    hls::stream<bool, 2> start_stream;
    hls::stream<TData, BURST_SIZE + 1> packed_input_stream;
    hls::stream<TData, BURST_SIZE + 1> packed_output_stream;
    pack_data(input_stream, packed_input_stream);
    write_to_ddr(packed_input_stream, ddr_buffer_write, valid_stream, start_stream);
    read_from_ddr(start_stream, valid_stream, ddr_buffer_read, packed_output_stream);
    unpack_data(packed_output_stream, output_stream);
  }
};