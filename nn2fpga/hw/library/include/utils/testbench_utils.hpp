#pragma once

#include "ap_int.h"
#include "ap_float.h"
#include "cnpy.h"
#include "hls_stream.h"
#include "utils/HLS_utils.hpp"
#include <cmath>
#include <vector>

template <typename TAxi, typename TData, typename TDataNumpy>
void hls_stream_to_npy(const std::string &output_path,
                       hls::stream<TAxi> &stream, int data_per_word,
                       const std::vector<size_t> &shape) {
  std::vector<TDataNumpy> output_data;
  size_t bits_per_data = data_width_v<TData>;
  while (!stream.empty()) {
    TAxi word = stream.read();
    ap_uint<decltype(word.keep)::width> valid_bytes = word.keep;
    for (int j = 0; j < data_per_word; j++) {
      TData hls_data = set_raw_bits<TData>(
          word.data.range((j + 1) * bits_per_data - 1, j * bits_per_data));
      TDataNumpy quant_data = static_cast<TDataNumpy>(hls_data);
      if ((valid_bytes.test(j))) { // Check if the j-th byte is valid
        output_data.push_back(quant_data);
      }
    }
  }

  // Save to .npy file
  cnpy::npy_save(output_path, &output_data[0], shape, "w");
}

template <typename TAxi, typename TData, typename TDataNumpy>
void npy_to_hls_stream(const std::string &input_path, hls::stream<TAxi> &stream,
                       int data_per_word) {
  cnpy::NpyArray input_array = cnpy::npy_load(input_path);
  TDataNumpy *input_data = input_array.data<TDataNumpy>();
  std::vector<size_t> shape = input_array.shape;
  size_t bits_per_data = data_width_v<TData>;

  // Compute the product of the shape dimensions
  size_t num_elements = 1;
  for (size_t dim : shape) {
    num_elements *= dim;
  }
  for (int i = 0; i < num_elements; i += data_per_word) {
    TAxi word;
    word.data = 0; // Initialize the data field to zero
    for (int j = 0; j < data_per_word; j++) {
      TData hls_data = static_cast<TData>(input_data[i + j]);
      word.data.range((j + 1) * bits_per_data - 1, j * bits_per_data) =
          get_raw_bits(hls_data);
    }
    word.last = (i + data_per_word >= num_elements)
                    ? 1
                    : 0; // Set last bit if this is the last word
    word.keep = (1 << data_per_word) - 1; // Set keep bits
    word.strb = (1 << data_per_word) - 1; // Set strb bits
    stream.write(word);
  }
}