#pragma once

#include "catapult/Axis.hpp"
#include "cnpy.h"

#include <ac_channel.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <unsigned DATA_WIDTH, typename TData, typename TNumpy>
void npy_to_catapult_stream(const std::string &input_path,
                            ac_channel<AxisWord<DATA_WIDTH>> &stream,
                            size_t element_count) {
  static_assert(DATA_WIDTH % TData::width == 0,
                "AXIS width must be divisible by the element width");
  constexpr size_t DATA_PER_WORD = DATA_WIDTH / TData::width;

  cnpy::NpyArray input_array = cnpy::npy_load(input_path);
  if (input_array.num_vals != element_count) {
    throw std::runtime_error("Unexpected input tensor element count");
  }
  const TNumpy *input_data = input_array.data<TNumpy>();

  for (size_t offset = 0; offset < element_count; offset += DATA_PER_WORD) {
    AxisWord<DATA_WIDTH> word{};
    word.data = 0;
    word.keep = -1;
    word.strb = -1;
    word.last = offset + DATA_PER_WORD >= element_count;
    for (size_t lane = 0; lane < DATA_PER_WORD && offset + lane < element_count;
         ++lane) {
      word.data.set_slc(static_cast<unsigned>(lane * TData::width),
                        static_cast<TData>(input_data[offset + lane]));
    }
    stream.write(word);
  }
}

template <unsigned DATA_WIDTH, typename TData, typename TNumpy>
void catapult_stream_to_npy(const std::string &output_path,
                            ac_channel<AxisWord<DATA_WIDTH>> &stream,
                            size_t element_count,
                            const std::vector<size_t> &shape) {
  static_assert(DATA_WIDTH % TData::width == 0,
                "AXIS width must be divisible by the element width");
  constexpr size_t DATA_PER_WORD = DATA_WIDTH / TData::width;
  std::vector<TNumpy> output_data;
  output_data.reserve(element_count);

  for (size_t offset = 0; offset < element_count; offset += DATA_PER_WORD) {
    const AxisWord<DATA_WIDTH> word = stream.read();
    for (size_t lane = 0; lane < DATA_PER_WORD && output_data.size() < element_count;
         ++lane) {
      output_data.push_back(static_cast<TNumpy>(
          word.data.template slc<TData::width>(
              static_cast<unsigned>(lane * TData::width))));
    }
  }
  cnpy::npy_save(output_path, output_data.data(), shape, "w");
}

template <unsigned DATA_WIDTH, typename TData, typename TNumpy>
bool catapult_stream_to_npy_and_compare(
    const std::string &actual_path, const std::string &expected_path,
    ac_channel<AxisWord<DATA_WIDTH>> &stream, size_t element_count,
    const std::vector<size_t> &shape) {
  static_assert(DATA_WIDTH % TData::width == 0,
                "AXIS width must be divisible by the element width");
  constexpr size_t DATA_PER_WORD = DATA_WIDTH / TData::width;
  cnpy::NpyArray expected_array = cnpy::npy_load(expected_path);
  if (expected_array.num_vals != element_count) {
    throw std::runtime_error("Unexpected expected tensor element count");
  }
  const TNumpy *expected_data = expected_array.data<TNumpy>();
  std::vector<TNumpy> actual_data;
  actual_data.reserve(element_count);
  bool matches = true;

  for (size_t offset = 0; offset < element_count; offset += DATA_PER_WORD) {
    const AxisWord<DATA_WIDTH> word = stream.read();
    const size_t valid_elements =
        std::min(DATA_PER_WORD, element_count - offset);
    const size_t valid_bytes =
        (valid_elements * TData::width + 7) / 8;
    ac_int<DATA_WIDTH / 8, false> valid_mask = 0;
    for (size_t byte = 0; byte < valid_bytes; ++byte) {
      valid_mask.set_slc(static_cast<unsigned>(byte), ac_int<1, false>(1));
    }
    const bool is_last_word = offset + DATA_PER_WORD >= element_count;
    if (word.last != is_last_word || word.keep != valid_mask ||
        word.strb != valid_mask) {
      std::cerr << "Invalid AXIS sidebands at output word "
                << offset / DATA_PER_WORD << std::endl;
      matches = false;
    }
    for (size_t lane = 0; lane < valid_elements; ++lane) {
      const TNumpy value = static_cast<TNumpy>(
          word.data.template slc<TData::width>(
              static_cast<unsigned>(lane * TData::width)));
      const size_t index = offset + lane;
      if (value != expected_data[index]) {
        std::cerr << "Mismatch at output " << index << ": expected "
                  << static_cast<long long>(expected_data[index]) << ", got "
                  << static_cast<long long>(value) << std::endl;
        matches = false;
      }
      actual_data.push_back(value);
    }
  }
  cnpy::npy_save(actual_path, actual_data.data(), shape, "w");
  return matches;
}
