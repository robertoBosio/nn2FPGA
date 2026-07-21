#include "DequantQuant.hpp"
#include "ap_int.h"
#include "test_config.hpp"
#include <cassert>
#include <iostream>

void wrap_run(test_config::TAcc input, test_config::TOut &output) {
  test_config::Quantizer quantizer;
  output = quantizer(input);
}

bool test_run() {
  test_config::TAcc input = test_config::input_tensor[0];
  test_config::TOut expected = test_config::output_tensor[0];
  test_config::TOut output;
  wrap_run(input, output);

  if (output != expected) {
    std::cout << "Test failed: input=" << input
              << ", expected=" << expected << ", got=" << output
              << std::endl;
    return false;
  }
  return true;
}

int main(int argc, char **argv) {

  bool all_passed = true;
  all_passed &= test_run();

  if (!all_passed) {
    std::cout << "Failed." << std::endl;
  } else {
    std::cout << "Passed." << std::endl;
  }

  (void)argc;
  (void)argv;
  return all_passed ? 0 : 1;
}
