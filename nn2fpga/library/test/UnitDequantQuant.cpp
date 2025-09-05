#include "DequantQuant.hpp"
#include "ap_int.h"
#include "test_config.hpp"
#include <cassert>
#include <iostream>

void wrap_run() {
  DequantQuantPo2<test_config::SHIFT, test_config::TAcc,
                  test_config::TOut>
      quantizer;
  test_config::TAcc input = test_config::INPUT;
  test_config::TOut output = quantizer(input);
}

bool test_run() {
  DequantQuantPo2<test_config::SHIFT, test_config::TAcc,
                  test_config::TOut>
      quantizer;

  test_config::TAcc input = test_config::INPUT;
  test_config::TOut expected = test_config::EXPECTED;
  test_config::TOut output = quantizer(input);

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

  return all_passed ? 0 : 1;
}
