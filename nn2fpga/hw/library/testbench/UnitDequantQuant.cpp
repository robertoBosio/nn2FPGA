#ifdef NN2FPGA_CATAPULT
#include "catapult/DequantQuant.hpp"
#include <ac_int.h>
#else
#include "vitis/DequantQuant.hpp"
#include "ap_int.h"
#endif
#include "test_config.hpp"
#ifdef NN2FPGA_CATAPULT
#include <mc_scverify.h>
#endif
#include <cassert>
#include <iostream>

#ifdef NN2FPGA_CATAPULT
#pragma hls_design top
#endif
void wrap_run(test_config::TAcc input, test_config::TOut &output) {
  test_config::Quantizer quantizer;
  output = quantizer(input);
}

bool test_run() {
  test_config::TAcc input = test_config::input_tensor[0];
  test_config::TOut expected = test_config::output_tensor[0];
  test_config::TOut output;
#ifdef NN2FPGA_CATAPULT
  CCS_DESIGN(wrap_run)(input, output);
#else
  wrap_run(input, output);
#endif

  if (output != expected) {
    std::cout << "Test failed: input=" << input
              << ", expected=" << expected << ", got=" << output
              << std::endl;
    return false;
  }
  return true;
}

#ifdef NN2FPGA_CATAPULT
CCS_MAIN(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif

  bool all_passed = true;
  all_passed &= test_run();

  if (!all_passed) {
    std::cout << "Failed." << std::endl;
  } else {
    std::cout << "Passed." << std::endl;
  }

  (void)argc;
  (void)argv;
#ifdef NN2FPGA_CATAPULT
  CCS_RETURN(all_passed ? 0 : 1);
#else
  return all_passed ? 0 : 1;
#endif
}
