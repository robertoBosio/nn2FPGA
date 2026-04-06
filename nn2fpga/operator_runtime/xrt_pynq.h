#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>

inline void program_with_pynq_cli_or_throw(const std::string& script, const std::string& bit_or_xclbin) {
  std::string cmd = "python3 " + script + " " + bit_or_xclbin;
  int rc = std::system(cmd.c_str());
  if (rc != 0) throw std::runtime_error("PYNQ loader failed, rc=" + std::to_string(rc));
}