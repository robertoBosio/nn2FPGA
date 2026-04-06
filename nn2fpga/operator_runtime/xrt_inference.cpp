#include <xrt/xclbin.h>
#include <xrt/xrt_device.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "xrt_dma.h"
#include "xrt_ps.h"
#include "xrt_pynq.h"
#include "xrt_mmio.hpp"


int main() {

  xrt::device dev{0};

  // Program the PL through PYNQ.
  program_with_pynq_cli_or_throw("pynq_program.py", "Overlay/mobilenet_v2.bit");

  // Map the single AXI-Lite window once.
  auto mmio = map_axil_window(0xA0000000, 0x10000);

  // Set the frequency of PL0 using IOPLL
  try {
    set_pl_from_iopll(0, 250.0, 1000.0); // Set PL0 to 250 MHz using IOPLL at 1000 MHz
  } catch (const std::exception &e) {
    std::cerr << "Error setting PL clock: " << e.what() << "\n";
    return 1;
  }

  // Preallocate I/O once
  constexpr size_t BATCH_MAX = 10;
  constexpr size_t BYTES_IN_TOTAL = BATCH_MAX * 7 * 7 * 1280;
  constexpr size_t BYTES_PER_IMAGE = 1 * 1 * 1280;

  xrt::bo in_bo{dev, BYTES_IN_TOTAL, 0, 0};
  xrt::bo out_bo{dev, BYTES_PER_IMAGE * BATCH_MAX, 0, 0};

  // Offsets in your single AXI-Lite window (adjust to your design)
  constexpr off_t MM2S_OFF = 0x0000; // input DMA (simple)
  constexpr off_t S2MM_OFF = 0x1000; // output DMA (SG)

  // Build persistent engines
  Mm2sSimple tx{mmio.regs, MM2S_OFF, in_bo, BYTES_IN_TOTAL};
  S2mmRingSg rx{mmio.regs, S2MM_OFF, dev, out_bo, BYTES_PER_IMAGE, BATCH_MAX};

  uint8_t input_data[BYTES_IN_TOTAL];
  int8_t output_data[BYTES_PER_IMAGE * BATCH_MAX];
  auto* in_ptr  = static_cast<uint8_t*>(in_bo.map());
  auto* out_ptr = static_cast<int8_t*>(out_bo.map());
  std::fill(input_data, input_data + BYTES_IN_TOTAL, 1); // Fill with ones
  std::fill(output_data, output_data + BYTES_PER_IMAGE * BATCH_MAX, 0); // Zero output

  // Load input data into the input buffer
  auto start_time = std::chrono::high_resolution_clock::now();
  // in_bo.write(input_data, BYTES_IN_TOTAL, 0);
  std::memcpy(in_ptr, input_data, BYTES_IN_TOTAL);

  // If CPU updated input buffer:
  tx.flush_input();
  rx.start_once(BATCH_MAX);
  tx.start_once(0);

  if (!rx.wait_done(20)) {
    throw std::runtime_error("S2MM timeout");
  }
  rx.invalidate_output_after();
  // out_bo.read(output_data, BYTES_PER_IMAGE * BATCH_MAX, 0);
  std::memcpy(output_data, out_ptr, BYTES_PER_IMAGE * BATCH_MAX);
  auto end_time = std::chrono::high_resolution_clock::now();

  // Calculate elapsed time
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  std::cout << "Inference completed in " << duration.count() << " us\n"
            << "Total images processed: " << BATCH_MAX << "\n"
            << "Average time per image: " << duration.count() / BATCH_MAX
            << " us\n";

  // Print output data for verification
  std::cout << "Output data:\n";
  for (size_t j = 0; j < 10; ++j) {
    std::cout << static_cast<int>(output_data[j]) << " ";
  }
  std::cout << "\n";

  return 0;
}
