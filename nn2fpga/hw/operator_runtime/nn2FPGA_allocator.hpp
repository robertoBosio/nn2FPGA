#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <onnxruntime_c_api.h>

// Runtime support for a CPU-visible, FPGA-visible ONNX Runtime allocator.
//
// The allocator advertises normal CPU memory to ONNX Runtime, but every
// allocation is backed by an XRT BO. ORT CPU kernels receive the BO mapped CPU
// pointer, while nn2FPGA custom ops can recover the BO device address through
// the C ABI below and program DMA without staging copies.

// Registers the global nn2FPGA allocator on an ORT environment.
//
// ORT requires user-provided allocators to be registered as OrtDeviceAllocator,
// even if they internally implement arena-like behavior. The memory info still
// represents CPU memory, so CPUExecutionProvider kernels can use it.
OrtStatus *nn2fpga_register_xrt_cpu_allocator(OrtEnv *env, const OrtApi *api);

// Returns the process-global allocator registered by
// nn2fpga_register_xrt_cpu_allocator(). This is mainly used by standalone
// runners that want model input tensors to be XRT-backed as well.
OrtAllocator *nn2fpga_xrt_cpu_allocator();

struct Nn2FpgaAllocatorStats {
  size_t alloc_count;
  size_t free_count;
  size_t live_count;
  size_t pooled_count;
  size_t allocated_bytes;
  size_t xrt_alloc_count;
  size_t pool_reuse_count;
  uint64_t xrt_alloc_time_us;
  uint64_t map_time_us;
  uint64_t sync_to_device_count;
  uint64_t sync_from_device_count;
  uint64_t sync_to_device_time_us;
  uint64_t sync_from_device_time_us;
};

Nn2FpgaAllocatorStats nn2fpga_allocator_stats();
void nn2fpga_allocator_dump_stats(FILE *stream);

extern "C" {

// Returns true if ptr belongs to an allocation owned by the nn2FPGA allocator.
// size_out receives the total allocation size, not the remaining bytes from ptr.
bool nn2fpga_allocator_contains(const void *ptr, size_t *size_out);

// Looks up allocator metadata for ptr. device_addr_out receives the
// physical/device address corresponding to ptr, including any offset from the
// allocation base.
bool nn2fpga_allocator_lookup(const void *ptr, size_t *size_out,
                              uint64_t *device_addr_out);

// Cache-coherency helpers. They return false if ptr is not an allocator-owned
// XRT BO or if the requested byte range is outside the allocation.
bool nn2fpga_allocator_sync_to_device(const void *ptr, size_t bytes,
                                      size_t offset);
bool nn2fpga_allocator_sync_from_device(void *ptr, size_t bytes,
                                        size_t offset);

}
