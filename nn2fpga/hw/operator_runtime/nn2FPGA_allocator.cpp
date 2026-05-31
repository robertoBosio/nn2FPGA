#include "nn2FPGA_allocator.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>

namespace {

struct AllocationRange {
  const void *base = nullptr;
  size_t size = 0;
  std::shared_ptr<xrt::bo> bo;
  uint64_t device_addr = 0;
};

struct SizeStats {
  size_t alloc_count = 0;
  size_t free_count = 0;
  size_t xrt_alloc_count = 0;
  size_t pool_reuse_count = 0;
  size_t total_bytes = 0;
};

struct RuntimeAllocator {
  OrtAllocator allocator{};
  OrtMemoryInfo *memory_info = nullptr;
  const OrtApi *api = nullptr;
  std::atomic<size_t> alloc_count{0};
  std::atomic<size_t> free_count{0};
  std::atomic<size_t> allocated_bytes{0};
  std::atomic<size_t> xrt_alloc_count{0};
  std::atomic<size_t> pool_reuse_count{0};
  std::atomic<uint64_t> xrt_alloc_time_us{0};
  std::atomic<uint64_t> map_time_us{0};
  std::atomic<uint64_t> sync_to_device_count{0};
  std::atomic<uint64_t> sync_from_device_count{0};
  std::atomic<uint64_t> sync_to_device_time_us{0};
  std::atomic<uint64_t> sync_from_device_time_us{0};
};

RuntimeAllocator g_allocator;
std::mutex g_ranges_mutex;
std::vector<AllocationRange> g_ranges;
std::vector<AllocationRange> g_pool;
std::map<size_t, SizeStats> g_size_stats;

uint64_t elapsed_us(std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count());
}

xrt::device &xrt_device() {
  static xrt::device dev{0};
  return dev;
}

bool allocator_log_enabled() {
  static const bool enabled = []() {
    const char *value = std::getenv("NN2FPGA_ALLOCATOR_LOG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

bool allocator_pool_enabled() {
  static const bool enabled = []() {
    const char *value = std::getenv("NN2FPGA_ALLOCATOR_POOL");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

bool allocator_cacheable_enabled() {
  static const bool enabled = []() {
    const char *value = std::getenv("NN2FPGA_ALLOCATOR_CACHEABLE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

void record_alloc_size_locked(size_t size, bool reused) {
  SizeStats &stats = g_size_stats[size];
  stats.alloc_count += 1;
  stats.total_bytes += size;
  if (reused) {
    stats.pool_reuse_count += 1;
  } else {
    stats.xrt_alloc_count += 1;
  }
}

void record_free_size_locked(size_t size) {
  g_size_stats[size].free_count += 1;
}

bool find_range_locked(const void *ptr, AllocationRange *out,
                       uintptr_t *offset_out) {
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  for (const auto &range : g_ranges) {
    const auto base = reinterpret_cast<uintptr_t>(range.base);
    if (addr >= base && addr < base + range.size) {
      if (out != nullptr) {
        *out = range;
      }
      if (offset_out != nullptr) {
        *offset_out = addr - base;
      }
      return true;
    }
  }
  return false;
}

void *ORT_API_CALL allocator_alloc(OrtAllocator *, size_t size) {
  const size_t actual_size = size == 0 ? 1 : size;
  void *ptr = nullptr;

  if (allocator_pool_enabled()) {
    std::lock_guard<std::mutex> lock(g_ranges_mutex);
    for (auto it = g_pool.begin(); it != g_pool.end(); ++it) {
      if (it->size == actual_size) {
        AllocationRange range = *it;
        g_pool.erase(it);
        ptr = const_cast<void *>(range.base);
        g_ranges.push_back(range);
        g_allocator.alloc_count.fetch_add(1, std::memory_order_relaxed);
        g_allocator.pool_reuse_count.fetch_add(1, std::memory_order_relaxed);
        g_allocator.allocated_bytes.fetch_add(actual_size,
                                             std::memory_order_relaxed);
        record_alloc_size_locked(actual_size, true);
        if (allocator_log_enabled()) {
          std::fprintf(stderr,
                       "[nn2fpga allocator] pool reuse size=%zu ptr=%p device_addr=0x%lx\n",
                       size, ptr,
                       static_cast<unsigned long>(range.device_addr));
        }
        return ptr;
      }
    }
  }

  std::shared_ptr<xrt::bo> bo;
  uint64_t device_addr = 0;
  try {
    const auto alloc_start = std::chrono::steady_clock::now();
    const auto flags = allocator_cacheable_enabled()
                           ? xrt::bo::flags::cacheable
                           : xrt::bo::flags::normal;
    bo = std::make_shared<xrt::bo>(xrt_device(), actual_size, flags, 0);
    const auto alloc_end = std::chrono::steady_clock::now();
    const auto map_start = alloc_end;
    ptr = bo->map<void *>();
    const auto map_end = std::chrono::steady_clock::now();
    device_addr = bo->address();
    g_allocator.xrt_alloc_time_us.fetch_add(elapsed_us(alloc_start, alloc_end),
                                           std::memory_order_relaxed);
    g_allocator.map_time_us.fetch_add(elapsed_us(map_start, map_end),
                                      std::memory_order_relaxed);
  } catch (const std::exception &e) {
    std::fprintf(stderr,
                 "[nn2fpga allocator] XRT allocation failed size=%zu: %s\n",
                 size, e.what());
    return nullptr;
  }

  g_allocator.alloc_count.fetch_add(1, std::memory_order_relaxed);
  g_allocator.xrt_alloc_count.fetch_add(1, std::memory_order_relaxed);
  g_allocator.allocated_bytes.fetch_add(actual_size, std::memory_order_relaxed);

  {
    std::lock_guard<std::mutex> lock(g_ranges_mutex);
    AllocationRange range;
    range.base = ptr;
    range.size = actual_size;
    range.bo = bo;
    range.device_addr = device_addr;
    g_ranges.push_back(range);
    record_alloc_size_locked(actual_size, false);
  }

  if (allocator_log_enabled()) {
    std::fprintf(stderr,
                 "[nn2fpga allocator] xrt alloc size=%zu ptr=%p device_addr=0x%lx\n",
                 size, ptr, static_cast<unsigned long>(device_addr));
  }

  return ptr;
}

void ORT_API_CALL allocator_free(OrtAllocator *, void *ptr) {
  g_allocator.free_count.fetch_add(1, std::memory_order_relaxed);

  AllocationRange freed;
  bool found = false;

  {
    std::lock_guard<std::mutex> lock(g_ranges_mutex);
    for (auto it = g_ranges.begin(); it != g_ranges.end(); ++it) {
      if (it->base == ptr) {
        freed = *it;
        found = true;
        g_ranges.erase(it);
        record_free_size_locked(freed.size);
        if (allocator_pool_enabled()) {
          g_pool.push_back(freed);
        }
        break;
      }
    }
  }

  if (allocator_log_enabled()) {
    std::fprintf(stderr, "[nn2fpga allocator] free ptr=%p%s\n", ptr,
                 found && allocator_pool_enabled() ? " pooled" : "");
  }
}

const OrtMemoryInfo *ORT_API_CALL allocator_info(const OrtAllocator *) {
  return g_allocator.memory_info;
}

} // namespace

OrtStatus *nn2fpga_register_xrt_cpu_allocator(OrtEnv *env, const OrtApi *api) {
  g_allocator.api = api;

  if (g_allocator.memory_info == nullptr) {
    OrtStatus *status = api->CreateCpuMemoryInfo(
        OrtDeviceAllocator, OrtMemTypeDefault, &g_allocator.memory_info);
    if (status != nullptr) {
      return status;
    }
  }

  g_allocator.allocator.version = ORT_API_VERSION;
  g_allocator.allocator.Alloc = allocator_alloc;
  g_allocator.allocator.Free = allocator_free;
  g_allocator.allocator.Info = allocator_info;

  std::fprintf(stderr, "Registering nn2FPGA CPU allocator as device allocator\n");
  std::fprintf(stderr, "Allocator backend: XRT BO mapped memory%s\n",
               allocator_cacheable_enabled() ? ", cacheable" : "");

  return api->RegisterAllocator(env, &g_allocator.allocator);
}

OrtAllocator *nn2fpga_xrt_cpu_allocator() { return &g_allocator.allocator; }

Nn2FpgaAllocatorStats nn2fpga_allocator_stats() {
  Nn2FpgaAllocatorStats stats{};
  stats.alloc_count = g_allocator.alloc_count.load(std::memory_order_relaxed);
  stats.free_count = g_allocator.free_count.load(std::memory_order_relaxed);
  stats.allocated_bytes =
      g_allocator.allocated_bytes.load(std::memory_order_relaxed);
  stats.xrt_alloc_count =
      g_allocator.xrt_alloc_count.load(std::memory_order_relaxed);
  stats.pool_reuse_count =
      g_allocator.pool_reuse_count.load(std::memory_order_relaxed);
  stats.xrt_alloc_time_us =
      g_allocator.xrt_alloc_time_us.load(std::memory_order_relaxed);
  stats.map_time_us = g_allocator.map_time_us.load(std::memory_order_relaxed);
  stats.sync_to_device_count =
      g_allocator.sync_to_device_count.load(std::memory_order_relaxed);
  stats.sync_from_device_count =
      g_allocator.sync_from_device_count.load(std::memory_order_relaxed);
  stats.sync_to_device_time_us =
      g_allocator.sync_to_device_time_us.load(std::memory_order_relaxed);
  stats.sync_from_device_time_us =
      g_allocator.sync_from_device_time_us.load(std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(g_ranges_mutex);
    stats.live_count = g_ranges.size();
    stats.pooled_count = g_pool.size();
  }
  return stats;
}

void nn2fpga_allocator_dump_stats(FILE *stream) {
  if (stream == nullptr) {
    stream = stderr;
  }
  const Nn2FpgaAllocatorStats stats = nn2fpga_allocator_stats();
  std::fprintf(stream,
               "Allocator stats: alloc_count=%zu free_count=%zu live_count=%zu pooled_count=%zu allocated_bytes=%zu xrt_alloc_count=%zu pool_reuse_count=%zu xrt_alloc_time=%.3f ms map_time=%.3f ms sync_to_device_count=%lu sync_to_device_time=%.3f ms sync_from_device_count=%lu sync_from_device_time=%.3f ms\n",
               stats.alloc_count, stats.free_count, stats.live_count,
               stats.pooled_count, stats.allocated_bytes, stats.xrt_alloc_count,
               stats.pool_reuse_count, stats.xrt_alloc_time_us / 1000.0,
               stats.map_time_us / 1000.0,
               static_cast<unsigned long>(stats.sync_to_device_count),
               stats.sync_to_device_time_us / 1000.0,
               static_cast<unsigned long>(stats.sync_from_device_count),
               stats.sync_from_device_time_us / 1000.0);

  std::lock_guard<std::mutex> lock(g_ranges_mutex);
  std::fprintf(stream, "Allocator size histogram:\n");
  std::fprintf(stream,
               "  %12s %10s %10s %10s %10s %14s\n",
               "size", "alloc", "free", "xrt", "reuse", "total_bytes");
  size_t rows = 0;
  for (const auto &entry : g_size_stats) {
    const SizeStats &s = entry.second;
    std::fprintf(stream,
                 "  %12zu %10zu %10zu %10zu %10zu %14zu\n",
                 entry.first, s.alloc_count, s.free_count, s.xrt_alloc_count,
                 s.pool_reuse_count, s.total_bytes);
    rows += 1;
  }
  if (rows == 0) {
    std::fprintf(stream, "  <empty>\n");
  }
}

extern "C" bool nn2fpga_allocator_contains(const void *ptr, size_t *size_out) {
  std::lock_guard<std::mutex> lock(g_ranges_mutex);
  AllocationRange range;
  if (!find_range_locked(ptr, &range, nullptr)) {
    return false;
  }
  if (size_out != nullptr) {
    *size_out = range.size;
  }
  return true;
}

extern "C" bool nn2fpga_allocator_lookup(const void *ptr, size_t *size_out,
                                          uint64_t *device_addr_out) {
  std::lock_guard<std::mutex> lock(g_ranges_mutex);
  AllocationRange range;
  uintptr_t offset = 0;
  if (!find_range_locked(ptr, &range, &offset)) {
    return false;
  }
  if (size_out != nullptr) {
    *size_out = range.size;
  }
  if (device_addr_out != nullptr) {
    *device_addr_out = range.device_addr + offset;
  }
  return true;
}

extern "C" bool nn2fpga_allocator_sync_to_device(const void *ptr, size_t bytes,
                                                  size_t offset) {
  std::lock_guard<std::mutex> lock(g_ranges_mutex);
  AllocationRange range;
  uintptr_t ptr_offset = 0;
  if (!find_range_locked(ptr, &range, &ptr_offset)) {
    return false;
  }
  if (ptr_offset + offset + bytes > range.size) {
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  range.bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, ptr_offset + offset);
  const auto end = std::chrono::steady_clock::now();
  g_allocator.sync_to_device_count.fetch_add(1, std::memory_order_relaxed);
  g_allocator.sync_to_device_time_us.fetch_add(elapsed_us(start, end),
                                              std::memory_order_relaxed);
  return true;
}

extern "C" bool nn2fpga_allocator_sync_from_device(void *ptr, size_t bytes,
                                                    size_t offset) {
  std::lock_guard<std::mutex> lock(g_ranges_mutex);
  AllocationRange range;
  uintptr_t ptr_offset = 0;
  if (!find_range_locked(ptr, &range, &ptr_offset)) {
    return false;
  }
  if (ptr_offset + offset + bytes > range.size) {
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  range.bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, ptr_offset + offset);
  const auto end = std::chrono::steady_clock::now();
  g_allocator.sync_from_device_count.fetch_add(1, std::memory_order_relaxed);
  g_allocator.sync_from_device_time_us.fetch_add(elapsed_us(start, end),
                                                std::memory_order_relaxed);
  return true;
}
