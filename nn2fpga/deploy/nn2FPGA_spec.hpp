#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <onnxruntime_cxx_api.h>
#include <vector>

enum class DType : uint8_t { u8, i8, i16, u16, i32, u32, f16, f32 };

inline size_t dtype_size(DType t) {
  switch (t) {
  case DType::u8:
    return 1;
  case DType::i8:
    return 1;
  case DType::i16:
    return 2;
  case DType::u16:
    return 2;
  case DType::i32:
    return 4;
  case DType::u32:
    return 4;
  case DType::f16:
    return 2;
  case DType::f32:
    return 4;
  }
  return 0;
}

inline size_t elements_per_image(const std::vector<int64_t> &inner) {
  size_t e = 1;
  for (auto d : inner)
    e *= static_cast<size_t>(d);
  return e;
}
inline size_t bytes_per_image(DType dt, const std::vector<int64_t> &inner) {
  return dtype_size(dt) * elements_per_image(inner);
}

enum class PortMode : uint8_t { Dynamic, StaticInit };

struct PortDesc {
  DType dtype;
  std::vector<int64_t> inner_dims; // Tensor shape (HWC format)
  off_t dma_off;                   // AXI-Lite offset

  PortMode mode;
  size_t buffer_size; // in bytes
};
