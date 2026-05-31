#include <onnxruntime_c_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "nn2FPGA_allocator.hpp"

namespace {

const OrtApi *g_ort = nullptr;

void fail_on_status(OrtStatus *status, const char *what) {
  if (status == nullptr) {
    return;
  }
  const char *msg = g_ort->GetErrorMessage(status);
  std::fprintf(stderr, "ERROR: %s: %s\n", what, msg);
  g_ort->ReleaseStatus(status);
  std::exit(1);
}

void usage(const char *argv0) {
  std::fprintf(stderr,
               "Usage: %s [--no-allocator] [--profile] <model.onnx> <custom_op.so> [runs] [dynamic_batch]\n"
               "\n"
               "By default, registers the nn2FPGA XRT allocator on the ORT\n"
               "environment and allocates model inputs with that allocator.\n"
               "Use --no-allocator to run the same C++ ORT session without\n"
               "registering the allocator. Use --profile to emit an ORT trace.\n",
               argv0);
}

bool env_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool runner_verbose() { return env_enabled("NN2FPGA_RUNNER_VERBOSE"); }

bool mem_pattern_enabled() { return env_enabled("NN2FPGA_ENABLE_MEM_PATTERN"); }

void print_allocator_stats(const char *label) {
  const Nn2FpgaAllocatorStats stats = nn2fpga_allocator_stats();
  std::fprintf(stderr,
               "Allocator stats (%s): alloc_count=%zu free_count=%zu live_count=%zu pooled_count=%zu allocated_bytes=%zu xrt_alloc_count=%zu pool_reuse_count=%zu xrt_alloc_time=%.3f ms map_time=%.3f ms sync_to_device_count=%lu sync_to_device_time=%.3f ms sync_from_device_count=%lu sync_from_device_time=%.3f ms\n",
               label, stats.alloc_count, stats.free_count, stats.live_count,
               stats.pooled_count, stats.allocated_bytes, stats.xrt_alloc_count,
               stats.pool_reuse_count, stats.xrt_alloc_time_us / 1000.0,
               stats.map_time_us / 1000.0,
               static_cast<unsigned long>(stats.sync_to_device_count),
               stats.sync_to_device_time_us / 1000.0,
               static_cast<unsigned long>(stats.sync_from_device_count),
               stats.sync_from_device_time_us / 1000.0);
}

std::string element_type_name(ONNXTensorElementDataType type) {
  switch (type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return "float";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return "uint8";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return "int8";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return "uint16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return "int16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return "int32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return "int64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return "bool";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return "uint32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    return "uint64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return "double";
  default:
    return "type_" + std::to_string(static_cast<int>(type));
  }
}

size_t element_size(ONNXTensorElementDataType type) {
  switch (type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return sizeof(float);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return sizeof(uint8_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return sizeof(int8_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return sizeof(uint16_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return sizeof(int16_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return sizeof(int32_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return sizeof(int64_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return sizeof(bool);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return sizeof(uint32_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    return sizeof(uint64_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return sizeof(double);
  default:
    std::fprintf(stderr, "ERROR: unsupported input element type %d\n",
                 static_cast<int>(type));
    std::exit(2);
  }
}

size_t element_count(const std::vector<int64_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                         [](size_t acc, int64_t dim) {
                           return acc * static_cast<size_t>(dim);
                         });
}

std::string shape_string(const std::vector<int64_t> &shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out += ",";
    }
    out += std::to_string(shape[i]);
  }
  out += "]";
  return out;
}

void fill_input(void *data, size_t count, ONNXTensorElementDataType type) {
  switch (type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
    auto *p = static_cast<float *>(data);
    for (size_t i = 0; i < count; ++i) {
      p[i] = static_cast<float>((static_cast<int>(i) % 17) - 8) / 8.0f;
    }
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
    auto *p = static_cast<double *>(data);
    for (size_t i = 0; i < count; ++i) {
      p[i] = static_cast<double>((static_cast<int>(i) % 17) - 8) / 8.0;
    }
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
    auto *p = static_cast<int8_t *>(data);
    for (size_t i = 0; i < count; ++i) {
      p[i] = static_cast<int8_t>((static_cast<int>(i) % 255) - 128);
    }
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
    auto *p = static_cast<uint8_t *>(data);
    for (size_t i = 0; i < count; ++i) {
      p[i] = static_cast<uint8_t>(i % 255);
    }
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
    auto *p = static_cast<int16_t *>(data);
    for (size_t i = 0; i < count; ++i) {
      p[i] = static_cast<int16_t>((static_cast<int>(i) % 511) - 255);
    }
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
    auto *p = static_cast<uint16_t *>(data);
    for (size_t i = 0; i < count; ++i) {
      p[i] = static_cast<uint16_t>(i % 511);
    }
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
    auto *p = static_cast<int32_t *>(data);
    for (size_t i = 0; i < count; ++i) {
      p[i] = static_cast<int32_t>((static_cast<int>(i) % 1023) - 511);
    }
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
    auto *p = static_cast<uint32_t *>(data);
    for (size_t i = 0; i < count; ++i) {
      p[i] = static_cast<uint32_t>(i);
    }
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
    auto *p = static_cast<int64_t *>(data);
    for (size_t i = 0; i < count; ++i) {
      p[i] = static_cast<int64_t>(i % 1023) - 511;
    }
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: {
    auto *p = static_cast<uint64_t *>(data);
    for (size_t i = 0; i < count; ++i) {
      p[i] = static_cast<uint64_t>(i);
    }
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
    auto *p = static_cast<bool *>(data);
    for (size_t i = 0; i < count; ++i) {
      p[i] = (i % 2) != 0;
    }
    break;
  }
  default:
    std::fprintf(stderr, "ERROR: unsupported input element type %d\n",
                 static_cast<int>(type));
    std::exit(2);
  }
}

struct TensorMeta {
  std::string name;
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  std::vector<int64_t> shape;
};

TensorMeta input_meta(OrtSession *session, OrtAllocator *name_allocator,
                      size_t index, int64_t dynamic_batch) {
  char *name = nullptr;
  fail_on_status(g_ort->SessionGetInputName(session, index, name_allocator, &name),
                 "SessionGetInputName");

  OrtTypeInfo *type_info = nullptr;
  fail_on_status(g_ort->SessionGetInputTypeInfo(session, index, &type_info),
                 "SessionGetInputTypeInfo");
  const OrtTensorTypeAndShapeInfo *tensor_info = nullptr;
  fail_on_status(g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info),
                 "CastTypeInfoToTensorInfo(input)");

  ONNXTensorElementDataType type;
  fail_on_status(g_ort->GetTensorElementType(tensor_info, &type),
                 "GetTensorElementType(input)");
  size_t rank = 0;
  fail_on_status(g_ort->GetDimensionsCount(tensor_info, &rank),
                 "GetDimensionsCount(input)");
  std::vector<int64_t> shape(rank);
  fail_on_status(g_ort->GetDimensions(tensor_info, shape.data(), rank),
                 "GetDimensions(input)");

  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0) {
      shape[i] = (i == 0) ? dynamic_batch : 1;
    }
  }

  TensorMeta meta;
  meta.name = name;
  meta.type = type;
  meta.shape = shape;
  name_allocator->Free(name_allocator, name);
  g_ort->ReleaseTypeInfo(type_info);
  return meta;
}

TensorMeta output_meta(OrtSession *session, OrtAllocator *name_allocator,
                       size_t index) {
  char *name = nullptr;
  fail_on_status(g_ort->SessionGetOutputName(session, index, name_allocator, &name),
                 "SessionGetOutputName");

  TensorMeta meta;
  meta.name = name;
  name_allocator->Free(name_allocator, name);
  return meta;
}

} // namespace

int main(int argc, char **argv) {
  bool use_allocator = true;
  bool enable_profiling = false;

  int argi = 1;
  while (argi < argc && std::strncmp(argv[argi], "--", 2) == 0) {
    if (std::strcmp(argv[argi], "--no-allocator") == 0) {
      use_allocator = false;
    } else if (std::strcmp(argv[argi], "--profile") == 0) {
      enable_profiling = true;
    } else {
      std::fprintf(stderr, "ERROR: unknown option '%s'\n", argv[argi]);
      usage(argv[0]);
      return 2;
    }
    ++argi;
  }

  const int remaining = argc - argi;
  if (remaining < 2 || remaining > 4) {
    usage(argv[0]);
    return 2;
  }

  const char *model_path = argv[argi];
  const char *custom_op_path = argv[argi + 1];
  const int runs = remaining >= 3 ? std::atoi(argv[argi + 2]) : 1;
  const int64_t dynamic_batch = remaining >= 4 ? std::atoll(argv[argi + 3]) : 1;

  if (runs <= 0 || dynamic_batch <= 0) {
    std::fprintf(stderr, "ERROR: runs and dynamic_batch must be positive.\n");
    return 2;
  }

  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  OrtEnv *env = nullptr;
  fail_on_status(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                  "nn2fpga_allocator_runner", &env),
                 "CreateEnv");
  if (use_allocator) {
    fail_on_status(nn2fpga_register_xrt_cpu_allocator(env, g_ort),
                   "nn2fpga_register_xrt_cpu_allocator");
  } else {
    std::fprintf(stderr, "Running without nn2FPGA allocator registration\n");
  }

  OrtSessionOptions *session_options = nullptr;
  fail_on_status(g_ort->CreateSessionOptions(&session_options),
                 "CreateSessionOptions");
  if (use_allocator) {
    fail_on_status(g_ort->AddSessionConfigEntry(session_options,
                                                "session.use_env_allocators", "1"),
                   "AddSessionConfigEntry(session.use_env_allocators)");
    if (mem_pattern_enabled()) {
      std::fprintf(stderr, "ORT memory pattern: enabled\n");
    } else {
      std::fprintf(stderr, "ORT memory pattern: disabled\n");
      fail_on_status(g_ort->DisableMemPattern(session_options), "DisableMemPattern");
    }
  }
  fail_on_status(g_ort->SetSessionGraphOptimizationLevel(session_options,
                                                         ORT_ENABLE_ALL),
                 "SetSessionGraphOptimizationLevel(ORT_ENABLE_ALL)");
  if (enable_profiling) {
    fail_on_status(g_ort->EnableProfiling(session_options,
                                          "nn2fpga_runner_profile"),
                   "EnableProfiling");
  }

  void *custom_op_handle = nullptr;
  std::fprintf(stderr, "Registering custom op library %s\n", custom_op_path);
  fail_on_status(g_ort->RegisterCustomOpsLibrary(session_options, custom_op_path,
                                                 &custom_op_handle),
                 "RegisterCustomOpsLibrary");

  OrtSession *session = nullptr;
  std::fprintf(stderr, "Creating session for %s\n", model_path);
  fail_on_status(g_ort->CreateSession(env, model_path, session_options, &session),
                 "CreateSession");

  OrtAllocator *name_allocator = nullptr;
  fail_on_status(g_ort->GetAllocatorWithDefaultOptions(&name_allocator),
                 "GetAllocatorWithDefaultOptions");

  size_t input_count = 0;
  size_t output_count = 0;
  fail_on_status(g_ort->SessionGetInputCount(session, &input_count),
                 "SessionGetInputCount");
  fail_on_status(g_ort->SessionGetOutputCount(session, &output_count),
                 "SessionGetOutputCount");

  std::vector<TensorMeta> inputs;
  std::vector<TensorMeta> outputs;
  inputs.reserve(input_count);
  outputs.reserve(output_count);
  for (size_t i = 0; i < input_count; ++i) {
    inputs.push_back(input_meta(session, name_allocator, i, dynamic_batch));
    const size_t bytes = element_count(inputs.back().shape) * element_size(inputs.back().type);
    std::fprintf(stderr, "Input %zu: name=%s type=%s shape=%s bytes=%zu\n", i,
                 inputs.back().name.c_str(), element_type_name(inputs.back().type).c_str(),
                 shape_string(inputs.back().shape).c_str(), bytes);
  }
  for (size_t i = 0; i < output_count; ++i) {
    outputs.push_back(output_meta(session, name_allocator, i));
    std::fprintf(stderr, "Output %zu: name=%s\n", i, outputs.back().name.c_str());
  }

  std::vector<const char *> input_names;
  std::vector<const char *> output_names;
  std::vector<OrtValue *> input_values(input_count, nullptr);
  input_names.reserve(input_count);
  output_names.reserve(output_count);
  for (const auto &input : inputs) {
    input_names.push_back(input.name.c_str());
  }
  for (const auto &output : outputs) {
    output_names.push_back(output.name.c_str());
  }

  OrtAllocator *input_allocator =
      use_allocator ? nn2fpga_xrt_cpu_allocator() : name_allocator;
  for (size_t i = 0; i < input_count; ++i) {
    fail_on_status(g_ort->CreateTensorAsOrtValue(
                       input_allocator, inputs[i].shape.data(), inputs[i].shape.size(),
                       inputs[i].type, &input_values[i]),
                   "CreateTensorAsOrtValue(input)");
    void *data = nullptr;
    fail_on_status(g_ort->GetTensorMutableData(input_values[i], &data),
                   "GetTensorMutableData(input)");
    fill_input(data, element_count(inputs[i].shape), inputs[i].type);
  }

  double total_run_ms = 0.0;
  const bool verbose = runner_verbose();
  for (int run = 0; run < runs; ++run) {
    std::vector<OrtValue *> output_values(output_count, nullptr);
    if (verbose) {
      std::fprintf(stderr, "Running inference %d/%d\n", run + 1, runs);
    }
    const auto start = std::chrono::steady_clock::now();
    fail_on_status(g_ort->Run(session, nullptr, input_names.data(),
                              input_values.data(), input_count,
                              output_names.data(), output_count,
                              output_values.data()),
                   "Run");
    const auto end = std::chrono::steady_clock::now();
    const double run_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    total_run_ms += run_ms;
    if (verbose) {
      std::fprintf(stderr, "Inference %d latency: %.3f ms\n", run + 1, run_ms);
    }

    for (size_t i = 0; i < output_count; ++i) {
      OrtTensorTypeAndShapeInfo *info = nullptr;
      fail_on_status(g_ort->GetTensorTypeAndShape(output_values[i], &info),
                     "GetTensorTypeAndShape(output)");
      ONNXTensorElementDataType type;
      fail_on_status(g_ort->GetTensorElementType(info, &type),
                     "GetTensorElementType(output)");
      size_t rank = 0;
      fail_on_status(g_ort->GetDimensionsCount(info, &rank),
                     "GetDimensionsCount(output)");
      std::vector<int64_t> shape(rank);
      fail_on_status(g_ort->GetDimensions(info, shape.data(), rank),
                     "GetDimensions(output)");
      const size_t bytes = element_count(shape) * element_size(type);
      void *data = nullptr;
      fail_on_status(g_ort->GetTensorMutableData(output_values[i], &data),
                     "GetTensorMutableData(output)");
      if (verbose || run == runs - 1) {
        std::fprintf(stderr,
                     "Output %zu: type=%s shape=%s bytes=%zu ptr=%p first_byte=0x%02x\n",
                     i, element_type_name(type).c_str(), shape_string(shape).c_str(),
                     bytes, data, bytes ? static_cast<unsigned>(static_cast<uint8_t *>(data)[0]) : 0);
      }
      g_ort->ReleaseTensorTypeAndShapeInfo(info);
      g_ort->ReleaseValue(output_values[i]);
    }
  }

  std::fprintf(stderr, "Average inference latency: %.3f ms over %d run(s)\n",
               total_run_ms / static_cast<double>(runs), runs);
  if (use_allocator) {
    print_allocator_stats("after runs");
    if (env_enabled("NN2FPGA_ALLOCATOR_STATS")) {
      nn2fpga_allocator_dump_stats(stderr);
    }
  }

  if (enable_profiling) {
    char *profile_file = nullptr;
    fail_on_status(g_ort->SessionEndProfiling(session, name_allocator,
                                              &profile_file),
                   "SessionEndProfiling");
    if (profile_file != nullptr) {
      std::fprintf(stderr, "ORT profiling trace: %s\n", profile_file);
      name_allocator->Free(name_allocator, profile_file);
    }
  }

  for (OrtValue *value : input_values) {
    g_ort->ReleaseValue(value);
  }
  g_ort->ReleaseSession(session);
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseEnv(env);
  if (use_allocator) {
    print_allocator_stats("after cleanup");
  }
  return 0;
}
