#include <onnxruntime_c_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nn2FPGA_allocator.hpp"

namespace {

const OrtApi* g_ort = nullptr;

void fail_on_status(OrtStatus* status, const char* what) {
  if (status == nullptr) {
    return;
  }
  const char* msg = g_ort->GetErrorMessage(status);
  std::fprintf(stderr, "ERROR: %s: %s\n", what, msg);
  g_ort->ReleaseStatus(status);
  std::exit(1);
}

void usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s <model.onnx> [runs] [custom_op.so]\n"
               "\n"
               "The model is expected to use input name 'input' and output name 'output'.\n"
               "Use tools/make_allocator_test_model.py or tools/make_allocator_probe_model.py\n"
               "to generate a matching model. The allocator is always XRT-backed.\n",
               argv0);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    usage(argv[0]);
    return 2;
  }

  const char* model_path = argv[1];
  const int runs = argc >= 3 ? std::atoi(argv[2]) : 3;
  const char* custom_op_path = argc >= 4 ? argv[3] : nullptr;

  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  OrtEnv* env = nullptr;
  fail_on_status(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "nn2fpga_allocator_test", &env),
                 "CreateEnv");

  fail_on_status(nn2fpga_register_xrt_cpu_allocator(env, g_ort),
                 "nn2fpga_register_xrt_cpu_allocator");

  OrtSessionOptions* session_options = nullptr;
  fail_on_status(g_ort->CreateSessionOptions(&session_options), "CreateSessionOptions");
  fail_on_status(g_ort->AddSessionConfigEntry(session_options, "session.use_env_allocators", "1"),
                 "AddSessionConfigEntry(session.use_env_allocators)");
  fail_on_status(g_ort->DisableMemPattern(session_options), "DisableMemPattern");
  fail_on_status(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_DISABLE_ALL),
                 "SetSessionGraphOptimizationLevel");
  void* custom_op_handle = nullptr;
  if (custom_op_path != nullptr && std::strlen(custom_op_path) > 0) {
    std::fprintf(stderr, "Registering custom op library %s\n", custom_op_path);
    fail_on_status(g_ort->RegisterCustomOpsLibrary(session_options, custom_op_path,
                                                   &custom_op_handle),
                   "RegisterCustomOpsLibrary");
  }

  OrtSession* session = nullptr;
  std::fprintf(stderr, "Creating session for %s\n", model_path);
  fail_on_status(g_ort->CreateSession(env, model_path, session_options, &session),
                 "CreateSession");

  std::vector<float> input(1 * 3 * 16 * 16, 1.0f);
  std::vector<int64_t> input_shape{1, 3, 16, 16};

  OrtMemoryInfo* input_memory_info = nullptr;
  fail_on_status(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                            &input_memory_info),
                 "CreateCpuMemoryInfo(input)");

  OrtValue* input_tensor = nullptr;
  fail_on_status(g_ort->CreateTensorWithDataAsOrtValue(
                     input_memory_info, input.data(), input.size() * sizeof(float),
                     input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                     &input_tensor),
                 "CreateTensorWithDataAsOrtValue");

  const char* input_names[] = {"input"};
  const OrtValue* input_values[] = {input_tensor};
  const char* output_names[] = {"output"};

  for (int i = 0; i < runs; ++i) {
    OrtValue* output_tensor = nullptr;
    std::fprintf(stderr, "Running inference %d/%d\n", i + 1, runs);
    fail_on_status(g_ort->Run(session, nullptr, input_names,
                              input_values, 1,
                              output_names, 1, &output_tensor),
                   "Run");

    float* output_data = nullptr;
    fail_on_status(g_ort->GetTensorMutableData(output_tensor,
                                               reinterpret_cast<void**>(&output_data)),
                   "GetTensorMutableData(output)");
    std::fprintf(stderr, "Output first value: %.6f\n", output_data[0]);
    g_ort->ReleaseValue(output_tensor);
  }

  g_ort->ReleaseValue(input_tensor);
  g_ort->ReleaseMemoryInfo(input_memory_info);
  g_ort->ReleaseSession(session);
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseEnv(env);
  return 0;
}
