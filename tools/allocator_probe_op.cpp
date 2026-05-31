#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <vector>

namespace {

using ContainsFn = bool (*)(const void*, size_t*);
using LookupFn = bool (*)(const void*, size_t*, uint64_t*);

ContainsFn get_contains_fn() {
  static ContainsFn fn = []() -> ContainsFn {
    void* sym = dlsym(RTLD_DEFAULT, "nn2fpga_allocator_contains");
    return reinterpret_cast<ContainsFn>(sym);
  }();
  return fn;
}

bool allocator_contains(const void* ptr, size_t* size_out) {
  ContainsFn fn = get_contains_fn();
  if (fn == nullptr) {
    if (size_out != nullptr) {
      *size_out = 0;
    }
    return false;
  }
  return fn(ptr, size_out);
}

bool allocator_lookup(const void* ptr, size_t* size_out, uint64_t* device_addr_out) {
  static LookupFn fn = []() -> LookupFn {
    void* sym = dlsym(RTLD_DEFAULT, "nn2fpga_allocator_lookup");
    return reinterpret_cast<LookupFn>(sym);
  }();
  if (fn == nullptr) {
    if (size_out != nullptr) {
      *size_out = 0;
    }
    if (device_addr_out != nullptr) {
      *device_addr_out = 0;
    }
    return false;
  }
  return fn(ptr, size_out, device_addr_out);
}

struct AllocatorProbeKernel {
  AllocatorProbeKernel(const OrtApi&, const OrtKernelInfo*) {}

  void Compute(OrtKernelContext* ctx) {
    Ort::KernelContext kctx{ctx};

    Ort::ConstValue input{kctx.GetInput(0)};
    auto input_info = input.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      ORT_CXX_API_THROW("AllocatorProbe expects float input.", ORT_INVALID_ARGUMENT);
    }

    std::vector<int64_t> shape = input_info.GetShape();
    const size_t elements = input_info.GetElementCount();
    const size_t bytes = elements * sizeof(float);

    const float* input_ptr = input.GetTensorData<float>();
    auto output = kctx.GetOutput(0, shape.data(), shape.size());
    float* output_ptr = output.GetTensorMutableData<float>();

    size_t input_alloc_size = 0;
    size_t output_alloc_size = 0;
    uint64_t input_device_addr = 0;
    uint64_t output_device_addr = 0;
    const bool input_registered = allocator_lookup(input_ptr, &input_alloc_size,
                                                   &input_device_addr) ||
                                  allocator_contains(input_ptr, &input_alloc_size);
    const bool output_registered = allocator_lookup(output_ptr, &output_alloc_size,
                                                    &output_device_addr) ||
                                   allocator_contains(output_ptr, &output_alloc_size);

    std::fprintf(stderr,
                 "[allocator_probe] input ptr=%p registered=%d alloc_size=%zu bytes=%zu device_addr=0x%lx\n",
                 static_cast<const void*>(input_ptr), input_registered ? 1 : 0,
                 input_alloc_size, bytes, static_cast<unsigned long>(input_device_addr));
    std::fprintf(stderr,
                 "[allocator_probe] output ptr=%p registered=%d alloc_size=%zu bytes=%zu device_addr=0x%lx\n",
                 static_cast<void*>(output_ptr), output_registered ? 1 : 0,
                 output_alloc_size, bytes, static_cast<unsigned long>(output_device_addr));

    std::memcpy(output_ptr, input_ptr, bytes);
  }
};

struct AllocatorProbeOp
    : Ort::CustomOpBase<AllocatorProbeOp, AllocatorProbeKernel> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new AllocatorProbeKernel(api, info);
  }

  const char* GetName() const { return "AllocatorProbe"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }

  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api_base) {
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);

#ifdef ORT_API_MANUAL_INIT
  Ort::InitApi(api);
#endif

  try {
    Ort::CustomOpDomain domain{"ai.nn2FPGA.test"};
    static AllocatorProbeOp op;
    domain.Add(&op);
    Ort::ThrowOnError(api->AddCustomOpDomain(options, domain));
    domain.release();
    return nullptr;
  } catch (const Ort::Exception& e) {
    return api->CreateStatus(e.GetOrtErrorCode(), e.what());
  } catch (const std::exception& e) {
    return api->CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
  }
}
