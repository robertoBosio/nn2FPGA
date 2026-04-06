#include <onnxruntime_cxx_api.h>
#include "nn2FPGA_kernel.hpp"
#include "generated_spec.hpp"

extern "C" OrtStatus* ORT_API_CALL
RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);

#ifdef ORT_API_MANUAL_INIT
  Ort::InitApi(api);
#endif
  try {
    Ort::CustomOpDomain domain{OpSpec::kDomain};
    static Nn2FpgaOpT<OpSpec> op;
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
