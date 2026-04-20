#pragma once
#include <cstring>
#include <fstream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "base64.h"
#include "xrt_dma.h"    // uses Mm2sSimple, S2mmSG
#include "xrt_mmio.hpp" // map_axil_window()
#include "xrt_ps.h"     // set_pl_from_iopll(), ZynqPllIndex
#include "xrt_pynq.h"   // program_with_pynq_cli_or_throw()
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>

#include "nn2FPGA_spec.hpp"

inline void check_ort_dtype(ONNXTensorElementDataType ort, DType d) {
  auto bad = [&]() {
    ORT_CXX_API_THROW("Type mismatch between ORT tensor and FPGA port dtype.",
                      ORT_INVALID_ARGUMENT);
  };
  switch (d) {
  case DType::u8:
    if (ort != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
      bad();
    break;
  case DType::i8:
    if (ort != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
      bad();
    break;
  case DType::i16:
    if (ort != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)
      bad();
    break;
  case DType::u16:
    if (ort != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)
      bad();
    break;
  case DType::i32:
    if (ort != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
      bad();
    break;
  case DType::u32:
    if (ort != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)
      bad();
    break;
  case DType::f16:
    if (ort != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
      bad();
    break;
  case DType::f32:
    if (ort != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      bad();
    break;
  }
}

// Generic FPGA runner (templated on Spec)
template <class Spec> class FpgaRunnerT {
public:
  static FpgaRunnerT &instance() {
    static FpgaRunnerT inst;
    return inst;
  }

  // Ensures that FPGA initialization (bitstream load, DMA setup, etc.) is
  // performed exactly once per process.
  //
  // We use std::call_once to guarantee:
  //   * thread-safe initialization
  //   * exactly-once execution across all kernel instances
  void ensure_loaded(const std::string &bit, const std::string &hwh,
                     const nlohmann::json &pkg) {
    std::call_once(init_once_, [&]() {
      load_bitstream(bit, hwh, pkg);
      initialized_ = true;
    });
  }

  // Runs the nn2FPGA kernel on a batch of inputs, producing outputs in host memory.
  void run(const std::vector<const void *> &in_ptrs,
           const std::vector<void *> &out_ptrs, size_t batch) {

    // Check that the bitstream has been loaded.
    if (!initialized_) {
      throw std::runtime_error("FPGA runner used before initialization");
    }

    // Basic sanity checks on input/output pointers and batch size.
    if (out_ptrs.size() != Spec::Outputs.size()) {
      throw std::invalid_argument("wrong #outputs");
    }
    if (batch == 0 || batch > static_cast<size_t>(Spec::N_MAX)) {
      ORT_CXX_API_THROW("Invalid batch (0 or > N_MAX).", ORT_INVALID_ARGUMENT);
    }

    std::lock_guard<std::mutex> lock(mtx_);

    // Copy + sync inputs
    for (size_t i = 0; i < Spec::Inputs.size(); ++i) {
      const auto &pd = Spec::Inputs[i];
      if (pd.mode == PortMode::StaticInit)
        continue; // already uploaded
      size_t bytes = batch * bytes_per_image(pd.dtype, pd.inner_dims);
      std::memcpy(in_host_ptrs_[i], in_ptrs[i], bytes);
      in_bos_[i].sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
    }

    // Prepare and start S2MM for each output
    for (size_t o = 0; o < Spec::Outputs.size(); ++o) {
      rx_[o]->transfer(static_cast<int>(batch));
    }

    // Start MM2S for each input
    for (size_t i = 0; i < Spec::Inputs.size(); ++i) {
      const auto &pd = Spec::Inputs[i];
      if (pd.mode == PortMode::StaticInit)
        continue; // already uploaded
      size_t bytes = batch * bytes_per_image(pd.dtype, pd.inner_dims);
      tx_[i]->transfer(bytes, 0);
    }

    // Wait inputs to complete
    for (size_t i = 0; i < Spec::Inputs.size(); ++i) {
      const auto &pd = Spec::Inputs[i];
      if (pd.mode == PortMode::StaticInit)
        continue; // already uploaded
      if (!tx_[i]->wait_done(200))
        throw std::runtime_error("MM2S timeout on input port " + std::to_string(i));
    }
    // Wait outputs to complete
    for (size_t o = 0; o < Spec::Outputs.size(); ++o) {
      if (!rx_[o]->wait_done(200, static_cast<int>(batch)))
        throw std::runtime_error("S2MM timeout on output port " + std::to_string(o));
    }

    // Sync back + copy to host
    for (size_t o = 0; o < Spec::Outputs.size(); ++o) {
      const auto &pd = Spec::Outputs[o];
      size_t bytes = batch * bytes_per_image(pd.dtype, pd.inner_dims);
      out_bos_[o].sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
      std::memcpy(out_ptrs[o], out_host_ptrs_[o], bytes);
    }
  }

private:
  FpgaRunnerT() : dev_(0) {}
  ~FpgaRunnerT() = default;
  
  void load_bitstream(const std::string &bit, const std::string &hwh,
                      const nlohmann::json &pkg) {

    // Create overlay files
    if (std::system("mkdir -p Overlay") != 0) {
      throw std::runtime_error("Failed to create Overlay directory");
    }
    {
      std::ofstream f("Overlay/design.bit", std::ios::binary);
      if (!f)
        throw std::runtime_error("Failed to open bitstream file");
      f.write(bit.data(), bit.size());
    }
    {
      std::ofstream f("Overlay/design.hwh");
      if (!f)
        throw std::runtime_error("Failed to open HWH file");
      f.write(hwh.data(), hwh.size());
    }

    // Program PL via PYNQ
    program_with_pynq_cli_or_throw("pynq_program.py", "Overlay/design.bit");

    // AXI-Lite map
    mmio_ = map_axil_window(Spec::AXIL_BASE, Spec::AXIL_SIZE);

    // Set FPGA clock
    float actual_freq =
        set_pl_from_iopll(static_cast<ZynqPllIndex>(Spec::PllIndex),
                          Spec::Freq_MHz, Spec::PLLFreq_MHz);

    fprintf(stderr, "FPGA clock set to %.2f MHz (IO PLL: %.2d MHz)\n",
            actual_freq, Spec::PLLFreq_MHz);

    // Build DMA ports and buffers
    build_ports();

    // One-shot upload of static inputs (e.g., weights) if provided in pkg
    upload_static_inputs_from_pkg(pkg);
  }

  void build_ports() {
    in_bos_.clear();
    out_bos_.clear();
    in_host_ptrs_.assign(Spec::Inputs.size(), nullptr);
    out_host_ptrs_.assign(Spec::Outputs.size(), nullptr);
    tx_.clear();
    rx_.clear();
    in_bos_.reserve(Spec::Inputs.size());
    out_bos_.reserve(Spec::Outputs.size());
    in_host_ptrs_.resize(Spec::Inputs.size());
    out_host_ptrs_.resize(Spec::Outputs.size());
    tx_.resize(Spec::Inputs.size());
    rx_.resize(Spec::Outputs.size());

    // Inputs: buffers + MM2S
    for (size_t i = 0; i < Spec::Inputs.size(); ++i) {
      const auto &pd = Spec::Inputs[i];
      const auto size = pd.buffer_size;
      in_bos_.emplace_back(dev_, size, 0, 0);
      in_host_ptrs_[i] = in_bos_.back().map<void *>();
      tx_[i].emplace(mmio_.regs, pd.dma_off, in_bos_.back());
    }

    // Outputs: buffers + S2MM SG
    for (size_t o = 0; o < Spec::Outputs.size(); ++o) {
      const auto &pd = Spec::Outputs[o];
      const auto size = pd.buffer_size;
      const size_t bpi = size / Spec::N_MAX; // bytes per image
      out_bos_.emplace_back(dev_, size, 0, 0);
      out_host_ptrs_[o] = out_bos_.back().map<void *>();
      rx_[o].emplace(mmio_.regs, pd.dma_off, dev_, out_bos_.back(), bpi,
                     Spec::N_MAX);
    }
  }

  void upload_static_inputs_from_pkg(const nlohmann::json &pkg) {
    if (!pkg.contains("input_map"))
      return;

    const auto &imap = pkg.at("input_map");

    for (const auto &entry : imap) {
      size_t i = entry.at("index").get<size_t>();

      const auto &pd = Spec::Inputs[i];
      if (pd.mode != PortMode::StaticInit) {
        continue;
      }

      if (!entry.contains("value") || entry.at("value").is_null()) {
        throw std::runtime_error("Static input port " + std::to_string(i) +
                                 " with name '" +
                                 entry.at("new_name").get<std::string>() +
                                 "' missing 'value' in package.");
      }

      const std::string raw =
          base64_decode(entry.at("value").get<std::string>());
      const size_t size = pd.buffer_size;
      if (raw.size() != size) {
        throw std::runtime_error("Static input port " + std::to_string(i) +
                                 " has wrong size in package, expected " +
                                 std::to_string(size * dtype_size(pd.dtype)) +
                                 " bytes, got " + std::to_string(raw.size()) +
                                 " bytes.");
      }

      std::memcpy(in_host_ptrs_[i], raw.data(), raw.size());
      in_bos_[i].sync(XCL_BO_SYNC_BO_TO_DEVICE, size, 0);

      tx_[i]->transfer(size, 0);
      if (!tx_[i]->wait_done(200)) {
        throw std::runtime_error("MM2S timeout during static upload on port " +
                                 std::to_string(i) + " with name " +
                                 entry.at("new_name").get<std::string>());
      }
    }
  }

  xrt::device dev_;
  Mmio mmio_;
  std::mutex mtx_;

  std::vector<xrt::bo> in_bos_;
  std::vector<xrt::bo> out_bos_;
  std::vector<void *> in_host_ptrs_;
  std::vector<void *> out_host_ptrs_;

  std::vector<std::optional<Mm2sSimple>> tx_;
  std::vector<std::optional<S2mmSG>> rx_;

  std::once_flag init_once_;
  bool initialized_ = false;
};

// ORT Kernel
template <class Spec> struct Nn2FpgaKernelT {
  Nn2FpgaKernelT(const OrtApi &api, const OrtKernelInfo *info) {
    Ort::ConstKernelInfo kinfo(info);
    const std::string pkg_json =
        kinfo.GetAttribute<std::string>("accelerator_package");
    nlohmann::json pkg = nlohmann::json::parse(pkg_json);
    const std::string bit =
        base64_decode(pkg.at("bitstream_b64").get<std::string>());
    const std::string hwh = base64_decode(pkg.at("hwh_b64").get<std::string>());

    FpgaRunnerT<Spec>::instance().ensure_loaded(bit, hwh, pkg);
  }

  void Compute(OrtKernelContext *ctx) {
    Ort::KernelContext kctx{ctx};

    const size_t Nin_spec = Spec::Inputs.size();
    const size_t Nout = Spec::Outputs.size();

    std::vector<const void *> in_ptrs(Nin_spec, nullptr);
    std::vector<void *> out_ptrs(Nout);

    int64_t batch = -1;

    int ort_in_idx = 0;
    for (size_t i = 0; i < Nin_spec; ++i) {
      if (Spec::Inputs[i].mode == PortMode::StaticInit)
        continue; // skip static in ORT

      Ort::ConstValue vin{
          kctx.GetInput(ort_in_idx++)};
      auto info = vin.GetTensorTypeAndShapeInfo();
      auto shape = info.GetShape();

      check_ort_dtype(info.GetElementType(), Spec::Inputs[i].dtype);
      if (shape.empty())
        ORT_CXX_API_THROW("Inputs must be at least 1D (batch).",
                          ORT_INVALID_ARGUMENT);

      if (batch < 0)
        batch = shape[0];
      if (shape[0] != batch)
        ORT_CXX_API_THROW("All inputs must share the same batch size.",
                          ORT_INVALID_ARGUMENT);

      const auto &idims = Spec::Inputs[i].inner_dims;
      if (shape.size() - 1 != idims.size())
        ORT_CXX_API_THROW("Input rank mismatch vs. spec.",
                          ORT_INVALID_ARGUMENT);
      for (size_t d = 0; d < idims.size(); ++d)
        if (shape[1 + d] != idims[d])
          ORT_CXX_API_THROW("Input shape mismatch vs. spec.",
                            ORT_INVALID_ARGUMENT);

      in_ptrs[i] = vin.GetTensorData<uint8_t>();
    }

    if (batch <= 0 || batch > Spec::N_MAX)
      ORT_CXX_API_THROW("Batch exceeds compiled N_MAX.", ORT_INVALID_ARGUMENT);

    // Prepare outputs with full shapes = {B} + inner_dims
    for (size_t o = 0; o < Nout; ++o) {
      const auto &odims = Spec::Outputs[o].inner_dims;
      std::vector<int64_t> out_shape;
      out_shape.reserve(1 + odims.size());
      out_shape.push_back(batch);
      out_shape.insert(out_shape.end(), odims.begin(), odims.end());

      auto vout = kctx.GetOutput(static_cast<int>(o), out_shape.data(),
                                 out_shape.size());
      auto info = vout.GetTensorTypeAndShapeInfo();
      check_ort_dtype(info.GetElementType(), Spec::Outputs[o].dtype);
      out_ptrs[o] = vout.GetTensorMutableData<uint8_t>();
    }

    FpgaRunnerT<Spec>::instance().run(in_ptrs, out_ptrs,
                                      static_cast<size_t>(batch));
  }
};

// ORT Op wrapper
template <class Spec>
struct Nn2FpgaOpT : Ort::CustomOpBase<Nn2FpgaOpT<Spec>, Nn2FpgaKernelT<Spec>> {

  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
    return new Nn2FpgaKernelT<Spec>(api, info);
  }
  const char *GetName() const { return Spec::kOpName; }
  const char *GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  }

  size_t GetInputTypeCount() const { return dyn_input_count(); }
  size_t GetOutputTypeCount() const { return Spec::Outputs.size(); }

  ONNXTensorElementDataType GetInputType(size_t i) const {
    const size_t spec_i = dyn_to_spec(i);
    return Spec::OrtInputTypes[spec_i];
  }

  ONNXTensorElementDataType GetOutputType(size_t i) const {
    return Spec::OrtOutputTypes[i];
  }

private:
  static size_t dyn_input_count() {
    size_t c = 0;
    for (const auto &pd : Spec::Inputs)
      if (pd.mode != PortMode::StaticInit)
        ++c;
    return c;
  }

  static size_t dyn_to_spec(size_t dyn_idx) {
    size_t c = 0;
    for (size_t i = 0; i < Spec::Inputs.size(); ++i) {
      if (Spec::Inputs[i].mode == PortMode::StaticInit)
        continue;
      if (c == dyn_idx)
        return i;
      ++c;
    }
    throw std::out_of_range("dyn_to_spec index");
  }
};
