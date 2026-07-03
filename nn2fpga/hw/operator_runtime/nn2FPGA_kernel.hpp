#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "base64.h"
#include "xrt_dma.h"    // uses Mm2sSimple, S2mmSimple
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

inline void write_u64(volatile uint32_t *regs, off_t off, uint64_t value) {
  wr32(regs, off, static_cast<uint32_t>(value & 0xFFFFFFFFull));
  wr32(regs, off + 4, static_cast<uint32_t>(value >> 32));
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

  // Runs the nn2FPGA kernel for one image, producing outputs in host memory.
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
    if (batch != 1) {
      ORT_CXX_API_THROW("nn2FPGA operator runtime currently supports batch == 1 only.",
                        ORT_INVALID_ARGUMENT);
    }

    const auto lease = acquire_slot();
    RequestSlot &slot = slots_[lease.slot_index];

    try {
      // Copy + sync this request's dynamic inputs into its private slot BOs.
      for (size_t i = 0; i < Spec::Inputs.size(); ++i) {
        const auto &pd = Spec::Inputs[i];
        if (pd.mode == PortMode::StaticInit)
          continue;
        const size_t bytes = bytes_per_image(pd.dtype, pd.inner_dims);
        std::memcpy(slot.in_host_ptrs[i], in_ptrs[i], bytes);
        slot.in_bos[i]->sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
      }

      trace_event("run_start", lease.sequence, lease.slot_index);
      submit_sg_request(lease.sequence, lease.slot_index, slot);
      wait_sg_outputs(lease.sequence, lease.slot_index, slot);

      complete_sg_request(lease.sequence, lease.slot_index, slot, out_ptrs);
    } catch (...) {
      trace_event("run_exception", lease.sequence, lease.slot_index);
      reset_sg_after_error(slot);
      release_slot(lease.slot_index);
      throw;
    }

    trace_event("run_done", lease.sequence, lease.slot_index);
    release_slot(lease.slot_index);
  }

private:
  struct RequestSlot {
    bool in_use = false;
    uint64_t sequence = 0;
    std::vector<std::optional<xrt::bo>> in_bos;
    std::vector<std::optional<xrt::bo>> out_bos;
    std::vector<void *> in_host_ptrs;
    std::vector<void *> out_host_ptrs;
    std::vector<std::optional<AxiDmaSgRing::Handle>> input_descs;
    std::vector<std::optional<AxiDmaSgRing::Handle>> output_descs;
  };

  struct SlotLease {
    size_t slot_index;
    uint64_t sequence;
  };

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
    static_in_bos_.clear();
    buffer_bos_.clear();
    static_in_host_ptrs_.assign(Spec::Inputs.size(), nullptr);
    buffer_host_ptrs_.assign(Spec::Buffers.size(), nullptr);
    static_tx_.clear();
    input_rings_.clear();
    output_rings_.clear();
    slots_.clear();
    static_in_bos_.resize(Spec::Inputs.size());
    static_tx_.resize(Spec::Inputs.size());
    input_rings_.resize(Spec::Inputs.size());
    output_rings_.resize(Spec::Outputs.size());
    buffer_bos_.reserve(Spec::Buffers.size());
    buffer_host_ptrs_.resize(Spec::Buffers.size());
    const size_t slot_count = std::max<size_t>(1, static_cast<size_t>(Spec::N_MAX));

    // Static inputs: one-time BOs + MM2S upload path.
    for (size_t i = 0; i < Spec::Inputs.size(); ++i) {
      const auto &pd = Spec::Inputs[i];
      if (pd.mode != PortMode::StaticInit)
        continue;
      static_in_bos_[i].emplace(dev_, pd.buffer_size, 0, 0);
      static_in_host_ptrs_[i] = static_in_bos_[i]->template map<void *>();
      static_tx_[i].emplace(mmio_.regs, pd.dma_off, *static_in_bos_[i]);
    }

    // Dynamic stream DMAs use SG rings. Static inputs remain simple one-shot
    // MM2S uploads because they are programmed once during initialization.
    for (size_t i = 0; i < Spec::Inputs.size(); ++i) {
      const auto &pd = Spec::Inputs[i];
      if (pd.mode == PortMode::StaticInit)
        continue;
      input_rings_[i] = std::make_unique<AxiDmaSgRing>(
          mmio_.regs, pd.dma_off, dev_, AxiDmaDirection::Mm2s, slot_count,
          "input" + std::to_string(i));
    }

    for (size_t o = 0; o < Spec::Outputs.size(); ++o) {
      const auto &pd = Spec::Outputs[o];
      output_rings_[o] = std::make_unique<AxiDmaSgRing>(
          mmio_.regs, pd.dma_off, dev_, AxiDmaDirection::S2mm, slot_count,
          "output" + std::to_string(o));
    }

    build_request_slots();

    // Internal DDR-backed buffers used by HLS m_axi pointer ports.
    for (size_t b = 0; b < Spec::Buffers.size(); ++b) {
      const auto &bd = Spec::Buffers[b];
      buffer_bos_.emplace_back(dev_, bd.size_bytes, 0, 0);
      buffer_host_ptrs_[b] = buffer_bos_.back().map<void *>();
      std::memset(buffer_host_ptrs_[b], 0, bd.size_bytes);
      buffer_bos_.back().sync(XCL_BO_SYNC_BO_TO_DEVICE, bd.size_bytes, 0);

      const uint64_t addr = buffer_bos_.back().address();
      write_u64(mmio_.regs, Spec::ControlAxiOffset + bd.read_axi_off, addr);
      write_u64(mmio_.regs, Spec::ControlAxiOffset + bd.write_axi_off, addr);
    }
  }

  void build_request_slots() {
    const size_t slot_count = std::max<size_t>(1, static_cast<size_t>(Spec::N_MAX));
    slots_.reserve(slot_count);

    for (size_t s = 0; s < slot_count; ++s) {
      slots_.emplace_back();
      auto &slot = slots_.back();
      slot.in_bos.resize(Spec::Inputs.size());
      slot.out_bos.resize(Spec::Outputs.size());
      slot.in_host_ptrs.assign(Spec::Inputs.size(), nullptr);
      slot.out_host_ptrs.assign(Spec::Outputs.size(), nullptr);
      slot.input_descs.resize(Spec::Inputs.size());
      slot.output_descs.resize(Spec::Outputs.size());

      for (size_t i = 0; i < Spec::Inputs.size(); ++i) {
        const auto &pd = Spec::Inputs[i];
        if (pd.mode == PortMode::StaticInit)
          continue;
        const size_t bytes = bytes_per_image(pd.dtype, pd.inner_dims);
        slot.in_bos[i].emplace(dev_, bytes, 0, 0);
        slot.in_host_ptrs[i] = slot.in_bos[i]->template map<void *>();
        trace_bo("slot_input_bo", s, i, slot.in_bos[i]->address(), bytes);
      }

      for (size_t o = 0; o < Spec::Outputs.size(); ++o) {
        const auto &pd = Spec::Outputs[o];
        const size_t bytes = bytes_per_image(pd.dtype, pd.inner_dims);
        slot.out_bos[o].emplace(dev_, bytes, 0, 0);
        slot.out_host_ptrs[o] = slot.out_bos[o]->template map<void *>();
        trace_bo("slot_output_bo", s, o, slot.out_bos[o]->address(), bytes);
      }
    }
  }

  SlotLease acquire_slot() {
    std::unique_lock<std::mutex> lock(slot_mtx_);
    slot_cv_.wait(lock, [&]() {
      for (const auto &slot : slots_) {
        if (!slot.in_use)
          return true;
      }
      return false;
    });

    for (size_t i = 0; i < slots_.size(); ++i) {
      if (!slots_[i].in_use) {
        const uint64_t sequence = next_sequence_.fetch_add(1);
        slots_[i].in_use = true;
        slots_[i].sequence = sequence;
        trace_event("slot_acquired", sequence, i);
        return SlotLease{i, sequence};
      }
    }

    throw std::runtime_error("No free nn2FPGA request slot after wait.");
  }

  void release_slot(size_t slot_index) {
    {
      std::lock_guard<std::mutex> lock(slot_mtx_);
      trace_event("slot_release", slots_[slot_index].sequence, slot_index);
      slots_[slot_index].in_use = false;
    }
    slot_cv_.notify_one();
  }

  void submit_sg_request(uint64_t sequence, size_t slot_index, RequestSlot &slot) {
    std::unique_lock<std::mutex> lock(submit_mtx_);
    trace_event("sg_submit_wait_enter", sequence, slot_index);
    submit_cv_.wait(lock, [&]() { return next_submit_sequence_.load() == sequence; });
    trace_event("sg_submit_wait_done", sequence, slot_index);

    // Critical invariant: every output destination for this sequence is queued
    // before the input descriptor can let the accelerator produce that output.
    for (size_t o = 0; o < Spec::Outputs.size(); ++o) {
      const auto &pd = Spec::Outputs[o];
      const size_t bytes = bytes_per_image(pd.dtype, pd.inner_dims);
      trace_event("sg_output_enqueue", sequence, slot_index, static_cast<int>(o));
      slot.output_descs[o] = output_rings_[o]->enqueue(sequence, *slot.out_bos[o], bytes);
      trace_dma("sg_output_enqueue", sequence, slot_index, static_cast<int>(o),
                output_rings_[o]->debug_status(&*slot.output_descs[o]));
    }

    for (size_t i = 0; i < Spec::Inputs.size(); ++i) {
      const auto &pd = Spec::Inputs[i];
      if (pd.mode == PortMode::StaticInit)
        continue;
      const size_t bytes = bytes_per_image(pd.dtype, pd.inner_dims);
      trace_event("sg_input_enqueue", sequence, slot_index, static_cast<int>(i));
      slot.input_descs[i] = input_rings_[i]->enqueue(sequence, *slot.in_bos[i], bytes);
      trace_dma("sg_input_enqueue", sequence, slot_index, static_cast<int>(i),
                input_rings_[i]->debug_status(&*slot.input_descs[i]));
    }

    next_submit_sequence_.fetch_add(1);
    trace_event("sg_submit_sequence_advance", sequence, slot_index);
    lock.unlock();
    submit_cv_.notify_all();
  }

  void wait_sg_outputs(uint64_t sequence, size_t slot_index, RequestSlot &slot) {
    for (size_t o = 0; o < Spec::Outputs.size(); ++o) {
      trace_event("sg_output_wait_start", sequence, slot_index, static_cast<int>(o));
      if (!slot.output_descs[o].has_value())
        throw std::runtime_error("Missing SG output descriptor for output port " + std::to_string(o));
      if (!output_rings_[o]->wait_done(*slot.output_descs[o], 2000)) {
        trace_event("sg_output_timeout", sequence, slot_index, static_cast<int>(o));
        trace_dma("sg_output_timeout", sequence, slot_index, static_cast<int>(o),
                  output_rings_[o]->debug_status(&*slot.output_descs[o]));
        trace_ctrl_regs("sg_output_timeout", sequence, slot_index);
        throw std::runtime_error("S2MM SG timeout on output port " + std::to_string(o));
      }
      trace_event("sg_output_wait_done", sequence, slot_index, static_cast<int>(o));
    }
  }

  void complete_sg_request(uint64_t sequence, size_t slot_index, RequestSlot &slot,
                           const std::vector<void *> &out_ptrs) {
    std::unique_lock<std::mutex> lock(completion_mtx_);
    trace_event("sg_complete_wait_enter", sequence, slot_index);
    completion_cv_.wait(lock, [&]() { return next_complete_sequence_.load() == sequence; });
    trace_event("sg_complete_wait_done", sequence, slot_index);

    // Copy and reclaim in sequence order. The AXI DMA walks a circular BD ring;
    // reusing descriptors out of order can break the tail/next chain even when
    // later ORT worker threads observe their descriptors as complete first.
    for (size_t o = 0; o < Spec::Outputs.size(); ++o) {
      const auto &pd = Spec::Outputs[o];
      const size_t bytes = bytes_per_image(pd.dtype, pd.inner_dims);
      slot.out_bos[o]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
      std::memcpy(out_ptrs[o], slot.out_host_ptrs[o], bytes);
    }

    reclaim_sg_descriptors(sequence, slot_index, slot);
    next_complete_sequence_.fetch_add(1);
    trace_event("sg_complete_sequence_advance", sequence, slot_index);
    lock.unlock();
    completion_cv_.notify_all();
  }

  void reclaim_sg_descriptors(uint64_t sequence, size_t slot_index, RequestSlot &slot) {
    for (size_t i = 0; i < Spec::Inputs.size(); ++i) {
      const auto &pd = Spec::Inputs[i];
      if (pd.mode == PortMode::StaticInit || !slot.input_descs[i].has_value())
        continue;
      trace_event("sg_input_reclaim", sequence, slot_index, static_cast<int>(i));
      if (!input_rings_[i]->wait_done(*slot.input_descs[i], 2000)) {
        trace_dma("sg_input_reclaim_timeout", sequence, slot_index, static_cast<int>(i),
                  input_rings_[i]->debug_status(&*slot.input_descs[i]));
        throw std::runtime_error("MM2S SG timeout on input port " + std::to_string(i));
      }
      input_rings_[i]->reclaim(*slot.input_descs[i]);
      slot.input_descs[i].reset();
    }

    for (size_t o = 0; o < Spec::Outputs.size(); ++o) {
      if (!slot.output_descs[o].has_value())
        continue;
      trace_event("sg_output_reclaim", sequence, slot_index, static_cast<int>(o));
      output_rings_[o]->reclaim(*slot.output_descs[o]);
      slot.output_descs[o].reset();
    }
  }

  void reset_sg_after_error(RequestSlot &slot) {
    for (auto &ring : input_rings_) {
      if (ring)
        ring->reset_and_release_all();
    }
    for (auto &ring : output_rings_) {
      if (ring)
        ring->reset_and_release_all();
    }
    for (auto &desc : slot.input_descs)
      desc.reset();
    for (auto &desc : slot.output_descs)
      desc.reset();
  }

  static bool env_flag(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
  }

  static bool trace_enabled() {
    static const bool enabled = env_flag("NN2FPGA_TRACE");
    return enabled;
  }

  static bool trace_dma_enabled() {
    static const bool enabled = env_flag("NN2FPGA_TRACE_DMA");
    return enabled;
  }

  static bool trace_ctrl_enabled() {
    static const bool enabled = env_flag("NN2FPGA_TRACE_CTRL");
    return enabled;
  }

  void trace_event(const char *event, uint64_t sequence, size_t slot_index,
                   int port = -1) const {
    if (!trace_enabled())
      return;

    using namespace std::chrono;
    const auto now = steady_clock::now().time_since_epoch();
    const auto us = duration_cast<microseconds>(now).count();
    const uint64_t next_submit = next_submit_sequence_.load();
    const uint64_t next_complete = next_complete_sequence_.load();

    std::fprintf(stderr,
                 "[nn2fpga trace] t_us=%lld event=%s seq=%llu slot=%zu port=%d "
                 "next_submit=%llu next_complete=%llu\n",
                 static_cast<long long>(us), event,
                 static_cast<unsigned long long>(sequence), slot_index, port,
                 static_cast<unsigned long long>(next_submit),
                 static_cast<unsigned long long>(next_complete));
  }

  void trace_dma(const char *event, uint64_t sequence, size_t slot_index,
                 int port, const std::string &status) const {
    if (!trace_dma_enabled())
      return;
    std::fprintf(stderr,
                 "[nn2fpga dma] event=%s seq=%llu slot=%zu port=%d %s\n",
                 event, static_cast<unsigned long long>(sequence), slot_index,
                 port, status.c_str());
  }

  void trace_bo(const char *event, size_t slot_index, size_t port,
                uint64_t address, size_t bytes) const {
    if (!trace_enabled())
      return;
    std::fprintf(stderr,
                 "[nn2fpga trace] event=%s slot=%zu port=%zu addr=0x%016llx bytes=%zu\n",
                 event, slot_index, port,
                 static_cast<unsigned long long>(address), bytes);
  }

  void trace_ctrl_regs(const char *event, uint64_t sequence,
                       size_t slot_index) const {
    if (!trace_ctrl_enabled())
      return;
    std::fprintf(stderr,
                 "[nn2fpga ctrl] event=%s seq=%llu slot=%zu base=0x%lx\n",
                 event, static_cast<unsigned long long>(sequence), slot_index,
                 static_cast<unsigned long>(Spec::ControlAxiOffset));
    for (off_t off = 0; off < 0x40; off += 4) {
      std::fprintf(stderr, "[nn2fpga ctrl] off=0x%02lx value=0x%08x\n",
                   static_cast<unsigned long>(off),
                   rd32(mmio_.regs, Spec::ControlAxiOffset + off));
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

      std::memcpy(static_in_host_ptrs_[i], raw.data(), raw.size());
      static_in_bos_[i]->sync(XCL_BO_SYNC_BO_TO_DEVICE, size, 0);

      static_tx_[i]->transfer(size, 0);
      if (!static_tx_[i]->wait_done(400)) {
        throw std::runtime_error("MM2S timeout during static upload on port " +
                                 std::to_string(i) + " with name " +
                                 entry.at("new_name").get<std::string>());
      }
    }
  }

  xrt::device dev_;
  Mmio mmio_;

  std::vector<std::optional<xrt::bo>> static_in_bos_;
  std::vector<xrt::bo> buffer_bos_;
  std::vector<void *> static_in_host_ptrs_;
  std::vector<void *> buffer_host_ptrs_;

  std::vector<std::optional<Mm2sSimple>> static_tx_;
  std::vector<std::unique_ptr<AxiDmaSgRing>> input_rings_;
  std::vector<std::unique_ptr<AxiDmaSgRing>> output_rings_;
  std::vector<RequestSlot> slots_;

  std::mutex slot_mtx_;
  std::condition_variable slot_cv_;
  std::atomic<uint64_t> next_sequence_{0};

  std::mutex submit_mtx_;
  std::condition_variable submit_cv_;
  std::atomic<uint64_t> next_submit_sequence_{0};

  std::mutex completion_mtx_;
  std::condition_variable completion_cv_;
  std::atomic<uint64_t> next_complete_sequence_{0};

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
