#pragma once
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <optional>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>

// ---------- AXI DMA regs (PG021) ----------
static constexpr off_t MM2S_DMACR = 0x00;
static constexpr off_t MM2S_DMASR = 0x04;
static constexpr off_t MM2S_SA = 0x18;
static constexpr off_t MM2S_SA_MSB = 0x1C;
static constexpr off_t MM2S_LEN = 0x28;

static constexpr off_t S2MM_DMACR = 0x30;
static constexpr off_t S2MM_DMASR = 0x34;
static constexpr off_t S2MM_CURDESC = 0x38;
static constexpr off_t S2MM_CURDESC_MSB = 0x3C;
static constexpr off_t S2MM_TAILDESC = 0x40;
static constexpr off_t S2MM_TAILDESC_MSB = 0x44;
static constexpr off_t S2MM_DA = 0x48;
static constexpr off_t S2MM_DA_MSB = 0x4C;
static constexpr off_t S2MM_LEN = 0x58;

static constexpr uint32_t DMACR_RS = (1u << 0);
static constexpr uint32_t DMACR_RESET = (1u << 2);
static constexpr uint32_t DMASR_HALTED = (1u << 0);
static constexpr uint32_t DMASR_IDLE = (1u << 1);
static constexpr uint32_t DMASR_IOCIRQ = (1u << 12);
static constexpr uint32_t DMASR_DLYIRQ = (1u << 13);
static constexpr uint32_t DMASR_ERRIRQ = (1u << 14);

// Error causes (internal/slave/decode).
static constexpr uint32_t DMASR_DMA_INTERR = (1u << 4);
static constexpr uint32_t DMASR_DMA_SLVERR = (1u << 5);
static constexpr uint32_t DMASR_DMA_DECERR = (1u << 6);
static constexpr uint32_t DMASR_ERROR_MASK =
    DMASR_ERRIRQ | DMASR_DMA_INTERR | DMASR_DMA_SLVERR | DMASR_DMA_DECERR;
  
static constexpr uint32_t DMASR_SG_INTERR = (1u << 8);
static constexpr uint32_t DMASR_SG_SLVERR = (1u << 9);
static constexpr uint32_t DMASR_SG_DECERR = (1u << 10);
static constexpr uint32_t DMASR_SG_ERROR_MASK =
    DMASR_SG_INTERR | DMASR_SG_SLVERR | DMASR_SG_DECERR;

static inline void wr32(volatile uint32_t *b, off_t off, uint32_t v) {
  b[off / 4] = v;
}

static inline uint32_t rd32(volatile uint32_t *b, off_t off) {
  return b[off / 4];
}

static inline void sleep_us(int us) {
  std::this_thread::sleep_for(std::chrono::microseconds(us));
}

// ---------- 64B Descriptor ----------
struct __attribute__((packed, aligned(64))) AxiDmaBd {
  uint32_t next_l, next_h; // [0..1]
  uint32_t buf_l, buf_h;   // [2..3]
  uint32_t rsv4, rsv5;     // [4..5]
  uint32_t ctrl_len;       // [6]  LEN[25:0] 
  uint32_t status;         // [7]  HW writes completion/EOP etc.
  uint32_t app0, app1, app2, app3, app4;
  uint32_t rsv13, rsv14, rsv15;
};

class Mm2sSimple {
public:
  Mm2sSimple(volatile uint32_t *regs_base, off_t mm2s_off, const xrt::bo &src)
      : regs_(regs_base + (mm2s_off / 4)), src_(src) {
    reset_and_halt_();
  }

  // default constructor
  Mm2sSimple() : regs_(nullptr), src_() {}

  /**
   * @brief Initiates a memory-to-stream (MM2S) DMA transfer.
   *
   * This function configures and starts a DMA transfer from the source buffer
   * to the stream interface. It performs bounds checking, clears the DMA status
   * register, sets the source address, enables the DMA, and writes the transfer
   * length to start the operation.
   *
   * @param nbytes Number of bytes to transfer. Must be greater than 0.
   * @param start Optional starting offset within the source buffer. Defaults to
   * 0.
   *
   * @throws std::runtime_error if nbytes is zero or if the transfer range is
   * out of bounds.
   */
  void transfer(size_t nbytes, size_t start = 0) {
    if (!nbytes)
      throw std::runtime_error("MM2S transfer length must be greater than 0.");
    if (start + nbytes > src_.size())
      throw std::runtime_error("MM2S transfer range is out of bounds.");

    // Clear status register.
    reset_and_halt_();

    // Program source address.
    uint64_t sa = src_.address() + start;
    wr32(regs_, MM2S_SA, (uint32_t)(sa & 0xFFFFFFFFull));
    wr32(regs_, MM2S_SA_MSB, (uint32_t)(sa >> 32));

    // Set the DMA in run state.
    wr32(regs_, MM2S_DMACR,
         (rd32(regs_, MM2S_DMACR) & ~DMACR_RESET) | DMACR_RS);

    // Write the length to the MM2S_LEN register, which starts the transfer.
    wr32(regs_, MM2S_LEN, (uint32_t)nbytes);
  }

  bool wait_done(int timeout_ms) {
    using namespace std::chrono;
    auto deadline = steady_clock::now() + milliseconds(timeout_ms);

    for (;;) {
      uint32_t sr = rd32(regs_, MM2S_DMASR);

      if (sr & DMASR_ERROR_MASK) {
        throw std::runtime_error("MM2S DMA error. DMASR: 0x" + std::to_string(sr));
      }
      if ((sr & DMASR_IDLE) || (sr & DMASR_IOCIRQ)) {
        break;
      }
      if (steady_clock::now() >= deadline) {
        return false;
      }
      sleep_us(5);
    }
    return true;
  }

private:
  void reset_and_halt_() {
    wr32(regs_, MM2S_DMASR, 0xFFFFFFFFu);
    wr32(regs_, MM2S_DMACR, DMACR_RESET);
    sleep_us(5);
    wr32(regs_, MM2S_DMACR, 0);
    wr32(regs_, MM2S_DMASR, 0xFFFFFFFFu);
  }

  volatile uint32_t *regs_;
  xrt::bo src_;
};

class S2mmSG {
public:
  S2mmSG(volatile uint32_t *regs_base, off_t s2mm_off, const xrt::device &dev,
         const xrt::bo &dst, size_t bytes_per_descriptor, size_t batch_max,
         unsigned bank = 0)
      : regs_(regs_base + (s2mm_off / 4)), dev_(dev), dst_(dst),
        bytes_per_descriptor_(bytes_per_descriptor), batch_max_(batch_max),
        bd_bo_(dev_, batch_max_ * sizeof(AxiDmaBd), 0, bank) {
    if (bytes_per_descriptor_ == 0)
      throw std::runtime_error(
          "S2MM bytes_per_descriptor must be greater than 0.");
    if (batch_max_ == 0)
      throw std::runtime_error("S2MM batch_max must be greater than 0.");

    build_descriptors();
  }

  // Dummy default constructor
  S2mmSG()
      : regs_(nullptr), dev_(), dst_(), bytes_per_descriptor_(0), batch_max_(0),
        bd_bo_() {}

  /**
   * @brief Initiates a DMA transfer for a specified batch of buffer descriptors
   * (BDs).
   *
   * This function prepares and starts a Scatter-Gather DMA (S2MM) transfer
   * using a batch of buffer descriptors. It performs the following steps:
   * - Validates the batch size to ensure it is within allowed limits.
   * - Resets the status of each BD in the batch.
   * - Synchronizes the buffer descriptor memory to the device.
   * - Clears the S2MM DMA status register.
   * - Programs the starting address of the first BD.
   * - Sets the DMA engine to the run state.
   * - Sets the tail descriptor to the last BD in the batch, triggering the
   * transfer.
   *
   * @param batch Number of buffer descriptors to transfer. Must be > 0 and <=
   * batch_max_.
   * @throws std::runtime_error if batch is out of valid range.
   */
  void transfer(size_t batch) {
    if (batch == 0 || batch > batch_max_)
      throw std::runtime_error(
          "S2MM transfer batch must be > 0 and <= batch_max_. Actual: " +
          std::to_string(batch) +
          ", batch_max_: " + std::to_string(batch_max_));

    // Zero status for the BDs in this batch.
    AxiDmaBd *bd_ = bd_bo_.map<AxiDmaBd *>();
    for (size_t i = 0; i < batch; i++)
      bd_[i].status = 0;
    bd_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE, batch * sizeof(AxiDmaBd), 0);

    // Clear the S2MM status register.
    reset_and_halt_();
    
    // Program the starting address of the first BD. 
    uint64_t bd0 = bd_bo_.address();
    wr32(regs_, S2MM_CURDESC, (uint32_t)(bd0 & 0xFFFFFFFFull));
    wr32(regs_, S2MM_CURDESC_MSB, (uint32_t)(bd0 >> 32));
    
    // Set the DMA in run state.
    wr32(regs_, S2MM_DMACR,
         (rd32(regs_, S2MM_DMACR) & ~DMACR_RESET) | DMACR_RS);

    // Set the tail descriptor to the last BD in this batch, which starts the
    // transfer.
    uint64_t bdtail = bd_bo_.address() + (batch - 1) * sizeof(AxiDmaBd);
    wr32(regs_, S2MM_TAILDESC, (uint32_t)(bdtail & 0xFFFFFFFFull));
    wr32(regs_, S2MM_TAILDESC_MSB, (uint32_t)(bdtail >> 32));
  }

  /**
   * @brief Waits for the DMA transfer to complete within a specified timeout.
   *
   * This function checks the completion status of the last buffer descriptor
   * (BD) for a given batch. It repeatedly synchronizes the BD from the device
   * and inspects the completion bit until either the transfer is done or the
   * timeout expires.
   *
   * @param timeout_ms Timeout in milliseconds to wait for completion.
   * @param batch Number of batches to check for completion (must be > 0 and <=
   * batch_max_).
   * @return true if the DMA transfer completed within the timeout, false
   * otherwise.
   * @throws std::runtime_error if batch is out of valid range.
   */
  bool wait_done(int timeout_ms, size_t batch = 1) {
    using namespace std::chrono;
    if (batch == 0 || batch > batch_max_)
      throw std::runtime_error(
          "S2MM wait_done batch must be > 0 and <= batch_max_. Actual: " +
          std::to_string(batch) +
          ", batch_max_: " + std::to_string(batch_max_));

    auto deadline = steady_clock::now() + milliseconds(timeout_ms);
    for (;;) {
      const uint32_t sr = rd32(regs_, S2MM_DMASR);
      if (sr & DMASR_SG_ERROR_MASK) {
        throw std::runtime_error("S2MM SG error. DMASR: 0x" + std::to_string(sr));
      }
      if (sr & DMASR_HALTED) {
        throw std::runtime_error("S2MM halted. DMASR: 0x" + std::to_string(sr));
      }
      if (sr & DMASR_IDLE || sr & DMASR_IOCIRQ) {
        break;
      }
      if (steady_clock::now() >= deadline)
        return false;

      // If not done, wait a bit before checking again.
      sleep_us(5);
    }

    // Single BD sync to confirm completion (read only last BD).
    static constexpr uint32_t Cmplt = (1u << 31);
    AxiDmaBd *bd_ = bd_bo_.map<AxiDmaBd *>();
    volatile AxiDmaBd *last = &bd_[batch - 1];
    // Wait for the completion bit in the last BD to update.
    for (;;) {
      bd_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE, sizeof(AxiDmaBd),
                  (batch - 1) * sizeof(AxiDmaBd));
      uint32_t st = last->status;
      if (st & Cmplt)
        return true; // done
      
      if (steady_clock::now() >= deadline)
        return false;

      // tiny sleep to avoid busy-looping too hard
      sleep_us(1);
    }
  }

private:
  /**
   * @brief Builds AXI DMA buffer descriptors for batched image transfers.
   *
   * This function initializes the buffer descriptor array for SG DMA
   * transfers. Each descriptor is configured to:
   * - Point to the next descriptor in a circular fashion (last points to
   * first).
   * - Reference the destination buffer for a single image.
   * - Set the buffer length.
   * - Clear reserved and application-specific fields.
   */
  void build_descriptors () {
    const uint64_t bds = bd_bo_.address();
    const uint64_t dst_base = dst_.address();
    AxiDmaBd *bd_ = bd_bo_.map<AxiDmaBd *>();
    for (size_t i = 0; i < batch_max_; ++i) {
      
      // Each BD points to the next one. (The last one points to the first.)
      uint64_t next = bds + (((i + 1) % batch_max_) * sizeof(AxiDmaBd));
      bd_[i].next_l = (uint32_t)(next & 0xFFFFFFFFull);
      bd_[i].next_h = (uint32_t)(next >> 32);

      // Each BD writes exactly one image’s bytes.
      uint64_t da = dst_base + i * bytes_per_descriptor_;
      bd_[i].buf_l = (uint32_t)(da & 0xFFFFFFFFull);
      bd_[i].buf_h = (uint32_t)(da >> 32);

      // Set the buffer length.
      // AXI DMA supports up to 26 bits for buffer length (see PG021), so we
      // mask with 0x3FFFFFFu.
      bd_[i].ctrl_len =
          (uint32_t)(bytes_per_descriptor_ & 0x3FFFFFFu);

      // Reserved fields.
      bd_[i].rsv4 = bd_[i].rsv5 = 0;
      bd_[i].app0 = bd_[i].app1 = bd_[i].app2 = bd_[i].app3 = bd_[i].app4 = 0;
      bd_[i].rsv13 = bd_[i].rsv14 = bd_[i].rsv15 = 0;
    }
  }

  void reset_and_halt_() {
    wr32(regs_, S2MM_DMASR, 0xFFFFFFFFu);
    wr32(regs_, S2MM_DMACR, DMACR_RESET);
    sleep_us(5);
    wr32(regs_, S2MM_DMACR, 0);
    wr32(regs_, S2MM_DMASR, 0xFFFFFFFFu);
  }

  volatile uint32_t *regs_;
  xrt::device dev_;
  xrt::bo dst_;
  size_t bytes_per_descriptor_;
  size_t batch_max_;
  xrt::bo bd_bo_;
};
