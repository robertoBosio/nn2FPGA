// Minimal PL clock setter for Zynq UltraScale+ MPSoC (Kria).
// Forces a PL clock (PL0..PL3) to use the IOPLL as source and sets integer
// divisors (DIV0, DIV1) to approximate a target frequency. Safe sequence:
//   1) Gate the PL clock output (CLKACT=0)
//   2) Program SRCSEL=IOPLL and DIVISOR0/1
//   3) Ungate the clock (CLKACT=1)
// This touches the PS clock controller registers at CRL_APB; it does NOT
// reprogram any PLL.
//
// Notes:
// - On ZynqMP, CRL_APB manages low-power domain clocks including PLx_REF_CTRL.
// - Field layout is from PYNQ’s ZU_* field tables you shared.
// - We assume SRCSEL=0 selects IOPLL (common on ZynqMP platforms).
// - DIV0/DIV1 are integer 1..63; exact 300 MHz from 1.0 GHz is not possible
//   with integers only; 250, 200, 125, etc. are exact.

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

// ---------------- PS clock controller register map (CRL_APB) ----------------
static constexpr off_t CRL_APB_BASE = 0xFF5E0000; // base phys addr
static constexpr off_t PL_REF_CTRL[4] = {0xC0, 0xC4, 0xC8,
                                         0xCC}; // PL0..PL3 reg

// Bitfields for PLx_REF_CTRL (ZU_CLK_FIELDS)
static constexpr uint32_t SRCSEL_SHIFT = 0; // bits [2:0]
static constexpr uint32_t SRCSEL_MASK = 0x7u << SRCSEL_SHIFT;

static constexpr uint32_t DIV0_SHIFT = 8; // bits [13:8]
static constexpr uint32_t DIV0_MASK = 0x3Fu << DIV0_SHIFT;

static constexpr uint32_t DIV1_SHIFT = 16; // bits [21:16]
static constexpr uint32_t DIV1_MASK = 0x3Fu << DIV1_SHIFT;

static constexpr uint32_t CLKACT_BIT = 24; // bit 24
static constexpr uint32_t CLKACT_MASK = 1u << CLKACT_BIT;

// Platform-conventional mapping: 0 = IOPLL for SRCSEL
static constexpr uint32_t SRCSEL_IOPLL = 0;

enum class ZynqPllIndex {
  PL0 = 0,
  PL1 = 1,
  PL2 = 2,
  PL3 = 3
};

// -------------- Lightweight helpers (mapping and barriers) ------------------

static inline void dsb() {
  asm volatile("dsb sy" ::: "memory");
} // Data Sync Barrier
static inline void isb() {
  asm volatile("isb" ::: "memory");
} // Instr Sync Barrier

struct Mmap {
  volatile uint32_t *p = nullptr;
  size_t sz = 0;
  int fd = -1;
  ~Mmap() {
    if (p)
      munmap((void *)p, sz);
    if (fd >= 0)
      close(fd);
  }
};

// Map a physical MMIO block via /dev/mem
static Mmap map_block(off_t phys, size_t size) {
  const long pg = sysconf(_SC_PAGESIZE);
  const off_t base = phys & ~(off_t(pg - 1)); // page-align base
  const off_t delta = phys - base;            // offset within page
  const size_t mapsz = ((size + delta + pg - 1) / pg) * pg; // whole pages

  int fd = ::open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0)
    throw std::runtime_error(std::string("open /dev/mem: ") + strerror(errno));

  void *v =
      ::mmap(nullptr, mapsz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
  if (v == MAP_FAILED) {
    int e = errno;
    close(fd);
    throw std::runtime_error(std::string("mmap: ") + strerror(e));
  }

  Mmap m;
  m.p = reinterpret_cast<volatile uint32_t *>((uint8_t *)v + delta);
  m.sz = mapsz;
  m.fd = fd;
  return m;
}

// Choose integer divisors (1..63) so src_hz/(DIV0*DIV1) ≈ target_hz.
// Preference is for DIV1=1 (lower jitter), but we try a small set of DIV1s.
static void choose_div(uint64_t src_hz, uint64_t tgt_hz, uint32_t &d0,
                       uint32_t &d1, double &achieved_hz) {
  double best_err = 1e300;
  uint32_t bd0 = 1, bd1 = 1;
  double bf = double(src_hz);

  const uint32_t d1candidates[] = {1,  2,  3,  4,  5,  6,  8,  10,
                                   12, 16, 20, 24, 32, 40, 48, 63};

  for (uint32_t a = 1; a <= 63; ++a) {
    for (uint32_t b : d1candidates) {
      if (b > 63)
        continue;
      const double f = double(src_hz) / double(a * b);
      const double err = std::abs(f - double(tgt_hz));
      if (err < best_err && f <= double(tgt_hz)) {
        best_err = err;
        bd0 = a;
        bd1 = b;
        bf = f;
      }
      if (err == 0.0)
        break; // exact match found
    }
  }
  d0 = bd0;
  d1 = bd1;
  achieved_hz = bf;
}

// -------------------------- Core routine (IOPLL only) -----------------------

static float set_pl_from_iopll(ZynqPllIndex pl_index, double target_MHz,
                               double iopll_MHz) {
  if (target_MHz <= 0)
    throw std::runtime_error("target_MHz must be > 0");
  if (iopll_MHz <= 0)
    throw std::runtime_error("iopll_MHz must be > 0");

  const uint64_t src_hz = (uint64_t)llround(iopll_MHz * 1e6); // IOPLL rate
  const uint64_t tgt_hz =
      (uint64_t)llround(target_MHz * 1e6); // requested PL rate

  // Pick integer divisors to approximate target frequency
  uint32_t d0 = 1, d1 = 1;
  double ach = double(src_hz);
  choose_div(src_hz, tgt_hz, d0, d1, ach);

  // Map CRL_APB and locate PLx_REF_CTRL register
  auto crl = map_block(CRL_APB_BASE, 0x1000);
  volatile uint32_t *reg = reinterpret_cast<volatile uint32_t *>(
      (uintptr_t)crl.p + PL_REF_CTRL[static_cast<int>(pl_index)]);

  // Read current value
  uint32_t v = *reg;

  // 1) Gate the clock output: clear CLKACT so downstream logic isn't fed
  // mid-change
  v &= ~CLKACT_MASK;
  *reg = v;
  dsb(); // ensure the write reaches hardware before we proceed

  // 2) Force source to IOPLL and program integer divisors
  v &= ~(SRCSEL_MASK | DIV0_MASK | DIV1_MASK);
  v |= (SRCSEL_IOPLL << SRCSEL_SHIFT);
  v |= ((d0 << DIV0_SHIFT) & DIV0_MASK);
  v |= ((d1 << DIV1_SHIFT) & DIV1_MASK);
  *reg = v;
  dsb(); // make sure the new setting is committed before ungating

  // 3) Ungate the clock output: set CLKACT to enable the PL clock output
  v |= CLKACT_MASK;
  *reg = v;
  dsb(); // ensure data write completes
  isb(); // synchronize subsequent instruction stream (belt & suspenders)

  return ach / 1e6f; // return achieved frequency in MHz
}