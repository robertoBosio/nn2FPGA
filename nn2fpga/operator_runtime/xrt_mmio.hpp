#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

// ---------- helpers ----------
static uint64_t parse_u64_hex(const std::string &s) {
  uint64_t v = 0;
  std::stringstream ss(s);
  ss >> std::hex >> v;
  return v;
}
static std::string hexstr(uint64_t v) {
  std::ostringstream oss;
  oss << std::showbase << std::hex << v;
  return oss.str();
}
static uint64_t read_hex_file(const fs::path &p) {
  std::ifstream f(p);
  std::string s;
  f >> s;
  return std::stoull(s, nullptr, 16);
}

struct Mmio {
  volatile uint32_t *regs = nullptr;
  size_t map_size = 0;
  int fd = -1;

  Mmio() = default;

  Mmio(Mmio&& other) noexcept
      : regs(other.regs), map_size(other.map_size), fd(other.fd) {
    other.regs = nullptr;
    other.map_size = 0;
    other.fd = -1;
  }

  Mmio& operator=(Mmio&& other) noexcept {
    if (this != &other) {
      if (regs && map_size) munmap((void*)regs, map_size);
      if (fd >= 0) close(fd);
      regs = other.regs;
      map_size = other.map_size;
      fd = other.fd;
      other.regs = nullptr;
      other.map_size = 0;
      other.fd = -1;
    }
    return *this;
  }

  Mmio(const Mmio&) = delete;
  Mmio& operator=(const Mmio&) = delete;

  ~Mmio() {
    if (regs && map_size) munmap((void*)regs, map_size);
    if (fd >= 0) close(fd);
  }
};


static inline volatile uint32_t *do_mmap(int fd, size_t size, off_t offset = 0,
                                         int prot = PROT_READ | PROT_WRITE) {
  void *v = ::mmap(nullptr, size, prot, MAP_SHARED, fd, offset);
  if (v == MAP_FAILED)
    throw std::runtime_error("mmap failed");
  return static_cast<volatile uint32_t *>(v);
}

struct UioMapDesc {
  std::string dev;
  int map = 0;
  size_t size = 0;
};

static std::optional<UioMapDesc> find_uio_map_for_base(uint64_t axil_base) {
  if (!fs::exists("/sys/class/uio"))
    return std::nullopt;
  for (const auto &uio : fs::directory_iterator("/sys/class/uio")) {
    auto maps = uio.path() / "maps";
    for (int i = 0;; ++i) {
      auto m = maps / ("map" + std::to_string(i));
      if (!fs::exists(m))
        break;
      auto addrf = m / "addr";
      auto sizef = m / "size";
      if (!fs::exists(addrf) || !fs::exists(sizef))
        continue;
      if (read_hex_file(addrf) == axil_base) {
        UioMapDesc d;
        d.dev = std::string("/dev/") + uio.path().filename().string();
        d.map = i;
        d.size = static_cast<size_t>(read_hex_file(sizef));
        return d;
      }
    }
  }
  return std::nullopt;
}

// Map the single AXI-Lite window via UIO.
static Mmio map_axil_window(uint64_t axil_base, size_t axil_size) {
  Mmio m;
  if (auto d = find_uio_map_for_base(axil_base)) {
    if (d->size < axil_size)
      throw std::runtime_error("UIO map too small: have " + hexstr(d->size) +
                               " need " + hexstr(axil_size));
    int fd = ::open(d->dev.c_str(), O_RDWR | O_SYNC);
    if (fd < 0)
      throw std::runtime_error("open " + d->dev + " failed");
    long ps = sysconf(_SC_PAGESIZE);
    off_t off = static_cast<off_t>(d->map) *
                static_cast<off_t>(ps); // UIO map offset rule
    m.regs = do_mmap(fd, d->size, off);
    m.map_size = d->size;
    m.fd = fd;
    return m;
  }
  throw std::runtime_error("AXI-Lite base " + hexstr(axil_base) +
                           " not found in UIO maps");
}