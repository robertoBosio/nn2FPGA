#pragma once
#include <vector>
#include <set>
#include <tuple>
#include <unordered_map>
#include <queue>
#include <fstream>
#include <iostream>

class FiringStatus {
public:
  FiringStatus(size_t remaining_time, size_t firing_index)
      : remaining_time(remaining_time), firing_index(firing_index) {}

  bool operator<(const FiringStatus &other) const {
    return std::tie(remaining_time, firing_index) <
           std::tie(other.remaining_time, other.firing_index);
  }

  bool operator==(const FiringStatus &other) const {
    return remaining_time == other.remaining_time &&
           firing_index == other.firing_index;
  }

  bool operator--() {
    --remaining_time;
    return remaining_time == 0;
  }

  size_t get_remaining_time() const { return remaining_time; }
  size_t get_firing_index() const { return firing_index; }

private:
  size_t remaining_time; // Remaining time for the firing to complete.
  size_t firing_index;   // Index of the firing within the actor.
};

#ifndef __SYNTHESIS__
class ActorStatus {
public:
  ActorStatus() : ActorStatus(1, 1) {}

  ActorStatus(size_t t, size_t N) : firings(), current_index(0), t(t), N(N) {
    // Initialize the actor status with the given execution time and number of
    // firings.
  }

  void fire() {
    // Add a new firing to the actor's status.
    firings.insert(FiringStatus(t, current_index));
    current_index = (current_index + 1) % N;
  }

  void advance() {
    // Advance the firings of the actor.
    if (!firings.empty()) {
      std::vector<FiringStatus> updated_firings;
      for (const auto &firing : firings) {
        FiringStatus updated = firing;
        --updated;
        updated_firings.push_back(updated);
      }
      firings.clear();
      for (const auto &firing : updated_firings) {
        if (firing.get_remaining_time() > 0) {
          firings.insert(firing);
        }
      }
    }
  }

  bool empty() const { return firings.empty(); }
  size_t size() const { return firings.size(); }

  bool operator==(const ActorStatus &other) const {
    return firings == other.firings && current_index == other.current_index;
  }

  bool operator!=(const ActorStatus &other) const {
    return !(*this == other);
  }
  // Define a string representation for debugging
  std::string to_string() const {
    std::string result = "ActorStatus:\n";
    result += "Current Index: " + std::to_string(current_index) + "\n";
    result += "Firings: ";
    for (const auto &firing : firings) {
      result += "(" + std::to_string(firing.get_remaining_time()) + ", " +
                std::to_string(firing.get_firing_index()) + ") ";
    }
    return result;
  }

  std::multiset<FiringStatus> get_firings() const { return firings; }
  size_t get_current_index() const { return current_index; }

private:
  std::multiset<FiringStatus> firings; // Set of current firings for the actor.
  size_t current_index; // Current index in the actor execution sequence.
  size_t t;             // Execution time of the firing.
  size_t N;             // Number of firings in the execution sequence.
};
#else
class ActorStatus {
public:
  ActorStatus() {}
  ActorStatus(size_t t, size_t N) {
    // Utilize t and N to remove the warning about unused parameters.
    (void)t;
    (void)N;
  }

  void fire() {}
  void advance() {}
  bool empty() const { return true; }
  size_t size() const { return 0; }
  bool operator==(const ActorStatus &other) const {
    (void)other; // Avoid unused parameter warning
    return true;
  }
  std::string to_string() const { return ""; }
  std::multiset<FiringStatus> get_firings() const { return {}; }
  size_t get_current_index() const { return 0; }
};
#endif // __SYNTHESIS__

class CSDFGState {
public:
  CSDFGState() : tokens(), actor_statuses() {}

  CSDFGState(const std::vector<ActorStatus> &actor_statuses,
             const std::vector<size_t> &tokens)
      : tokens(tokens), actor_statuses(actor_statuses) {}

  bool operator==(const CSDFGState &other) const {
    return tokens == other.tokens && actor_statuses == other.actor_statuses;
  }

  std::vector<size_t> get_tokens() const { return tokens; }

  std::vector<ActorStatus> get_actor_statuses() const {
    return actor_statuses;
  }

  void clear() {
    tokens.clear();
    actor_statuses.clear();
  }

  void set_tokens(const std::vector<size_t> &new_tokens) {
    tokens = new_tokens;
  }

  void set_actor_statuses(const std::vector<ActorStatus> &new_actor_statuses) {
    actor_statuses = new_actor_statuses;
  }

  // Define a string representation for debugging
  std::string to_string() const {
    std::string result = "CSDFGState:\n";
    result += "Channel quantity: ";
    for (const auto &token : tokens) {
      result += std::to_string(token) + " ";
    }
    result += "Actor Statuses: \n";
    for (const auto &status : actor_statuses) {
      result += status.to_string() + "\n";
    }
    return result;
  }

private:
  std::vector<size_t> tokens; // Channel quantity. Associate with each channel the
                           // amount of tokens it has.
  std::vector<ActorStatus>
      actor_statuses; // Each actor has a
                      // vector of ActorStatus, which contains the
                      // current firings and the current index.
};

struct CSDFGStateHasher {
  std::size_t operator()(const CSDFGState &s) const {
    std::size_t h = 0;
    for (int t : s.get_tokens()) {
      h ^= std::hash<int>()(t) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    for (const auto &actor : s.get_actor_statuses()) {
      for (const auto &firing : actor.get_firings()) {
        h ^= std::hash<int>()(firing.get_remaining_time()) + 0x9e3779b9 + (h << 6) +
             (h >> 2);
        h ^= std::hash<int>()(firing.get_firing_index()) + 0x9e3779b9 + (h << 6) +
             (h >> 2);
      }
      h ^= std::hash<int>()(actor.get_current_index()) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
};

#ifndef __SYNTHESIS__
struct CompactState {
    std::vector<uint32_t> data;
    bool operator==(const CompactState &o) const { return data == o.data; }
};

// struct CompactHasher {
//     size_t operator()(const CompactState &s) const {
//         size_t h = 0;
//         for (auto v : s.data)
//             h ^= std::hash<uint32_t>()(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
//         return h;
//     }
// };

using StateSig = std::uint64_t;

struct StateRef {
    std::uint64_t offset;  // byte offset in the file
    std::uint32_t clock;   // enough up to 4e9 cycles
};

std::unordered_map<StateSig, std::vector<StateRef>> visited;
std::ofstream states_out("states.bin", std::ios::binary | std::ios::app);
std::ifstream states_in("states.bin", std::ios::binary);

uint64_t append_state_to_file(const std::vector<uint32_t> &data) {
    states_out.seekp(0, std::ios::end);
    uint64_t offset = static_cast<uint64_t>(states_out.tellp());

    uint32_t len = static_cast<uint32_t>(data.size());
    states_out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    states_out.write(reinterpret_cast<const char*>(data.data()),
                     len * sizeof(uint32_t));
    // no flush per write; let OS buffer

    return offset;
}

bool states_equal_on_disk(uint64_t offset,
                          const std::vector<uint32_t> &data) {
    static thread_local std::vector<uint32_t> buf;

    states_in.seekg(offset, std::ios::beg);

    uint32_t len = 0;
    states_in.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (len != data.size()) return false;

    buf.resize(len);
    states_in.read(reinterpret_cast<char*>(buf.data()),
                   len * sizeof(uint32_t));

    return std::equal(buf.begin(), buf.end(), data.begin());
}


StateSig make_signature(const CompactState &c) {
    // Example: 64-bit splitmix-style mixer
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (auto v : c.data) {
        uint64_t x = v;
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

CompactState make_compact_state(const CSDFGState &s) {
    CompactState c;
    const auto &tokens = s.get_tokens();
    const auto &actors = s.get_actor_statuses();
    c.data.reserve(tokens.size() + 8 * actors.size()); // heuristic
    for (auto t : tokens)
        c.data.push_back(static_cast<uint32_t>(t));
    c.data.push_back(0xFFFFFFFF); // separator

    for (const auto &a : actors) {
        c.data.push_back(static_cast<uint32_t>(a.get_current_index()));
        for (const auto &f : a.get_firings()) {
            c.data.push_back(static_cast<uint32_t>(f.get_remaining_time()));
            c.data.push_back(static_cast<uint32_t>(f.get_firing_index()));
        }
        c.data.push_back(0xFFFFFFFF); // actor separator
    }
    return c;
}

template <typename T> class PipelineDelayBuffer {
public:
  PipelineDelayBuffer(size_t depth) : pipeline_depth(depth) {
    for (size_t i = 0; i < pipeline_depth - 1; ++i) {
      valid_flags.push(false);
    }
  }

  PipelineDelayBuffer() : pipeline_depth(1) {}

  // Push a new output element into the pipeline (valid == true) or a delay slot
  // (valid == false)
  void push(const T &value, bool valid) {
    if (valid) {
      data_queue.push(value);
    }
    valid_flags.push(valid);
  }

  // Step the pipeline forward, and return whether the front is valid and its
  // value
  bool pop(T &out_value) {
    bool valid = valid_flags.front();
    valid_flags.pop();

    if (valid) {
      out_value = data_queue.front();
      data_queue.pop();
    }
    return valid;
  }

  bool peek() const {
    if (valid_flags.empty()) {
      return false;
    }
    return valid_flags.front();
  }

  std::string to_string() const {
    std::string result = "PipelineDelayBuffer:\n";
    result += "Depth: " + std::to_string(pipeline_depth) + "\n";
    result += "Data Queue Size: " + std::to_string(data_queue.size()) + "\n";
    result += "Valid Flags: ";
    std::queue<bool> temp_flags = valid_flags;
    while (!temp_flags.empty()) {
      result += (temp_flags.front() ? "1 " : "0 ");
      temp_flags.pop();
    }
    return result;
  }

private:
  size_t pipeline_depth;        // Depth of the pipeline
  std::queue<T> data_queue;     // Queue to hold the data elements
  std::queue<bool> valid_flags; // Queue to hold the valid flags
};
#else
template <typename T> class PipelineDelayBuffer {
public:
  PipelineDelayBuffer() {}
  PipelineDelayBuffer(size_t depth) {
    (void)depth; // Avoid unused parameter warning
  }
  void push(const T &value, bool valid) {
    (void)value; // Avoid unused parameter warning
    (void)valid; // Avoid unused parameter warning
  }

  bool pop(T &out_value) {
    (void)out_value; // Avoid unused parameter warning
    return false;    // Always return false in synthesis mode
  }

  bool peek() const { return false; }

  std::string to_string() const { return ""; }
};
#endif // __SYNTHESIS__
