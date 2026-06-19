#pragma once
#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <unordered_map>

template <typename TWord, size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH>
class TransposeRowCol {
public:
  static_assert(IN_HEIGHT > 0 && IN_WIDTH > 0,
                "IN_HEIGHT and IN_WIDTH must be greater than 0");
  TransposeRowCol() = default;

  template <size_t HLS_TAG>
  void run(hls::stream<TWord> i_data[1], hls::stream<TWord> o_data[1]) {
    TWord buffer[IN_HEIGHT * IN_WIDTH * IN_CH];
#pragma HLS array_partition variable = buffer complete dim = 2
    for (size_t i_step = 0; i_step < 2; i_step++) {
    TRANSPOSE_RUN_LOOP:
      for (size_t index = 0; index < IN_HEIGHT * IN_WIDTH * IN_CH; index++) {
#pragma HLS PIPELINE II = 1
        TransposeRowCol::pipeline_body(i_data, o_data, buffer, i_step, index);
      }
    }
  }

  struct StepState {
    // Loop iteration indexes.
    size_t i_step = 0, index = 0;
    TWord buffer[IN_HEIGHT * IN_WIDTH * IN_CH];

    PipelineDelayBuffer<TWord> delayed_output;
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      delayed_output = PipelineDelayBuffer<TWord>(depth);
      actor_status = ActorStatus(depth, IN_HEIGHT * IN_WIDTH * IN_CH * 2);
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  ActorStatus step(hls::stream<TWord> i_data[1], hls::stream<TWord> o_data[1]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    if (st.i_step == 0) {
      if (i_data[0].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      // If there is data in the input stream, process it.
      hls::stream<TWord> instant_output_stream[1];
      TransposeRowCol::pipeline_body(i_data, instant_output_stream, st.buffer,
                                     st.i_step, st.index);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      if (!instant_output_stream[0].empty()) {
        st.delayed_output.push(instant_output_stream[0].read(), true);
      } else {
        // If the output stream is empty, push a placeholder.
        st.delayed_output.push(TWord(), false);
      }

      // Update the counters.
      st.index++;
      if (st.index >= IN_HEIGHT * IN_WIDTH * IN_CH) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        st.index = 0;
        st.i_step++;
      }
      if (st.i_step >= 2) {
        st.i_step = 0;
      }

    } else {
      // If there is no data in the input stream, push a delay slot.
      st.delayed_output.push(TWord(), false);
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TWord out;
    if (st.delayed_output.pop(out)) {
      o_data[0].write(out);
    }

    // Return the actor status.
    return st.actor_status;
  }

private:
  static void pipeline_body(hls::stream<TWord> i_data[1],
                            hls::stream<TWord> o_data[1],
                            TWord buffer[IN_HEIGHT * IN_WIDTH * IN_CH],
                            size_t i_step, size_t index) {
#pragma HLS inline
    if (i_step == 0) {
      // Read input data into the buffer.
      TWord in_word = i_data[0].read();

      size_t h = index % IN_HEIGHT;
      size_t ch = (index / IN_HEIGHT) % IN_CH;
      size_t w = (index / (IN_HEIGHT * IN_CH)) % IN_WIDTH;
      size_t buffer_index = ((h * IN_CH + ch) * IN_WIDTH + w);
      buffer[buffer_index] = in_word;
    } else {
      // Write transposed data from the buffer to the output stream.
      o_data[0].write(buffer[index]);
    }
  }
};

template <typename TInputWord, typename TOutputWord, size_t IN_HEIGHT,
          size_t IN_WIDTH, size_t REPLAY_COUNT, size_t REDUCE_PAR>
class TransposeRowColReplayPacked {
public:
  static_assert(IN_HEIGHT > 0 && IN_WIDTH > 0,
                "IN_HEIGHT and IN_WIDTH must be greater than 0");
  static_assert(REDUCE_PAR > 0, "REDUCE_PAR must be greater than 0");
  static_assert(IN_WIDTH % REDUCE_PAR == 0,
                "IN_WIDTH must be a multiple of REDUCE_PAR");

  TransposeRowColReplayPacked() = default;

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[REDUCE_PAR],
           hls::stream<TOutputWord> o_data[1]) {
    TOutputWord buffer[IN_HEIGHT * (IN_WIDTH / REDUCE_PAR)];

    for (size_t i_pass = 0; i_pass < REPLAY_COUNT + 1; i_pass++) {
      for (size_t i_word = 0; i_word < IN_HEIGHT * (IN_WIDTH / REDUCE_PAR);
           i_word++) {
#pragma HLS PIPELINE II = 1
          pipeline_body(i_data, o_data, buffer, i_pass, i_word);
      }
    }
  }

  struct StepState {
    size_t i_pass = 0;
    size_t i_word = 0;
    TOutputWord buffer[IN_HEIGHT * (IN_WIDTH / REDUCE_PAR)];

    PipelineDelayBuffer<TOutputWord> delayed_output;
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      delayed_output = PipelineDelayBuffer<TOutputWord>(depth);
      actor_status = ActorStatus(depth,
                                 (REPLAY_COUNT + 1) * IN_HEIGHT *
                                     (IN_WIDTH / REDUCE_PAR));
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  ActorStatus step(hls::stream<TInputWord> i_data[REDUCE_PAR],
                   hls::stream<TOutputWord> o_data[1]) {
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    bool firing_condition = true;
    if (st.i_pass == 0) {
      for (size_t i_reduce = 0; i_reduce < REDUCE_PAR; i_reduce++) {
        if (i_data[i_reduce].empty()) {
          firing_condition = false;
        }
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> instant_output_stream[1];
      pipeline_body(i_data, instant_output_stream, st.buffer, st.i_pass,
                    st.i_word);

      st.actor_status.fire();

      if (!instant_output_stream[0].empty()) {
        st.delayed_output.push(instant_output_stream[0].read(), true);
      } else {
        st.delayed_output.push(TOutputWord(), false);
      }

      st.i_word++;
      if (st.i_word >= IN_HEIGHT * (IN_WIDTH / REDUCE_PAR)) {
        st.i_word = 0;
        st.i_pass++;
      }
      if (st.i_pass > REPLAY_COUNT) {
        st.i_pass = 0;
      }
    } else {
      st.delayed_output.push(TOutputWord(), false);
    }

    st.actor_status.advance();

    TOutputWord out;
    if (st.delayed_output.pop(out)) {
      o_data[0].write(out);
    }

    return st.actor_status;
  }

private:
  static void pipeline_body(hls::stream<TInputWord> i_data[REDUCE_PAR],
                            hls::stream<TOutputWord> o_data[1],
                            TOutputWord buffer[IN_HEIGHT * (IN_WIDTH / REDUCE_PAR)],
                            size_t i_pass, size_t i_word) {
#pragma HLS inline
    if (i_pass == 0) {
      const size_t i_seq_group = i_word / IN_HEIGHT;
      const size_t i_vrow = i_word % IN_HEIGHT;
      const size_t buffer_index = i_vrow * (IN_WIDTH / REDUCE_PAR) + i_seq_group;
      TOutputWord out_word;
      for (size_t i_reduce = 0; i_reduce < REDUCE_PAR; i_reduce++) {
#pragma HLS unroll
        out_word[i_reduce] = i_data[i_reduce].read()[0];
      }
      buffer[buffer_index] = out_word;
    } else {
      const size_t i_vrow = i_word / (IN_WIDTH / REDUCE_PAR);
      const size_t i_seq_group = i_word % (IN_WIDTH / REDUCE_PAR);
      const size_t buffer_index = i_vrow * (IN_WIDTH / REDUCE_PAR) + i_seq_group;
      o_data[0].write(buffer[buffer_index]);
    }
  }
};
