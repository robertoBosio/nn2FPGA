#pragma once
#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <unordered_map>

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t IN_HEIGHT,
          size_t IN_WIDTH, size_t IN_CH, size_t REDUCE_PAR>
class SplitReshapeQKV {
public:
  static_assert(IN_CH % REDUCE_PAR == 0,
                "IN_CH must be a multiple of REDUCE_PAR");
  static_assert(REDUCE_PAR > 0, "REDUCE_PAR must be greater than 0");
  static_assert(IN_HEIGHT > 0 && IN_WIDTH > 0,
                "IN_HEIGHT and IN_WIDTH must be greater than 0");
  static_assert(
      32 % REDUCE_PAR == 0,
      "REDUCE_PAR must divide 32 for correct output channel assignment");
  SplitReshapeQKV() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_hw = 0, i_ch = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output_q[2];
    PipelineDelayBuffer<TOutputWord> delayed_output_k[2];
    PipelineDelayBuffer<TOutputWord> delayed_output_v[2];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      delayed_output_q[0] = PipelineDelayBuffer<TOutputWord>(depth);
      delayed_output_k[0] = PipelineDelayBuffer<TOutputWord>(depth);
      delayed_output_v[0] = PipelineDelayBuffer<TOutputWord>(depth);
      actor_status =
          ActorStatus(depth, IN_HEIGHT * IN_WIDTH * IN_CH / (REDUCE_PAR));
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

  ActorStatus step(hls::stream<TInputWord> i_data[1],
                   hls::stream<TOutputWord> o_data_q[2],
                   hls::stream<TOutputWord> o_data_k[2],
                   hls::stream<TOutputWord> o_data_v[2]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    if (i_data[0].empty()) {
      firing_condition = false;
    }

    if (firing_condition) {

      // If there is data in the input stream, process it.
      hls::stream<TOutputWord> instant_output_stream_q[2];
      hls::stream<TOutputWord> instant_output_stream_k[2];
      hls::stream<TOutputWord> instant_output_stream_v[2];
      SplitReshapeQKV::pipeline_body(i_data, instant_output_stream_q,
                                  instant_output_stream_k,
                                  instant_output_stream_v, st.i_ch);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      if (!instant_output_stream_q[0].empty()) {
        st.delayed_output_q[0].push(instant_output_stream_q[0].read(), true);
      } else {
        // If the output stream is empty, push a placeholder.
        st.delayed_output_q[0].push(TOutputWord(), false);
      }
      if (!instant_output_stream_q[1].empty()) {
        st.delayed_output_q[1].push(instant_output_stream_q[1].read(), true);
      } else {
        // If the output stream is empty, push a placeholder.
        st.delayed_output_q[1].push(TOutputWord(), false);
      }
      if (!instant_output_stream_k[0].empty()) {
        st.delayed_output_k[0].push(instant_output_stream_k[0].read(), true);
      } else {
        // If the output stream is empty, push a placeholder.
        st.delayed_output_k[0].push(TOutputWord(), false);
      }
      if (!instant_output_stream_k[1].empty()) {
        st.delayed_output_k[1].push(instant_output_stream_k[1].read(), true);
      } else {
        // If the output stream is empty, push a placeholder.
        st.delayed_output_k[1].push(TOutputWord(), false);
      }
      if (!instant_output_stream_v[0].empty()) {
        st.delayed_output_v[0].push(instant_output_stream_v[0].read(), true);
      } else {
        // If the output stream is empty, push a placeholder.
        st.delayed_output_v[0].push(TOutputWord(), false);
      }
      if (!instant_output_stream_v[1].empty()) {
        st.delayed_output_v[1].push(instant_output_stream_v[1].read(), true);
      } else {
        // If the output stream is empty, push a placeholder.
        st.delayed_output_v[1].push(TOutputWord(), false);
      }

      // Update the counters.
      st.i_ch += REDUCE_PAR;
      if (st.i_ch >= IN_CH) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        st.i_ch = 0;
        st.i_hw++;
      }
      if (st.i_hw >= IN_HEIGHT * IN_WIDTH) {
        st.i_hw = 0; // Reset the height/width index if we have processed all
                     // iterations.
      }

    } else {
      // If there is no data in the input stream, push a delay slot.
      st.delayed_output_q[0].push(TOutputWord(), false);
      st.delayed_output_k[0].push(TOutputWord(), false);
      st.delayed_output_v[0].push(TOutputWord(), false);
      st.delayed_output_q[1].push(TOutputWord(), false);
      st.delayed_output_k[1].push(TOutputWord(), false);
      st.delayed_output_v[1].push(TOutputWord(), false);
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    if (st.delayed_output_q[0].pop(out)) {
      o_data_q[0].write(out);
    }
    if (st.delayed_output_k[0].pop(out)) {
      o_data_k[0].write(out);
    }
    if (st.delayed_output_v[0].pop(out)) {
      o_data_v[0].write(out);
    }
    if (st.delayed_output_q[1].pop(out)) {
      o_data_q[1].write(out);
    }
    if (st.delayed_output_k[1].pop(out)) {
      o_data_k[1].write(out);
    }
    if (st.delayed_output_v[1].pop(out)) {
      o_data_v[1].write(out);
    }

    // Return the actor status.
    return st.actor_status;
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[1],
           hls::stream<TOutputWord> o_data_q[2],
           hls::stream<TOutputWord> o_data_k[2],
           hls::stream<TOutputWord> o_data_v[2]) {
    for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH; i_hw++) {
    STREAMINGSPLITCHANNELS_RUN_LOOP:
      for (size_t i_ch = 0; i_ch < IN_CH; i_ch += REDUCE_PAR) {
#pragma HLS PIPELINE II = 1
        pipeline_body(i_data, o_data_q, o_data_k, o_data_v, i_ch);
      }
    }
  }

private:
  void pipeline_body(hls::stream<TInputWord> i_data[1],
                     hls::stream<TOutputWord> o_data_q[2],
                     hls::stream<TOutputWord> o_data_k[2],
                     hls::stream<TOutputWord> o_data_v[2], size_t i_ch) {
#pragma HLS inline
    Quantizer quantizer;
    TInputWord in_word = i_data[0].read();
    TOutputWord out_word;
    for (size_t ch_par = 0; ch_par < REDUCE_PAR; ch_par++) {
      out_word[ch_par] = quantizer(in_word[ch_par]);
    }
    if (i_ch < 32) {
      o_data_q[0].write(out_word);
    } else if (i_ch < 64) {
      o_data_k[0].write(out_word);
    } else if (i_ch < 128) {
      o_data_v[0].write(out_word);
    } else if (i_ch < 160) {
      o_data_q[1].write(out_word);
    } else if (i_ch < 192) {
      o_data_k[1].write(out_word);
    } else if (i_ch < 256) {
      o_data_v[1].write(out_word);
    }
  }
};

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t IN_HEIGHT,
          size_t IN_WIDTH, size_t IN_CH, size_t REDUCE_PAR>
class ReshapeV {
public:
  static_assert(IN_CH % REDUCE_PAR == 0,
                "IN_CH must be a multiple of REDUCE_PAR");
  static_assert(REDUCE_PAR > 0, "REDUCE_PAR must be greater than 0");
  static_assert(IN_HEIGHT > 0 && IN_WIDTH > 0,
                "IN_HEIGHT and IN_WIDTH must be greater than 0");
  static_assert(
      32 % REDUCE_PAR == 0,
      "REDUCE_PAR must divide 32 for correct output channel assignment");
  ReshapeV() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_hw = 0, i_ch = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output;
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      delayed_output = PipelineDelayBuffer<TOutputWord>(depth);
      actor_status =
          ActorStatus(depth, IN_HEIGHT * IN_WIDTH * IN_CH / (REDUCE_PAR));
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

  ActorStatus step(hls::stream<TInputWord> i_data[2],
                   hls::stream<TOutputWord> o_data[1]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    if (st.i_ch < 64) {
      if (i_data[0].empty()) {
        firing_condition = false;
      }
    } else {
      if (i_data[1].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      // If there is data in the input stream, process it.
      hls::stream<TOutputWord> instant_output_stream[1];
      ReshapeV::pipeline_body(i_data, instant_output_stream, st.i_ch);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      if (!instant_output_stream[0].empty()) {
        st.delayed_output.push(instant_output_stream[0].read(), true);
      } else {
        // If the output stream is empty, push a placeholder.
        st.delayed_output.push(TOutputWord(), false);
      }

      // Update the counters.
      st.i_ch += REDUCE_PAR;
      if (st.i_ch >= IN_CH) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        st.i_ch = 0;
        st.i_hw++;
      }
      if (st.i_hw >= IN_HEIGHT * IN_WIDTH) {
        st.i_hw = 0; // Reset the height/width index if we have processed all
                     // iterations.
      }

    } else {
      // If there is no data in the input stream, push a delay slot.
      st.delayed_output.push(TOutputWord(), false);
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    if (st.delayed_output.pop(out)) {
      o_data[0].write(out);
    }

    // Return the actor status.
    return st.actor_status;
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[2],
           hls::stream<TOutputWord> o_data[1]) {
    for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH; i_hw++) {
    STREAMINGSPLITCHANNELS_RUN_LOOP:
      for (size_t i_ch = 0; i_ch < IN_CH; i_ch += REDUCE_PAR) {
#pragma HLS PIPELINE II = 1
        pipeline_body(i_data, o_data, i_ch);
      }
    }
  }

private:
  void pipeline_body(hls::stream<TInputWord> i_data[2],
                     hls::stream<TOutputWord> o_data[1], size_t i_ch) {
#pragma HLS inline
    Quantizer quantizer;
    TInputWord in_word;
    TOutputWord out_word;

    if (i_ch < 64) {
      in_word = i_data[0].read();
    } else {
      in_word = i_data[1].read();
    }
    for (size_t ch_par = 0; ch_par < REDUCE_PAR; ch_par++) {
      out_word[ch_par] = quantizer(in_word[ch_par]);
    }
    o_data[0].write(out_word);
  }
};