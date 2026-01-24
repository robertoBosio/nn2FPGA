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
          typename TOutput, typename Quantizer, size_t SPLIT, size_t IN_HEIGHT,
          size_t IN_WIDTH, size_t IN_CH, size_t CH_PAR, size_t W_PAR>
class StreamingSplitChannels {
public:
  static_assert(IN_CH % CH_PAR == 0, "IN_CH must be a multiple of CH_PAR");
  static_assert(CH_PAR > 0, "CH_PAR must be greater than 0");
  static_assert(SPLIT > 0, "SPLIT must be greater than 0");
  static_assert(SPLIT < IN_CH, "SPLIT must be less than IN_CH");
  static_assert(IN_HEIGHT > 0 && IN_WIDTH > 0,
                "IN_HEIGHT and IN_WIDTH must be greater than 0");
  static_assert(SPLIT % CH_PAR == 0, "SPLIT must be a multiple of CH_PAR");
  static_assert((IN_CH - SPLIT) % CH_PAR == 0,
                "IN_CH - SPLIT must be a multiple of CH_PAR");
  StreamingSplitChannels() = default;
  
  struct StepState {
    // Loop iteration indexes.
    size_t i_hw = 0, i_ch = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output_1[W_PAR];
    PipelineDelayBuffer<TOutputWord> delayed_output_2[W_PAR];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < W_PAR; i++) {
        delayed_output_1[i] = PipelineDelayBuffer<TOutputWord>(depth);
        delayed_output_2[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status =
          ActorStatus(depth, IN_HEIGHT * IN_WIDTH * IN_CH / (CH_PAR * W_PAR));
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
  
  ActorStatus step(hls::stream<TInputWord> i_data[W_PAR],
                   hls::stream<TOutputWord> o_data_1[W_PAR],
                   hls::stream<TOutputWord> o_data_2[W_PAR]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t w_par = 0; w_par < W_PAR; w_par++) {
      if (i_data[w_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      // If there is data in the input stream, process it.
      hls::stream<TOutputWord> instant_output_stream_1[W_PAR];
      hls::stream<TOutputWord> instant_output_stream_2[W_PAR];
        StreamingSplitChannels::pipeline_body(i_data, instant_output_stream_1,
                                                instant_output_stream_2,
                                                st.i_ch);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t w_par = 0; w_par < W_PAR; w_par++) {
        if (!instant_output_stream_1[w_par].empty()) {
          st.delayed_output_1[w_par].push(instant_output_stream_1[w_par].read(),
                                          true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output_1[w_par].push(TOutputWord(), false);
        }
        if (!instant_output_stream_2[w_par].empty()) {
          st.delayed_output_2[w_par].push(instant_output_stream_2[w_par].read(),
                                          true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output_2[w_par].push(TOutputWord(), false);
        }
      }

      // Update the counters.
      st.i_ch += CH_PAR;
      if (st.i_ch >= IN_CH) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        st.i_ch = 0;
        st.i_hw++;
      }
      if (st.i_hw >= IN_HEIGHT * IN_WIDTH / W_PAR) {
        st.i_hw = 0; // Reset the height/width index if we have processed all
                     // iterations.
      }

    } else {
      // If there is no data in the input stream, push a delay slot.
      for (size_t w_par = 0; w_par < W_PAR; w_par++) {
        st.delayed_output_1[w_par].push(TOutputWord(), false);
        st.delayed_output_2[w_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    for (size_t w_par = 0; w_par < W_PAR; w_par++) {
      if (st.delayed_output_1[w_par].pop(out)) {
        o_data_1[w_par].write(out);
      }
      if (st.delayed_output_2[w_par].pop(out)) {
        o_data_2[w_par].write(out);
      }
    }

    // Return the actor status.
    return st.actor_status;
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[W_PAR],
           hls::stream<TOutputWord> o_data_1[W_PAR],
           hls::stream<TOutputWord> o_data_2[W_PAR]) {
    for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH / W_PAR; i_hw++) {
    STREAMINGSPLITCHANNELS_RUN_LOOP:
      for (size_t i_ch = 0; i_ch < IN_CH; i_ch += CH_PAR) {
#pragma HLS PIPELINE II = 1
        pipeline_body(i_data, o_data_1, o_data_2, i_ch);
      }
    }
  }

private:
  void pipeline_body(hls::stream<TInputWord> i_data[W_PAR],
                     hls::stream<TOutputWord> o_data_1[W_PAR],
                     hls::stream<TOutputWord> o_data_2[W_PAR], size_t i_ch) {
#pragma HLS inline
    Quantizer quantizer;
    for (size_t w_par = 0; w_par < W_PAR; w_par++) {
      TInputWord in_word = i_data[w_par].read();
      TOutputWord out_word;
      for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++) {
        out_word[ch_par] = quantizer(in_word[ch_par]);
      }
      if (i_ch < SPLIT) {
        o_data_1[w_par].write(out_word);
      } else {
        o_data_2[w_par].write(out_word);
      }
    }
  }
};

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t SPLIT, size_t IN_HEIGHT,
          size_t IN_WIDTH, size_t IN_CH, size_t CH_PAR, size_t W_PAR>
class StreamingSplitWidths {
public:
  static_assert(IN_CH % CH_PAR == 0, "IN_CH must be a multiple of CH_PAR");
  static_assert(CH_PAR > 0, "CH_PAR must be greater than 0");
  static_assert(SPLIT > 0, "SPLIT must be greater than 0");
  static_assert(SPLIT < IN_WIDTH, "SPLIT must be less than IN_WIDTH");
  static_assert(IN_HEIGHT > 0 && IN_WIDTH > 0,
                "IN_HEIGHT and IN_WIDTH must be greater than 0");
  static_assert(SPLIT % W_PAR == 0, "SPLIT must be a multiple of W_PAR");
  static_assert((IN_WIDTH - SPLIT) % W_PAR == 0,
                "IN_WIDTH - SPLIT must be a multiple of W_PAR");
  StreamingSplitWidths() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_w = 0, i_ch = 0, i_h = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output_1[W_PAR];
    PipelineDelayBuffer<TOutputWord> delayed_output_2[W_PAR];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < W_PAR; i++) {
        delayed_output_1[i] = PipelineDelayBuffer<TOutputWord>(depth);
        delayed_output_2[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status =
          ActorStatus(depth, IN_HEIGHT * IN_WIDTH * IN_CH / (CH_PAR * W_PAR));
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

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[W_PAR],
           hls::stream<TOutputWord> o_data_1[W_PAR],
           hls::stream<TOutputWord> o_data_2[W_PAR]) {
    for (size_t i_h = 0; i_h < IN_HEIGHT; i_h++) {
      for (size_t i_w = 0; i_w < IN_WIDTH; i_w += W_PAR) {
      STREAMINGSPLITWIDTHS_RUN_LOOP:
        for (size_t i_ch = 0; i_ch < IN_CH; i_ch += CH_PAR) {
#pragma HLS PIPELINE II = 1
          pipeline_body(i_data, o_data_1, o_data_2, i_w);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[W_PAR],
                   hls::stream<TOutputWord> o_data_1[W_PAR],
                   hls::stream<TOutputWord> o_data_2[W_PAR]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t w_par = 0; w_par < W_PAR; w_par++) {
      if (i_data[w_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      // If there is data in the input stream, process it.
      hls::stream<TOutputWord> instant_output_stream_1[W_PAR];
      hls::stream<TOutputWord> instant_output_stream_2[W_PAR];
      StreamingSplitWidths::pipeline_body(i_data, instant_output_stream_1,
                                            instant_output_stream_2, st.i_w);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t w_par = 0; w_par < W_PAR; w_par++) {
        if (!instant_output_stream_1[w_par].empty()) {
          st.delayed_output_1[w_par].push(instant_output_stream_1[w_par].read(),
                                          true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output_1[w_par].push(TOutputWord(), false);
        }
        if (!instant_output_stream_2[w_par].empty()) {
          st.delayed_output_2[w_par].push(instant_output_stream_2[w_par].read(),
                                          true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output_2[w_par].push(TOutputWord(), false);
        }
      }

      // Update the counters.
      st.i_ch += CH_PAR;
      if (st.i_ch >= IN_CH) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        st.i_ch = 0;
        st.i_w += W_PAR;
      }
      if (st.i_w >= IN_WIDTH) {
        st.i_w = 0; // Reset the height/width index if we have processed all
        // iterations.
        st.i_h++;
      }
      if (st.i_h >= IN_HEIGHT) {
        st.i_h = 0; // Reset the height index if we have processed all
                    // iterations.
      }
    } else {
      // If there is no data in the input stream, push a delay slot.
      for (size_t w_par = 0; w_par < W_PAR; w_par++) {
        st.delayed_output_1[w_par].push(TOutputWord(), false);
        st.delayed_output_2[w_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    for (size_t w_par = 0; w_par < W_PAR; w_par++) {
      if (st.delayed_output_1[w_par].pop(out)) {
        o_data_1[w_par].write(out);
      }
      if (st.delayed_output_2[w_par].pop(out)) {
        o_data_2[w_par].write(out);
      }
    }

    // Return the actor status.
    return st.actor_status;
  }

private:
  void pipeline_body(hls::stream<TInputWord> i_data[W_PAR],
                     hls::stream<TOutputWord> o_data_1[W_PAR],
                     hls::stream<TOutputWord> o_data_2[W_PAR], size_t i_w) {
#pragma HLS inline
    Quantizer quantizer;
    for (size_t w_par = 0; w_par < W_PAR; w_par++) {
      TInputWord in_word = i_data[w_par].read();
      TOutputWord out_word;
      for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++) {
        out_word[ch_par] = quantizer(in_word[ch_par]);
      }
      if (i_w < SPLIT) {
        o_data_1[w_par].write(out_word);
      } else {
        o_data_2[w_par].write(out_word);
      }
    }
  }
};

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t SPLIT, size_t IN_HEIGHT,
          size_t IN_WIDTH, size_t IN_CH, size_t CH_PAR, size_t W_PAR>
class StreamingSplitHeights {
public:
  static_assert(IN_CH % CH_PAR == 0, "IN_CH must be a multiple of CH_PAR");
  static_assert(CH_PAR > 0, "CH_PAR must be greater than 0");
  static_assert(SPLIT > 0, "SPLIT must be greater than 0");
  static_assert(SPLIT < IN_HEIGHT, "SPLIT must be less than IN_HEIGHT");
  static_assert(IN_HEIGHT > 0 && IN_WIDTH > 0,
                "IN_HEIGHT and IN_WIDTH must be greater than 0");
  StreamingSplitHeights() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_h = 0, i_w = 0, i_ch = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output_1[W_PAR];
    PipelineDelayBuffer<TOutputWord> delayed_output_2[W_PAR];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < W_PAR; i++) {
        delayed_output_1[i] = PipelineDelayBuffer<TOutputWord>(depth);
        delayed_output_2[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status =
          ActorStatus(depth, IN_HEIGHT * IN_WIDTH * IN_CH / (CH_PAR * W_PAR));
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

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[W_PAR],
           hls::stream<TOutputWord> o_data_1[W_PAR],
           hls::stream<TOutputWord> o_data_2[W_PAR]) {
    for (size_t i_h = 0; i_h < IN_HEIGHT; i_h++) {
      for (size_t i_w = 0; i_w < IN_WIDTH; i_w += W_PAR) {
        for (size_t i_ch = 0; i_ch < IN_CH; i_ch += CH_PAR) {
#pragma HLS PIPELINE II = 1
          pipeline_body(i_data, o_data_1, o_data_2, i_h);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[W_PAR],
                   hls::stream<TOutputWord> o_data_1[W_PAR],
                   hls::stream<TOutputWord> o_data_2[W_PAR]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t w_par = 0; w_par < W_PAR; w_par++) {
      if (i_data[w_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      // If there is data in the input stream, process it.
      hls::stream<TOutputWord> instant_output_stream_1[W_PAR];
      hls::stream<TOutputWord> instant_output_stream_2[W_PAR];
      StreamingSplitHeights::pipeline_body(i_data, instant_output_stream_1,
                                           instant_output_stream_2, st.i_h);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t w_par = 0; w_par < W_PAR; w_par++) {
        if (!instant_output_stream_1[w_par].empty()) {
          st.delayed_output_1[w_par].push(instant_output_stream_1[w_par].read(),
                                          true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output_1[w_par].push(TOutputWord(), false);
        }
        if (!instant_output_stream_2[w_par].empty()) {
          st.delayed_output_2[w_par].push(instant_output_stream_2[w_par].read(),
                                          true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output_2[w_par].push(TOutputWord(), false);
        }
      }

      // Update the counters.
      st.i_ch += CH_PAR;
      if (st.i_ch >= IN_CH) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        st.i_ch = 0;
        st.i_w += W_PAR;
      }
      if (st.i_w >= IN_WIDTH) {
        st.i_w = 0; // Reset the height/width index if we have processed all
        // iterations.
        st.i_h++;
      }
      if (st.i_h >= IN_HEIGHT) {
        st.i_h = 0; // Reset the height index if we have processed all
                    // iterations.
      }
    } else {
      // If there is no data in the input stream, push a delay slot.
      for (size_t w_par = 0; w_par < W_PAR; w_par++) {
        st.delayed_output_1[w_par].push(TOutputWord(), false);
        st.delayed_output_2[w_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    for (size_t w_par = 0; w_par < W_PAR; w_par++) {
      if (st.delayed_output_1[w_par].pop(out)) {
        o_data_1[w_par].write(out);
      }
      if (st.delayed_output_2[w_par].pop(out)) {
        o_data_2[w_par].write(out);
      }
    }

    // Return the actor status.
    return st.actor_status;
  }

private:
  void pipeline_body(hls::stream<TInputWord> i_data[W_PAR],
                     hls::stream<TOutputWord> o_data_1[W_PAR],
                     hls::stream<TOutputWord> o_data_2[W_PAR], size_t i_h) {
#pragma HLS inline
    Quantizer quantizer;
    for (size_t w_par = 0; w_par < W_PAR; w_par++) {
      TInputWord in_word = i_data[w_par].read();
      TOutputWord out_word;
      for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++) {
        out_word[ch_par] = quantizer(in_word[ch_par]);
      }
      if (i_h < SPLIT) {
        o_data_1[w_par].write(out_word);
      } else {
        o_data_2[w_par].write(out_word);
      }
    }
  }
};