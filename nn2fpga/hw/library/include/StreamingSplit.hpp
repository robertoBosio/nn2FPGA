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
          typename TOutput, typename Quantizer, size_t SPLIT, size_t DIM0,
          size_t DIM1, size_t DIM2, size_t DIM1_UNROLL, size_t DIM2_UNROLL>
class StreamingSplitDim2 {
  static_assert(DIM2 % DIM2_UNROLL == 0,
                "DIM2 must be a multiple of DIM2_UNROLL");
  static_assert(DIM2_UNROLL > 0, "DIM2_UNROLL must be greater than 0");
  static_assert(SPLIT > 0, "SPLIT must be greater than 0");
  static_assert(SPLIT < DIM2, "SPLIT must be less than DIM2");
  static_assert(DIM0 > 0 && DIM1 > 0, "DIM0 and DIM1 must be greater than 0");
  static_assert(SPLIT % DIM2_UNROLL == 0,
                "SPLIT must be a multiple of DIM2_UNROLL");
  static_assert((DIM2 - SPLIT) % DIM2_UNROLL == 0,
                "DIM2 - SPLIT must be a multiple of DIM2_UNROLL");

public:
  StreamingSplitDim2() = default;

  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  ActorStatus step(hls::stream<TInputWord> i_data[DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data_1[DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data_2[DIM1_UNROLL]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (i_data[i_dim1_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      // If there is data in the input stream, process it.
      hls::stream<TOutputWord> instant_output_stream_1[DIM1_UNROLL];
      hls::stream<TOutputWord> instant_output_stream_2[DIM1_UNROLL];
      StreamingSplitDim2::pipeline_body(i_data, instant_output_stream_1,
                                        instant_output_stream_2, st.i_dim2);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        if (!instant_output_stream_1[i_dim1_par].empty()) {
          st.delayed_output_1[i_dim1_par].push(
              instant_output_stream_1[i_dim1_par].read(), true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output_1[i_dim1_par].push(TOutputWord(), false);
        }
        if (!instant_output_stream_2[i_dim1_par].empty()) {
          st.delayed_output_2[i_dim1_par].push(
              instant_output_stream_2[i_dim1_par].read(), true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output_2[i_dim1_par].push(TOutputWord(), false);
        }
      }

      // Update the counters.
      st.i_dim2 += DIM2_UNROLL;
      if (st.i_dim2 >= DIM2) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        st.i_dim2 = 0;
        st.i_dim01++;
      }
      if (st.i_dim01 >= DIM0 * DIM1 / DIM1_UNROLL) {
        st.i_dim01 = 0; // Reset the height/width index if we have processed all
                        // iterations.
      }

    } else {
      // If there is no data in the input stream, push a delay slot.
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        st.delayed_output_1[i_dim1_par].push(TOutputWord(), false);
        st.delayed_output_2[i_dim1_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (st.delayed_output_1[i_dim1_par].pop(out)) {
        o_data_1[i_dim1_par].write(out);
      }
      if (st.delayed_output_2[i_dim1_par].pop(out)) {
        o_data_2[i_dim1_par].write(out);
      }
    }

    // Return the actor status.
    return st.actor_status;
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[DIM1_UNROLL],
           hls::stream<TOutputWord> o_data_1[DIM1_UNROLL],
           hls::stream<TOutputWord> o_data_2[DIM1_UNROLL]) {
    for (size_t i_dim01 = 0; i_dim01 < DIM0 * DIM1 / DIM1_UNROLL; i_dim01++) {
    STREAMINGSPLITDIM2_RUN_LOOP:
      for (size_t i_dim2 = 0; i_dim2 < DIM2; i_dim2 += DIM2_UNROLL) {
#pragma HLS PIPELINE II = 1
        pipeline_body(i_data, o_data_1, o_data_2, i_dim2);
      }
    }
  }

private:
  struct StepState {
    // Loop iteration indexes.
    size_t i_dim01 = 0, i_dim2 = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output_1[DIM1_UNROLL];
    PipelineDelayBuffer<TOutputWord> delayed_output_2[DIM1_UNROLL];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < DIM1_UNROLL; i++) {
        delayed_output_1[i] = PipelineDelayBuffer<TOutputWord>(depth);
        delayed_output_2[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status =
          ActorStatus(depth, DIM0 * DIM1 * DIM2 / (DIM2_UNROLL * DIM1_UNROLL));
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

  void pipeline_body(hls::stream<TInputWord> i_data[DIM1_UNROLL],
                     hls::stream<TOutputWord> o_data_1[DIM1_UNROLL],
                     hls::stream<TOutputWord> o_data_2[DIM1_UNROLL],
                     size_t i_dim2) {
#pragma HLS inline
    Quantizer quantizer;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      TInputWord in_word = i_data[i_dim1_par].read();
      TOutputWord out_word;
      for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL; i_dim2_par++) {
        out_word[i_dim2_par] = quantizer(in_word[i_dim2_par]);
      }
      if (i_dim2 < SPLIT) {
        o_data_1[i_dim1_par].write(out_word);
      } else {
        o_data_2[i_dim1_par].write(out_word);
      }
    }
  }
};

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t SPLIT, size_t DIM0,
          size_t DIM1, size_t DIM2, size_t DIM1_UNROLL, size_t DIM2_UNROLL>
class StreamingSplitDim1 {
  static_assert(DIM2 % DIM2_UNROLL == 0,
                "DIM2 must be a multiple of DIM2_UNROLL");
  static_assert(DIM2_UNROLL > 0, "DIM2_UNROLL must be greater than 0");
  static_assert(SPLIT > 0, "SPLIT must be greater than 0");
  static_assert(SPLIT < DIM1, "SPLIT must be less than DIM1");
  static_assert(DIM0 > 0 && DIM1 > 0, "DIM0 and DIM1 must be greater than 0");
  static_assert(SPLIT % DIM1_UNROLL == 0,
                "SPLIT must be a multiple of DIM1_UNROLL");
  static_assert((DIM1 - SPLIT) % DIM1_UNROLL == 0,
                "DIM1 - SPLIT must be a multiple of DIM1_UNROLL");

public:
  StreamingSplitDim1() = default;

  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[DIM1_UNROLL],
           hls::stream<TOutputWord> o_data_1[DIM1_UNROLL],
           hls::stream<TOutputWord> o_data_2[DIM1_UNROLL]) {
    for (size_t i_dim0 = 0; i_dim0 < DIM0; i_dim0++) {
      for (size_t i_dim1 = 0; i_dim1 < DIM1; i_dim1 += DIM1_UNROLL) {
      STREAMINGSPLITDIM1_RUN_LOOP:
        for (size_t i_dim2 = 0; i_dim2 < DIM2; i_dim2 += DIM2_UNROLL) {
#pragma HLS PIPELINE II = 1
          pipeline_body(i_data, o_data_1, o_data_2, i_dim1);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data_1[DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data_2[DIM1_UNROLL]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (i_data[i_dim1_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      // If there is data in the input stream, process it.
      hls::stream<TOutputWord> instant_output_stream_1[DIM1_UNROLL];
      hls::stream<TOutputWord> instant_output_stream_2[DIM1_UNROLL];
      StreamingSplitDim1::pipeline_body(i_data, instant_output_stream_1,
                                        instant_output_stream_2, st.i_dim1);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        if (!instant_output_stream_1[i_dim1_par].empty()) {
          st.delayed_output_1[i_dim1_par].push(
              instant_output_stream_1[i_dim1_par].read(), true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output_1[i_dim1_par].push(TOutputWord(), false);
        }
        if (!instant_output_stream_2[i_dim1_par].empty()) {
          st.delayed_output_2[i_dim1_par].push(
              instant_output_stream_2[i_dim1_par].read(), true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output_2[i_dim1_par].push(TOutputWord(), false);
        }
      }

      // Update the counters.
      st.i_dim2 += DIM2_UNROLL;
      if (st.i_dim2 >= DIM2) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        st.i_dim2 = 0;
        st.i_dim1 += DIM1_UNROLL;
      }
      if (st.i_dim1 >= DIM1) {
        st.i_dim1 = 0; // Reset the height/width index if we have processed all
        // iterations.
        st.i_dim0++;
      }
      if (st.i_dim0 >= DIM0) {
        st.i_dim0 = 0; // Reset the height index if we have processed all
                       // iterations.
      }
    } else {
      // If there is no data in the input stream, push a delay slot.
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        st.delayed_output_1[i_dim1_par].push(TOutputWord(), false);
        st.delayed_output_2[i_dim1_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (st.delayed_output_1[i_dim1_par].pop(out)) {
        o_data_1[i_dim1_par].write(out);
      }
      if (st.delayed_output_2[i_dim1_par].pop(out)) {
        o_data_2[i_dim1_par].write(out);
      }
    }

    // Return the actor status.
    return st.actor_status;
  }

private:
  struct StepState {
    // Loop iteration indexes.
    size_t i_dim1 = 0, i_dim2 = 0, i_dim0 = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output_1[DIM1_UNROLL];
    PipelineDelayBuffer<TOutputWord> delayed_output_2[DIM1_UNROLL];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < DIM1_UNROLL; i++) {
        delayed_output_1[i] = PipelineDelayBuffer<TOutputWord>(depth);
        delayed_output_2[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status =
          ActorStatus(depth, DIM0 * DIM1 * DIM2 / (DIM2_UNROLL * DIM1_UNROLL));
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

  void pipeline_body(hls::stream<TInputWord> i_data[DIM1_UNROLL],
                     hls::stream<TOutputWord> o_data_1[DIM1_UNROLL],
                     hls::stream<TOutputWord> o_data_2[DIM1_UNROLL],
                     size_t i_dim1) {
#pragma HLS inline
    Quantizer quantizer;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      TInputWord in_word = i_data[i_dim1_par].read();
      TOutputWord out_word;
      for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL; i_dim2_par++) {
        out_word[i_dim2_par] = quantizer(in_word[i_dim2_par]);
      }
      if (i_dim1 < SPLIT) {
        o_data_1[i_dim1_par].write(out_word);
      } else {
        o_data_2[i_dim1_par].write(out_word);
      }
    }
  }
};

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t SPLIT, size_t DIM0,
          size_t DIM1, size_t DIM2, size_t DIM1_UNROLL, size_t DIM2_UNROLL>
class StreamingSplitDim0 {
  static_assert(DIM2 % DIM2_UNROLL == 0,
                "DIM2 must be a multiple of DIM2_UNROLL");
  static_assert(DIM2_UNROLL > 0, "DIM2_UNROLL must be greater than 0");
  static_assert(SPLIT > 0, "SPLIT must be greater than 0");
  static_assert(SPLIT < DIM0, "SPLIT must be less than DIM0");
  static_assert(DIM0 > 0 && DIM1 > 0, "DIM0 and DIM1 must be greater than 0");

public:
  StreamingSplitDim0() = default;

  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[DIM1_UNROLL],
           hls::stream<TOutputWord> o_data_1[DIM1_UNROLL],
           hls::stream<TOutputWord> o_data_2[DIM1_UNROLL]) {
    for (size_t i_dim0 = 0; i_dim0 < DIM0; i_dim0++) {
      for (size_t i_dim1 = 0; i_dim1 < DIM1; i_dim1 += DIM1_UNROLL) {
      STREAMINGSPLITDIM0_RUN_LOOP:
        for (size_t i_dim2 = 0; i_dim2 < DIM2; i_dim2 += DIM2_UNROLL) {
#pragma HLS PIPELINE II = 1
          pipeline_body(i_data, o_data_1, o_data_2, i_dim0);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data_1[DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data_2[DIM1_UNROLL]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (i_data[i_dim1_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      // If there is data in the input stream, process it.
      hls::stream<TOutputWord> instant_output_stream_1[DIM1_UNROLL];
      hls::stream<TOutputWord> instant_output_stream_2[DIM1_UNROLL];
      StreamingSplitDim0::pipeline_body(i_data, instant_output_stream_1,
                                        instant_output_stream_2, st.i_dim0);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        if (!instant_output_stream_1[i_dim1_par].empty()) {
          st.delayed_output_1[i_dim1_par].push(
              instant_output_stream_1[i_dim1_par].read(), true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output_1[i_dim1_par].push(TOutputWord(), false);
        }
        if (!instant_output_stream_2[i_dim1_par].empty()) {
          st.delayed_output_2[i_dim1_par].push(
              instant_output_stream_2[i_dim1_par].read(), true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output_2[i_dim1_par].push(TOutputWord(), false);
        }
      }

      // Update the counters.
      st.i_dim2 += DIM2_UNROLL;
      if (st.i_dim2 >= DIM2) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        st.i_dim2 = 0;
        st.i_dim1 += DIM1_UNROLL;
      }
      if (st.i_dim1 >= DIM1) {
        st.i_dim1 = 0; // Reset the height/width index if we have processed all
        // iterations.
        st.i_dim0++;
      }
      if (st.i_dim0 >= DIM0) {
        st.i_dim0 = 0; // Reset the height index if we have processed all
                       // iterations.
      }
    } else {
      // If there is no data in the input stream, push a delay slot.
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        st.delayed_output_1[i_dim1_par].push(TOutputWord(), false);
        st.delayed_output_2[i_dim1_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (st.delayed_output_1[i_dim1_par].pop(out)) {
        o_data_1[i_dim1_par].write(out);
      }
      if (st.delayed_output_2[i_dim1_par].pop(out)) {
        o_data_2[i_dim1_par].write(out);
      }
    }

    // Return the actor status.
    return st.actor_status;
  }

private:
  struct StepState {
    // Loop iteration indexes.
    size_t i_dim0 = 0, i_dim1 = 0, i_dim2 = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output_1[DIM1_UNROLL];
    PipelineDelayBuffer<TOutputWord> delayed_output_2[DIM1_UNROLL];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < DIM1_UNROLL; i++) {
        delayed_output_1[i] = PipelineDelayBuffer<TOutputWord>(depth);
        delayed_output_2[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status =
          ActorStatus(depth, DIM0 * DIM1 * DIM2 / (DIM2_UNROLL * DIM1_UNROLL));
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

  void pipeline_body(hls::stream<TInputWord> i_data[DIM1_UNROLL],
                     hls::stream<TOutputWord> o_data_1[DIM1_UNROLL],
                     hls::stream<TOutputWord> o_data_2[DIM1_UNROLL],
                     size_t i_dim0) {
#pragma HLS inline
    Quantizer quantizer;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      TInputWord in_word = i_data[i_dim1_par].read();
      TOutputWord out_word;
      for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL; i_dim2_par++) {
        out_word[i_dim2_par] = quantizer(in_word[i_dim2_par]);
      }
      if (i_dim0 < SPLIT) {
        o_data_1[i_dim1_par].write(out_word);
      } else {
        o_data_2[i_dim1_par].write(out_word);
      }
    }
  }
};