#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t IN_DIM0, size_t IN_DIM1,
          size_t IN_DIM2, size_t IN_DIM1_UNROLL, size_t OUT_DIM1_UNROLL,
          size_t IN_DIM2_UNROLL, size_t OUT_DIM2_UNROLL>
class BandwidthAdjustIncreaseStreams {
public:
  static_assert(IN_DIM1_UNROLL > 0, "IN_DIM1_UNROLL must be greater than 0");
  static_assert(OUT_DIM1_UNROLL > 0, "OUT_DIM1_UNROLL must be greater than 0");
  static_assert(IN_DIM1_UNROLL < OUT_DIM1_UNROLL,
                "IN_DIM1_UNROLL must be less than OUT_DIM1_UNROLL");
  static_assert(OUT_DIM1_UNROLL % IN_DIM1_UNROLL == 0,
                "OUT_DIM1_UNROLL must be a multiple of IN_DIM1_UNROLL");
  static_assert(IN_DIM1 % IN_DIM1_UNROLL == 0,
                "IN_DIM1 must be a multiple of IN_DIM1_UNROLL");
  static_assert(IN_DIM1 % OUT_DIM1_UNROLL == 0,
                "IN_DIM1 must be a multiple of OUT_DIM1_UNROLL");
  static_assert(IN_DIM2 % IN_DIM2_UNROLL == 0,
                "IN_DIM2_UNROLL must be a multiple of IN_DIM2");
  static_assert(IN_DIM2 % OUT_DIM2_UNROLL == 0,
                "OUT_DIM2_UNROLL must be a multiple of IN_DIM2");
  static_assert(IN_DIM2_UNROLL == OUT_DIM2_UNROLL,
                "IN_DIM2_UNROLL must be equal to OUT_DIM2_UNROLL");

  BandwidthAdjustIncreaseStreams() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_dim01 = 0, i_out_dim1_par = 0, i_dim2 = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output[OUT_DIM1_UNROLL];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < OUT_DIM1_UNROLL; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status = ActorStatus(depth, IN_DIM0 * IN_DIM1 * IN_DIM2 /
                                            (OUT_DIM2_UNROLL * IN_DIM1_UNROLL));
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
  void run(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
           hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL]) {

    // This is the case in which we increment the number of output streams, i.e.
    // the unrolling factor of DIM 1 over the streams.

    // Iterate over the input at groups of OUT_DIM1_UNROLL.
    for (size_t i_dim01 = 0; i_dim01 < IN_DIM0 * IN_DIM1;
         i_dim01 += OUT_DIM1_UNROLL) {
      // Iterate over the groups with a step of IN_DIM1_UNROLL.
      for (size_t i_out_dim1_par = 0; i_out_dim1_par < OUT_DIM1_UNROLL;
           i_out_dim1_par += IN_DIM1_UNROLL) {

        // Iterate over the last dimension with a step of OUT_DIM2_UNROLL.
      BANDWIDTHADJUSTINCREASESTREAMS_RUN_LOOP:
        for (size_t i_dim2 = 0; i_dim2 < IN_DIM2; i_dim2 += OUT_DIM2_UNROLL) {
#pragma HLS pipeline II = 1
          BandwidthAdjustIncreaseStreams::pipeline_body(i_data, o_data,
                                                        i_out_dim1_par);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_in_dim1_par = 0; i_in_dim1_par < IN_DIM1_UNROLL;
         i_in_dim1_par++) {
      if (i_data[i_in_dim1_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      hls::stream<TOutputWord> instant_output_stream[OUT_DIM1_UNROLL];
      BandwidthAdjustIncreaseStreams::pipeline_body(
          i_data, instant_output_stream, st.i_out_dim1_par);

      // Update the loop iteration indexes for the next step.
      st.i_dim2 += OUT_DIM2_UNROLL;
      if (st.i_dim2 >= IN_DIM2) {
        st.i_dim2 = 0;
        st.i_out_dim1_par += IN_DIM1_UNROLL;
      }
      if (st.i_out_dim1_par >= OUT_DIM1_UNROLL) {
        st.i_out_dim1_par = 0;
        st.i_dim01 += OUT_DIM1_UNROLL;
      }
      if (st.i_dim01 >= IN_DIM0 * IN_DIM1) {
        st.i_dim01 = 0;
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_out_dim1_par = 0; i_out_dim1_par < OUT_DIM1_UNROLL;
           i_out_dim1_par++) {
        if (!instant_output_stream[i_out_dim1_par].empty()) {
          st.delayed_output[i_out_dim1_par].push(
              instant_output_stream[i_out_dim1_par].read(), true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output[i_out_dim1_par].push(TOutputWord(), false);
        }
      }
    } else {
      // If no data is available, push empty outputs.
      for (size_t i_out_dim1_par = 0; i_out_dim1_par < OUT_DIM1_UNROLL;
           i_out_dim1_par++) {
        st.delayed_output[i_out_dim1_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    for (size_t i_out_dim1_par = 0; i_out_dim1_par < OUT_DIM1_UNROLL;
         i_out_dim1_par++) {
      if (st.delayed_output[i_out_dim1_par].pop(out)) {
        o_data[i_out_dim1_par].write(out);
      }
    }

    // Return the current firing iteration index.
    return st.actor_status;
  }

private:
  static void pipeline_body(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
                            hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL],
                            size_t i_out_dim1_par) {
#pragma HLS inline
    // Input word to read data from the input stream.
    TInputWord in_word;
    TOutputWord out_word; // Output word to hold the results.
    Quantizer quantizer;  // Quantizer instance for quantization.

    for (size_t i_in_dim1_par = 0; i_in_dim1_par < IN_DIM1_UNROLL;
         i_in_dim1_par++) {
      // Read the input data word from the input stream.
      in_word = i_data[i_in_dim1_par].read();

      for (size_t i_out_dim2_par = 0; i_out_dim2_par < OUT_DIM2_UNROLL;
           i_out_dim2_par++) {
        TInput in_data = in_word[i_out_dim2_par];

        // Quantize the input data.
        TOutput out_data = quantizer(in_data);

        // Store the quantized data in the output word.
        out_word[i_out_dim2_par] = out_data;
      }
      o_data[i_out_dim1_par + i_in_dim1_par].write(out_word);
    }
  }
};

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t IN_DIM0, size_t IN_DIM1,
          size_t IN_DIM2, size_t IN_DIM1_UNROLL, size_t OUT_DIM1_UNROLL,
          size_t IN_DIM2_UNROLL, size_t OUT_DIM2_UNROLL>
class BandwidthAdjustDecreaseStreams {
public:
  static_assert(IN_DIM1_UNROLL > 0, "IN_DIM1_UNROLL must be greater than 0");
  static_assert(OUT_DIM1_UNROLL > 0, "OUT_DIM1_UNROLL must be greater than 0");
  static_assert(IN_DIM1_UNROLL > OUT_DIM1_UNROLL,
                "OUT_DIM1_UNROLL must be less than IN_DIM1_UNROLL");
  static_assert(IN_DIM1_UNROLL % OUT_DIM1_UNROLL == 0,
                "OUT_DIM1_UNROLL must be a multiple of IN_DIM1_UNROLL");
  static_assert(IN_DIM1 % IN_DIM1_UNROLL == 0,
                "IN_DIM1 must be a multiple of IN_DIM1_UNROLL");
  static_assert(IN_DIM1 % OUT_DIM1_UNROLL == 0,
                "IN_DIM1 must be a multiple of OUT_DIM1_UNROLL");
  static_assert(IN_DIM2 % IN_DIM2_UNROLL == 0,
                "IN_DIM2_UNROLL must be a multiple of IN_DIM2");
  static_assert(IN_DIM2 % OUT_DIM2_UNROLL == 0,
                "OUT_DIM2_UNROLL must be a multiple of IN_DIM2");
  static_assert(IN_DIM2_UNROLL == OUT_DIM2_UNROLL,
                "IN_DIM2_UNROLL must be equal to OUT_DIM2_UNROLL");

  BandwidthAdjustDecreaseStreams() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_dim01 = 0, i_in_dim1_par = 0, i_dim2 = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output[OUT_DIM1_UNROLL];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < OUT_DIM1_UNROLL; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status =
          ActorStatus(depth, IN_DIM0 * IN_DIM1 * IN_DIM2 /
                                 (OUT_DIM2_UNROLL * OUT_DIM1_UNROLL));
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
  void run(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
           hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL]) {
    for (size_t i_dim01 = 0; i_dim01 < IN_DIM0 * IN_DIM1;
         i_dim01 += IN_DIM1_UNROLL) {
      for (size_t i_in_dim1_par = 0; i_in_dim1_par < IN_DIM1_UNROLL;
           i_in_dim1_par += OUT_DIM1_UNROLL) {
      BANDWIDTHADJUSTDECREASESTREAMS_RUN_LOOP:
        for (size_t i_dim2 = 0; i_dim2 < IN_DIM2; i_dim2 += OUT_DIM2_UNROLL) {
#pragma HLS pipeline II = 1
          BandwidthAdjustDecreaseStreams::pipeline_body(i_data, o_data,
                                                        i_in_dim1_par);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Check if there is data in the input streams considered in this step.
    bool firing_condition = true;
    for (size_t i_in_dim1_par = 0; i_in_dim1_par < OUT_DIM1_UNROLL;
         i_in_dim1_par++) {
      if (i_data[st.i_in_dim1_par + i_in_dim1_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      hls::stream<TOutputWord> instant_output_stream[OUT_DIM1_UNROLL];
      BandwidthAdjustDecreaseStreams::pipeline_body(
          i_data, instant_output_stream, st.i_in_dim1_par);

      // Update the loop iteration indexes for the next step.
      st.i_dim2 += OUT_DIM2_UNROLL;
      if (st.i_dim2 >= IN_DIM2) {
        st.i_dim2 = 0;
        st.i_in_dim1_par += OUT_DIM1_UNROLL;
      }
      if (st.i_in_dim1_par >= IN_DIM1_UNROLL) {
        st.i_in_dim1_par = 0;
        st.i_dim01 += IN_DIM1_UNROLL;
      }
      if (st.i_dim01 >= IN_DIM0 * IN_DIM1) {
        st.i_dim01 = 0;
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_out_dim1_par = 0; i_out_dim1_par < OUT_DIM1_UNROLL;
           ++i_out_dim1_par) {
        st.delayed_output[i_out_dim1_par].push(
            instant_output_stream[i_out_dim1_par].read(), true);
      }
    } else {
      // If no data is available, push empty outputs.
      for (size_t i_out_dim1_par = 0; i_out_dim1_par < OUT_DIM1_UNROLL;
           ++i_out_dim1_par) {
        st.delayed_output[i_out_dim1_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out_word;
    for (size_t i_out_dim1_par = 0; i_out_dim1_par < OUT_DIM1_UNROLL;
         i_out_dim1_par++) {
      if (st.delayed_output[i_out_dim1_par].pop(out_word)) {
        o_data[i_out_dim1_par].write(
            out_word); // Write the output to the stream.
      }
    }

    // Return the current actor status.
    return st.actor_status;
  }

private:
  static void pipeline_body(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
                            hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL],
                            size_t i_in_dim1_par) {
#pragma HLS inline
    // Input word to read data from the input stream.
    TInputWord in_word;
    TOutputWord out_word; // Output word to hold the results.
    Quantizer quantizer;  // Quantizer instance for quantization.

    for (size_t i_out_dim1_par = 0; i_out_dim1_par < OUT_DIM1_UNROLL;
         i_out_dim1_par++) {
      // Read the input data word from the input stream.
      in_word = i_data[i_in_dim1_par + i_out_dim1_par].read();

      for (size_t i_out_dim2_par = 0; i_out_dim2_par < OUT_DIM2_UNROLL;
           i_out_dim2_par++) {
        TInput in_data = in_word[i_out_dim2_par];

        // Quantize the input data.
        TOutput out_data = quantizer(in_data);

        // Store the quantized data in the output word.
        out_word[i_out_dim2_par] = out_data;
      }
      o_data[i_out_dim1_par].write(out_word);
    }
  }
};

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t IN_DIM0, size_t IN_DIM1,
          size_t IN_DIM2, size_t IN_DIM1_UNROLL, size_t OUT_DIM1_UNROLL,
          size_t IN_DIM2_UNROLL, size_t OUT_DIM2_UNROLL>
class BandwidthAdjustIncreaseWord {
public:
  static_assert(IN_DIM1_UNROLL > 0, "IN_DIM1_UNROLL must be greater than 0");
  static_assert(OUT_DIM1_UNROLL > 0, "OUT_DIM1_UNROLL must be greater than 0");
  static_assert(IN_DIM2_UNROLL < OUT_DIM2_UNROLL,
                "IN_DIM2_UNROLL must be less than OUT_DIM2_UNROLL");
  static_assert(OUT_DIM2_UNROLL % IN_DIM2_UNROLL == 0,
                "OUT_DIM2_UNROLL must be a multiple of IN_DIM2_UNROLL");
  static_assert(IN_DIM1 % IN_DIM1_UNROLL == 0,
                "IN_DIM1 must be a multiple of IN_DIM1_UNROLL");
  static_assert(IN_DIM1 % OUT_DIM1_UNROLL == 0,
                "IN_DIM1 must be a multiple of OUT_DIM1_UNROLL");
  static_assert(IN_DIM1_UNROLL == OUT_DIM1_UNROLL,
                "IN_DIM1_UNROLL must be equal to OUT_DIM1_UNROLL");
  static_assert(IN_DIM2 % IN_DIM2_UNROLL == 0,
                "IN_DIM2 must be a multiple of IN_DIM2_UNROLL");
  static_assert(IN_DIM2 % OUT_DIM2_UNROLL == 0,
                "IN_DIM2 must be a multiple of OUT_DIM2_UNROLL");

  BandwidthAdjustIncreaseWord() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_dim01 = 0, i_out_dim2_par = 0, i_dim2 = 0;

    // Output data buffer
    TOutputWord output_data[OUT_DIM1_UNROLL];

    PipelineDelayBuffer<TOutputWord> delayed_output[OUT_DIM1_UNROLL];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < OUT_DIM1_UNROLL; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status = ActorStatus(depth, IN_DIM0 * IN_DIM1 * IN_DIM2 /
                                            (IN_DIM2_UNROLL * IN_DIM1_UNROLL));
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
  void run(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
           hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL]) {
    TOutputWord
        output_data[OUT_DIM1_UNROLL]; // Output word to hold the results.
    for (size_t i_dim01 = 0; i_dim01 < IN_DIM0 * IN_DIM1;
         i_dim01 += IN_DIM1_UNROLL) {
      for (size_t i_dim2 = 0; i_dim2 < IN_DIM2; i_dim2 += OUT_DIM2_UNROLL) {
      BANDWIDTHADJUSTINCREASEDIM2_RUN_LOOP:
        for (size_t i_out_dim2_par = 0; i_out_dim2_par < OUT_DIM2_UNROLL;
             i_out_dim2_par += IN_DIM2_UNROLL) {
#pragma HLS loop_flatten
#pragma HLS pipeline II = 1
          BandwidthAdjustIncreaseWord::pipeline_body(
              i_data, o_data, output_data, i_out_dim2_par);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_in_dim1_par = 0; i_in_dim1_par < IN_DIM1_UNROLL;
         i_in_dim1_par++) {
      if (i_data[i_in_dim1_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> instant_output_stream[OUT_DIM1_UNROLL];
      BandwidthAdjustIncreaseWord::pipeline_body(
          i_data, instant_output_stream, st.output_data, st.i_out_dim2_par);

      // Update the loop iteration indexes for the next step.
      st.i_out_dim2_par += IN_DIM2_UNROLL;
      if (st.i_out_dim2_par >= OUT_DIM2_UNROLL) {
        st.i_out_dim2_par = 0;
        st.i_dim2 += OUT_DIM2_UNROLL;
      }
      if (st.i_dim2 >= IN_DIM2) {
        st.i_dim2 = 0;
        st.i_dim01 += IN_DIM1_UNROLL;
      }
      if (st.i_dim01 >= IN_DIM0 * IN_DIM1) {
        st.i_dim01 = 0;
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_out_dim1_par = 0; i_out_dim1_par < OUT_DIM1_UNROLL;
           ++i_out_dim1_par) {
        if (!instant_output_stream[i_out_dim1_par].empty()) {
          st.delayed_output[i_out_dim1_par].push(
              instant_output_stream[i_out_dim1_par].read(), true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output[i_out_dim1_par].push(TOutputWord(), false);
        }
      }
    } else {
      // If no data is available, push empty outputs.
      for (size_t i_out_dim1_par = 0; i_out_dim1_par < OUT_DIM1_UNROLL;
           ++i_out_dim1_par) {
        st.delayed_output[i_out_dim1_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out_word;
    for (size_t i_out_dim1_par = 0; i_out_dim1_par < OUT_DIM1_UNROLL;
         i_out_dim1_par++) {
      if (st.delayed_output[i_out_dim1_par].pop(out_word)) {
        o_data[i_out_dim1_par].write(out_word);
      }
    }

    // Return the current firing iteration index.
    return st.actor_status;
  }

private:
  static void pipeline_body(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
                            hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL],
                            TOutputWord out_word[OUT_DIM1_UNROLL],
                            size_t i_out_dim2_par) {
#pragma HLS inline
    TInputWord in_word;  // Input word to read data from the input stream.
    Quantizer quantizer; // Quantizer instance for quantization.

    for (size_t i_in_dim1_par = 0; i_in_dim1_par < IN_DIM1_UNROLL;
         i_in_dim1_par++) {
      // Read the input data word from the input stream.
      in_word = i_data[i_in_dim1_par].read();

      for (size_t i_in_dim2_par = 0; i_in_dim2_par < IN_DIM2_UNROLL;
           i_in_dim2_par++) {
        TInput in_data = in_word[i_in_dim2_par];

        // Quantize the input data.
        TOutput out_data = quantizer(in_data);

        // Store the quantized data in the output word.
        out_word[i_in_dim1_par][i_out_dim2_par + i_in_dim2_par] = out_data;
      }

      // If we have processed all output channels, write the output word to
      // the output stream.
      if (i_out_dim2_par == OUT_DIM2_UNROLL - IN_DIM2_UNROLL) {
        o_data[i_in_dim1_par].write(out_word[i_in_dim1_par]);
      }
    }
  }
};

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t IN_DIM0, size_t IN_DIM1,
          size_t IN_DIM2, size_t IN_DIM1_UNROLL, size_t OUT_DIM1_UNROLL,
          size_t IN_DIM2_UNROLL, size_t OUT_DIM2_UNROLL>
class BandwidthAdjustDecreaseWord {
public:
  static_assert(IN_DIM1_UNROLL > 0, "IN_DIM1_UNROLL must be greater than 0");
  static_assert(OUT_DIM1_UNROLL > 0, "OUT_DIM1_UNROLL must be greater than 0");
  static_assert(IN_DIM2_UNROLL > OUT_DIM2_UNROLL,
                "OUT_DIM2_UNROLL must be less than IN_DIM2_UNROLL");
  static_assert(IN_DIM2_UNROLL % OUT_DIM2_UNROLL == 0,
                "IN_DIM2_UNROLL must be a multiple of OUT_DIM2_UNROLL");
  static_assert(IN_DIM1 % IN_DIM1_UNROLL == 0,
                "IN_DIM1 must be a multiple of IN_DIM1_UNROLL");
  static_assert(IN_DIM1 % OUT_DIM1_UNROLL == 0,
                "IN_DIM1 must be a multiple of OUT_DIM1_UNROLL");
  static_assert(IN_DIM1_UNROLL == OUT_DIM1_UNROLL,
                "IN_DIM1_UNROLL must be equal to OUT_DIM1_UNROLL");
  static_assert(IN_DIM2 % IN_DIM2_UNROLL == 0,
                "IN_DIM2 must be a multiple of IN_DIM2_UNROLL");
  static_assert(IN_DIM2 % OUT_DIM2_UNROLL == 0,
                "IN_DIM2 must be a multiple of OUT_DIM2_UNROLL");

  BandwidthAdjustDecreaseWord() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_dim01 = 0, i_in_dim2_par = 0, i_dim2 = 0;

    // Input data buffer
    TInputWord input_data[IN_DIM1_UNROLL];

    PipelineDelayBuffer<TOutputWord> delayed_output[OUT_DIM1_UNROLL];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < OUT_DIM1_UNROLL; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status = ActorStatus(depth, IN_DIM0 * IN_DIM1 * IN_DIM2 /
                                            (OUT_DIM2_UNROLL * IN_DIM1_UNROLL));
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
  void run(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
           hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL]) {
    TInputWord input_data[IN_DIM1_UNROLL]; // Input word to hold the data read.
    for (size_t i_dim01 = 0; i_dim01 < IN_DIM0 * IN_DIM1;
         i_dim01 += IN_DIM1_UNROLL) {
      for (size_t i_dim2 = 0; i_dim2 < IN_DIM2; i_dim2 += IN_DIM2_UNROLL) {
      BANDWIDTHADJUSTDECREASEDIM2_RUN_LOOP:
        for (size_t i_in_dim2_par = 0; i_in_dim2_par < IN_DIM2_UNROLL;
             i_in_dim2_par += OUT_DIM2_UNROLL) {
#pragma HLS loop_flatten
#pragma HLS pipeline II = 1
          BandwidthAdjustDecreaseWord::pipeline_body(i_data, o_data, input_data,
                                                     i_in_dim2_par);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    if (st.i_in_dim2_par == 0) {
      for (size_t i_in_dim1_par = 0; i_in_dim1_par < IN_DIM1_UNROLL;
           i_in_dim1_par++) {
        if (i_data[i_in_dim1_par].empty()) {
          firing_condition = false;
        }
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> instant_output_stream[OUT_DIM1_UNROLL];
      BandwidthAdjustDecreaseWord::pipeline_body(
          i_data, instant_output_stream, st.input_data, st.i_in_dim2_par);

      // Update the loop iteration indexes for the next step.
      st.i_in_dim2_par += OUT_DIM2_UNROLL;
      if (st.i_in_dim2_par >= IN_DIM2_UNROLL) {
        st.i_in_dim2_par = 0;
        st.i_dim2 += IN_DIM2_UNROLL;
      }
      if (st.i_dim2 >= IN_DIM2) {
        st.i_dim2 = 0;
        st.i_dim01 += IN_DIM1_UNROLL;
      }
      if (st.i_dim01 >= IN_DIM0 * IN_DIM1) {
        st.i_dim01 = 0;
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_out_dim1_par = 0; i_out_dim1_par < OUT_DIM1_UNROLL;
           ++i_out_dim1_par) {
        st.delayed_output[i_out_dim1_par].push(
            instant_output_stream[i_out_dim1_par].read(), true);
      }
    } else {
      // If no data is available, push empty outputs.
      for (size_t i_out_dim1_par = 0; i_out_dim1_par < OUT_DIM1_UNROLL;
           ++i_out_dim1_par) {
        st.delayed_output[i_out_dim1_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out_word;
    for (size_t i_out_dim1_par = 0; i_out_dim1_par < OUT_DIM1_UNROLL;
         i_out_dim1_par++) {
      if (st.delayed_output[i_out_dim1_par].pop(out_word)) {
        o_data[i_out_dim1_par].write(out_word);
      }
    }

    // Return the current actor status.
    return st.actor_status;
  }

private:
  static void pipeline_body(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
                            hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL],
                            TInputWord in_word[IN_DIM1_UNROLL],
                            size_t i_in_dim2_par) {
#pragma HLS inline
    TOutputWord out_word; // Output word to read data from the input stream.
    Quantizer quantizer;  // Quantizer instance for quantization.

    for (size_t i_in_dim1_par = 0; i_in_dim1_par < IN_DIM1_UNROLL;
         i_in_dim1_par++) {
      // Read the input data word from the input stream.
      if (i_in_dim2_par == 0) {
        in_word[i_in_dim1_par] = i_data[i_in_dim1_par].read();
      }

      for (size_t i_out_dim2_par = 0; i_out_dim2_par < OUT_DIM2_UNROLL;
           i_out_dim2_par++) {
        TInput in_data = in_word[i_in_dim1_par][i_in_dim2_par + i_out_dim2_par];

        // Quantize the input data.
        TOutput out_data = quantizer(in_data);

        // Store the quantized data in the output word.
        out_word[i_out_dim2_par] = out_data;
      }

      o_data[i_in_dim1_par].write(out_word);
    }
  }
};