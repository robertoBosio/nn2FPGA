#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>

/** @brief A class for concatenating streams along the second dimension.
 *
 * @tparam TInputWord The type of the input stream words.
 * @tparam TInput The type of the input data elements.
 * @tparam TOutputWord The type of the output stream words.
 * @tparam TOutput The type of the output data elements.
 * @tparam Quantizer A functor type for quantizing the output data.
 * @tparam IN_DIM0 The size of the first dimension of the input tensors.
 * @tparam IN_DIM1 The size of the second dimension of the input tensors.
 * @tparam IN_DIM2_A The size of the third dimension of the first input tensor.
 * @tparam IN_DIM2_B The size of the third dimension of the second input tensor.
 * @tparam DIM1_UNROLL The unrolling factor for the second dimension.
 * @tparam DIM2_UNROLL The unrolling factor for the third dimension.
 */

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t IN_DIM0, size_t IN_DIM1,
          size_t IN_DIM2_A, size_t IN_DIM2_B, size_t DIM1_UNROLL,
          size_t DIM2_UNROLL>
class StreamingConcatDim2 {
  static_assert(IN_DIM1 % DIM1_UNROLL == 0, "DIM1_UNROLL must divide IN_DIM1");
  static_assert(IN_DIM2_A % DIM2_UNROLL == 0,
                "DIM2_UNROLL must divide IN_DIM2_A");
  static_assert(IN_DIM2_B % DIM2_UNROLL == 0,
                "DIM2_UNROLL must divide IN_DIM2_B");

public:
  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  StreamingConcatDim2() = default;

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_dataA[DIM1_UNROLL],
           hls::stream<TInputWord> i_dataB[DIM1_UNROLL],
           hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
    for (size_t i_dim01 = 0; i_dim01 < IN_DIM0 * IN_DIM1 / DIM1_UNROLL;
         i_dim01++) {
    STREAMINGCONCATCHANNEL_RUN_LOOP:
      for (size_t i_dim2 = 0; i_dim2 < (IN_DIM2_A + IN_DIM2_B) / DIM2_UNROLL;
           i_dim2++) {
#pragma HLS PIPELINE II = 1
        StreamingConcatDim2::pipeline_body(i_dataA, i_dataB, o_data, i_dim2);
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_dataA[DIM1_UNROLL],
                   hls::stream<TInputWord> i_dataB[DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (st.i_dim2 < IN_DIM2_A / DIM2_UNROLL) {
        if (i_dataA[i_dim1_par].empty()) {
          firing_condition = false;
          break;
        }
      } else {
        if (i_dataB[i_dim1_par].empty()) {
          firing_condition = false;
          break;
        }
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> o_data_instant[DIM1_UNROLL];
      StreamingConcatDim2::pipeline_body(i_dataA, i_dataB, o_data_instant,
                                         st.i_dim2);

      // Update iterators
      st.i_dim2++;
      if (st.i_dim2 >= (IN_DIM2_A + IN_DIM2_B) / DIM2_UNROLL) {
        st.i_dim2 = 0;
        st.i_dim01++;
        if (st.i_dim01 >= IN_DIM0 * IN_DIM1 / DIM1_UNROLL) {
          st.i_dim01 = 0;
        }
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Add the output to the delayed output buffers
      TOutputWord out_value;
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        out_value = o_data_instant[i_dim1_par].read();
        st.delayed_output[i_dim1_par].push(out_value, true);
      }
    } else {
      // If not firing, push invalid data to maintain pipeline timing
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        st.delayed_output[i_dim1_par].push(TOutputWord(), false);
      }
    }

    // Advance the actor status
    st.actor_status.advance();

    // Read from the delayed output buffers
    TOutputWord out_value;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (st.delayed_output[i_dim1_par].pop(out_value)) {
        o_data[i_dim1_par].write(out_value);
      }
    }

    return st.actor_status;
  }

private:
  struct StepState {
    // Loop iteration indexes.
    size_t i_dim01 = 0, i_dim2 = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output[DIM1_UNROLL];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < DIM1_UNROLL; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status =
          ActorStatus(depth, IN_DIM0 * IN_DIM1 * (IN_DIM2_A + IN_DIM2_B) /
                                 (DIM2_UNROLL * DIM1_UNROLL));
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

  static void pipeline_body(hls::stream<TInputWord> i_dataA[DIM1_UNROLL],
                            hls::stream<TInputWord> i_dataB[DIM1_UNROLL],
                            hls::stream<TOutputWord> o_data[DIM1_UNROLL],
                            size_t i_dim2) {
#pragma HLS inline
    TInputWord input_word;
    TOutputWord output_word;
    Quantizer quantizer;

    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      // Read the input data structure from the input streams.
      if (i_dim2 < IN_DIM2_A / DIM2_UNROLL) {
        input_word = i_dataA[i_dim1_par].read();
      } else {
        input_word = i_dataB[i_dim1_par].read();
      }

      for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL; i_dim2_par++) {
        // Extract the data for the current pixel channel.
        TInput input_data = input_word[i_dim2_par];

        // Quantize the sum.
        TOutput output_data = quantizer(input_data);

        // Store the quantized data in the output structure.
        output_word[i_dim2_par] = output_data;
      }
      o_data[i_dim1_par].write(output_word);
    }
  }
};

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t IN_DIM0_A,
          size_t IN_DIM0_B, size_t IN_DIM1, size_t IN_DIM2, size_t DIM1_UNROLL,
          size_t DIM2_UNROLL>
class StreamingConcatDim0 {
  static_assert(IN_DIM1 % DIM1_UNROLL == 0, "DIM1_UNROLL must divide IN_DIM1");
  static_assert(IN_DIM2 % DIM2_UNROLL == 0, "DIM2_UNROLL must divide IN_DIM2");

public:
  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  StreamingConcatDim0() = default;

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_dataA[DIM1_UNROLL],
           hls::stream<TInputWord> i_dataB[DIM1_UNROLL],
           hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
    for (size_t i_dim0 = 0; i_dim0 < (IN_DIM0_A + IN_DIM0_B); i_dim0++) {
      for (size_t i_dim1 = 0; i_dim1 < IN_DIM1 / DIM1_UNROLL; i_dim1++) {
      STREAMINGCONCATDIM0_RUN_LOOP:
        for (size_t i_dim2 = 0; i_dim2 < IN_DIM2 / DIM2_UNROLL; i_dim2++) {
#pragma HLS PIPELINE II = 1
          StreamingConcatDim0::pipeline_body(i_dataA, i_dataB, o_data, i_dim0);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_dataA[DIM1_UNROLL],
                   hls::stream<TInputWord> i_dataB[DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (st.i_dim0 < IN_DIM0_A) {
        if (i_dataA[i_dim1_par].empty()) {
          firing_condition = false;
          break;
        }
      } else {
        if (i_dataB[i_dim1_par].empty()) {
          firing_condition = false;
          break;
        }
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> o_data_instant[DIM1_UNROLL];
      StreamingConcatDim0::pipeline_body(i_dataA, i_dataB, o_data_instant,
                                         st.i_dim0);

      // Update iterators
      st.i_dim2++;
      if (st.i_dim2 >= IN_DIM2 / DIM2_UNROLL) {
        st.i_dim1 = 0;
        st.i_dim0++;
      }
      if (st.i_dim0 >= IN_DIM1 / DIM1_UNROLL) {
        st.i_dim0 = 0;
        st.i_dim1++;
      }
      if (st.i_dim1 >= (IN_DIM0_A + IN_DIM0_B)) {
        st.i_dim1 = 0;
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Add the output to the delayed output buffers
      TOutputWord out_value;
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        out_value = o_data_instant[i_dim1_par].read();
        st.delayed_output[i_dim1_par].push(out_value, true);
      }
    } else {
      // If not firing, push invalid data to maintain pipeline timing
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        st.delayed_output[i_dim1_par].push(TOutputWord(), false);
      }
    }

    // Advance the actor status
    st.actor_status.advance();

    // Read from the delayed output buffers
    TOutputWord out_value;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (st.delayed_output[i_dim1_par].pop(out_value)) {
        o_data[i_dim1_par].write(out_value);
      }
    }

    return st.actor_status;
  }

private:
  struct StepState {
    // Loop iteration indexes.
    size_t i_dim0 = 0, i_dim1 = 0, i_dim2 = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output[DIM1_UNROLL];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < DIM1_UNROLL; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status =
          ActorStatus(depth, (IN_DIM0_A + IN_DIM0_B) * IN_DIM1 * IN_DIM2 /
                                 (DIM2_UNROLL * DIM1_UNROLL));
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

  static void pipeline_body(hls::stream<TInputWord> i_dataA[DIM1_UNROLL],
                            hls::stream<TInputWord> i_dataB[DIM1_UNROLL],
                            hls::stream<TOutputWord> o_data[DIM1_UNROLL],
                            size_t i_dim0) {
#pragma HLS inline
    TInputWord input_word;
    TOutputWord output_word;
    Quantizer quantizer;

    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      // Read the input data structure from the input streams.
      if (i_dim0 < IN_DIM0_A) {
        input_word = i_dataA[i_dim1_par].read();
      } else {
        input_word = i_dataB[i_dim1_par].read();
      }

      for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL; i_dim2_par++) {
        // Extract the data for the current pixel channel.
        TInput input_data = input_word[i_dim2_par];

        // Quantize the sum.
        TOutput output_data = quantizer(input_data);

        // Store the quantized data in the output structure.
        output_word[i_dim2_par] = output_data;
      }
      o_data[i_dim1_par].write(output_word);
    }
  }
};

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t IN_DIM0,
          size_t IN_DIM1_A, size_t IN_DIM1_B, size_t IN_DIM2,
          size_t DIM1_UNROLL, size_t DIM2_UNROLL>
class StreamingConcatDim1 {
  static_assert(IN_DIM1_A % DIM1_UNROLL == 0,
                "DIM1_UNROLL must divide IN_DIM1_A");
  static_assert(IN_DIM1_B % DIM1_UNROLL == 0,
                "DIM1_UNROLL must divide IN_DIM1_B");
  static_assert(IN_DIM2 % DIM2_UNROLL == 0, "DIM2_UNROLL must divide IN_DIM2");

public:
  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  StreamingConcatDim1() = default;

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_dataA[DIM1_UNROLL],
           hls::stream<TInputWord> i_dataB[DIM1_UNROLL],
           hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
    for (size_t i_dim0 = 0; i_dim0 < IN_DIM0; i_dim0++) {
      for (size_t i_dim1 = 0; i_dim1 < (IN_DIM1_A + IN_DIM1_B);
           i_dim1 += DIM1_UNROLL) {
      STREAMINGCONCATWIDTH_RUN_LOOP:
        for (size_t i_dim2 = 0; i_dim2 < IN_DIM2 / DIM2_UNROLL; i_dim2++) {
#pragma HLS PIPELINE II = 1
          StreamingConcatDim1::pipeline_body(i_dataA, i_dataB, o_data, i_dim1);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_dataA[DIM1_UNROLL],
                   hls::stream<TInputWord> i_dataB[DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (st.i_dim1 < IN_DIM1_A) {
        if (i_dataA[i_dim1_par].empty()) {
          firing_condition = false;
          break;
        }
      } else {
        if (i_dataB[i_dim1_par].empty()) {
          firing_condition = false;
          break;
        }
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> o_data_instant[DIM1_UNROLL];
      StreamingConcatDim1::pipeline_body(i_dataA, i_dataB, o_data_instant,
                                         st.i_dim1);

      // Update iterators
      st.i_dim2++;
      if (st.i_dim2 >= IN_DIM2 / DIM2_UNROLL) {
        st.i_dim2 = 0;
        st.i_dim1 += DIM1_UNROLL;
      }
      if (st.i_dim1 >= (IN_DIM1_A + IN_DIM1_B)) {
        st.i_dim1 = 0;
        st.i_dim0++;
      }
      if (st.i_dim0 >= IN_DIM0) {
        st.i_dim0 = 0;
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Add the output to the delayed output buffers
      TOutputWord out_value;
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        out_value = o_data_instant[i_dim1_par].read();
        st.delayed_output[i_dim1_par].push(out_value, true);
      }
    } else {
      // If not firing, push invalid data to maintain pipeline timing
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        st.delayed_output[i_dim1_par].push(TOutputWord(), false);
      }
    }

    // Advance the actor status
    st.actor_status.advance();

    // Read from the delayed output buffers
    TOutputWord out_value;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (st.delayed_output[i_dim1_par].pop(out_value)) {
        o_data[i_dim1_par].write(out_value);
      }
    }

    return st.actor_status;
  }

private:
  struct StepState {
    // Loop iteration indexes.
    size_t i_dim0 = 0, i_dim1 = 0, i_dim2 = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output[DIM1_UNROLL];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < DIM1_UNROLL; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status =
          ActorStatus(depth, IN_DIM0 * (IN_DIM1_A + IN_DIM1_B) * IN_DIM2 /
                                 (DIM2_UNROLL * DIM1_UNROLL));
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

  static void pipeline_body(hls::stream<TInputWord> i_dataA[DIM1_UNROLL],
                            hls::stream<TInputWord> i_dataB[DIM1_UNROLL],
                            hls::stream<TOutputWord> o_data[DIM1_UNROLL],
                            size_t i_dim1) {
#pragma HLS inline
    TInputWord input_word;
    TOutputWord output_word;
    Quantizer quantizer;

    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      // Read the input data structure from the input streams.
      if (i_dim1 < IN_DIM1_A) {
        input_word = i_dataA[i_dim1_par].read();
      } else {
        input_word = i_dataB[i_dim1_par].read();
      }

      for (size_t i_ch_par = 0; i_ch_par < DIM2_UNROLL; i_ch_par++) {
        // Extract the data for the current pixel channel.
        TInput input_data = input_word[i_ch_par];

        // Quantize the sum.
        TOutput output_data = quantizer(input_data);

        // Store the quantized data in the output structure.
        output_word[i_ch_par] = output_data;
      }
      o_data[i_dim1_par].write(output_word);
    }
  }
};