#pragma once
#include "StreamingReLU.hpp"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>

template <typename TInputWordA, typename TInputA, typename TInputWordB,
          typename TInputB, typename TOutputWord, typename TOutput,
          typename TMul, typename OutputTransform, size_t DIM0,
          size_t DIM1, size_t DIM2, size_t DIM1_UNROLL, size_t DIM2_UNROLL>
class StreamingMul {
  static_assert(DIM1 % DIM1_UNROLL == 0, "DIM1 must be divisible by DIM1_UNROLL");
  static_assert(DIM2 % DIM2_UNROLL == 0, "DIM2 must be divisible by DIM2_UNROLL");

public:
  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  StreamingMul() = default;

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWordA> i_dataA[DIM1_UNROLL],
           hls::stream<TInputWordB> i_dataB[DIM1_UNROLL],
           hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
  STREAMINGADD_RUN_LOOP:
    for (size_t i_dim012 = 0;
         i_dim012 < DIM0 * DIM1 * DIM2 / (DIM2_UNROLL * DIM1_UNROLL);
         i_dim012++) {
#pragma HLS PIPELINE II = 1
      StreamingMul::pipeline_body(i_dataA, i_dataB, o_data);
    }
  }

  ActorStatus step(hls::stream<TInputWordA> i_dataA[DIM1_UNROLL],
                   hls::stream<TInputWordB> i_dataB[DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (i_dataA[i_dim1_par].empty() || i_dataB[i_dim1_par].empty()) {
        firing_condition = false;
        break;
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> o_data_instant[DIM1_UNROLL];
      StreamingMul::pipeline_body(i_dataA, i_dataB, o_data_instant);

      // Update iterators
      st.i_dim012 = (st.i_dim012 + 1) %
                    (DIM0 * DIM1 * DIM2 / (DIM2_UNROLL * DIM1_UNROLL));

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Mul the output to the delayed output buffers
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
    size_t i_dim012 = 0;

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
          ActorStatus(depth, DIM0 * DIM1 * DIM2 / (DIM2_UNROLL * DIM1_UNROLL));
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }
  static void pipeline_body(hls::stream<TInputWordA> i_dataA[DIM1_UNROLL],
                            hls::stream<TInputWordB> i_dataB[DIM1_UNROLL],
                            hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
#pragma HLS inline
    TInputWordA inputA_word;
    TInputWordB inputB_word;
    TOutputWord output_word;
    OutputTransform output_transform;

    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      // Read the input data structure from the input streams.
      inputA_word = i_dataA[i_dim1_par].read();
      inputB_word = i_dataB[i_dim1_par].read();

      for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL; i_dim2_par++) {
        // Extract the data for the current pixel channel.
        TInputA inputA_data = inputA_word[i_dim2_par];
        TInputB inputB_data = inputB_word[i_dim2_par];

        // Perform the multiplication.
        TMul s_mul = inputA_data * inputB_data;
        TOutput output_data = output_transform(s_mul);

        // Store the quantized data in the output structure.
        output_word[i_dim2_par] = output_data;
      }
      o_data[i_dim1_par].write(output_word);
    }
  }
};
