#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>
#include <cassert>

template <typename TInputWordA, typename TInputA, typename TInputB,
          typename TOutputWord, typename TOutput, typename TMul,
          typename Activation, typename Quantizer, size_t IN_HEIGHT,
          size_t IN_WIDTH, size_t IN_CH, size_t W_PAR, size_t CH_PAR>
class StreamingConstMul {

private:
  struct StepState {
    // Loop iteration indexes.
    size_t i = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output[W_PAR];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < W_PAR; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
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

public:
  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

public:
  StreamingConstMul() = default;

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWordA> i_dataA[W_PAR],
           const TInputB constant_input,
           hls::stream<TOutputWord> o_data[W_PAR]) {
  STREAMINGADD_RUN_LOOP:
    for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH * IN_CH / (CH_PAR * W_PAR);
         i++) {
#pragma HLS PIPELINE II = 1
      StreamingConstMul::pipeline_body(i_dataA, constant_input, o_data);
    }
  }

  ActorStatus step(hls::stream<TInputWordA> i_dataA[W_PAR],
                   const TInputB constant_input,
                   hls::stream<TOutputWord> o_data[W_PAR]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      if (i_dataA[i_w_par].empty()) {
        firing_condition = false;
        break;
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> o_data_instant[W_PAR];
      StreamingConstMul::pipeline_body(i_dataA, constant_input, o_data_instant);

      // Update iterators
      st.i = (st.i + 1) % (IN_HEIGHT * IN_WIDTH * IN_CH / (CH_PAR * W_PAR));

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Mul the output to the delayed output buffers
      TOutputWord out_value;
      for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
        out_value = o_data_instant[i_w_par].read();
        st.delayed_output[i_w_par].push(out_value, true);
      }
    } else {
      // If not firing, push invalid data to maintain pipeline timing
      for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
        st.delayed_output[i_w_par].push(TOutputWord(), false);
      }
    }

    // Advance the actor status
    st.actor_status.advance();

    // Read from the delayed output buffers
    TOutputWord out_value;
    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      if (st.delayed_output[i_w_par].pop(out_value)) {
        o_data[i_w_par].write(out_value);
      }
    }

    return st.actor_status;
  }

private:
  static void pipeline_body(hls::stream<TInputWordA> i_dataA[W_PAR],
                            const TInputB constant_input,
                            hls::stream<TOutputWord> o_data[W_PAR]) {
#pragma HLS inline
    TInputWordA s_inputA_struct;
    TOutputWord s_output_struct;
    Quantizer quantizer;
    Activation activation;

    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      // Read the input data structure from the input streams.
      s_inputA_struct = i_dataA[i_w_par].read();

      for (size_t i_ch_par = 0; i_ch_par < CH_PAR; i_ch_par++) {
        // Extract the data for the current pixel channel.
        TInputA s_inputA_data = s_inputA_struct[i_ch_par];

        // Perform the multiplication.
        TMul s_mul = s_inputA_data * constant_input;
        // Apply activation function.
        s_mul = activation(s_mul);

        // Quantize the sum.
        TOutput s_output_data = quantizer(s_mul);

        // Store the quantized data in the output structure.
        s_output_struct[i_ch_par] = s_output_data;
      }
      o_data[i_w_par].write(s_output_struct);
    }
  }
};