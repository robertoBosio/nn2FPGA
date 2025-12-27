#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>
#include <cassert>

template <typename TInputWord, typename TInput, typename TOutputWord, typename TOutput,
          typename Quantizer, size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH_A,
          size_t IN_CH_B, size_t W_PAR, size_t CH_PAR>
class StreamingConcat {

  struct StepState {
    // Loop iteration indexes.
    size_t i_hw = 0, i_ch = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output[W_PAR];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < W_PAR; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status = ActorStatus(
          depth, IN_HEIGHT * IN_WIDTH * (IN_CH_A + IN_CH_B) / (CH_PAR * W_PAR));
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

  StreamingConcat() = default;

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_dataA[W_PAR],
           hls::stream<TInputWord> i_dataB[W_PAR],
           hls::stream<TOutputWord> o_data[W_PAR]) {
  STREAMINGADD_RUN_LOOP:
    for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH / W_PAR; i_hw++) {
      for (size_t i_ch = 0; i_ch < (IN_CH_A + IN_CH_B) / CH_PAR; i_ch++) {
#pragma HLS PIPELINE II = 1
        StreamingConcat::pipeline_body(i_dataA, i_dataB, o_data, i_ch);
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_dataA[W_PAR],
                   hls::stream<TInputWord> i_dataB[W_PAR],
                   hls::stream<TOutputWord> o_data[W_PAR]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      if (st.i_ch < IN_CH_A / CH_PAR) {
        if (i_dataA[i_w_par].empty()) {
          firing_condition = false;
          break;
        }
      } else {
        if (i_dataB[i_w_par].empty()) {
          firing_condition = false;
          break;
        }
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> o_data_instant[W_PAR];
      StreamingConcat::pipeline_body(i_dataA, i_dataB,  o_data_instant, st.i_ch);

      // Update iterators
      st.i_ch++;
      if (st.i_ch >= (IN_CH_A + IN_CH_B) / CH_PAR) {
        st.i_ch = 0;
        st.i_hw++;
        if (st.i_hw >= IN_HEIGHT * IN_WIDTH / W_PAR) {
          st.i_hw = 0;
        }
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Add the output to the delayed output buffers
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
  static void pipeline_body(hls::stream<TInputWord> i_dataA[W_PAR],
                            hls::stream<TInputWord> i_dataB[W_PAR],
                            hls::stream<TOutputWord> o_data[W_PAR], size_t i_ch) {
#pragma HLS inline
    TInputWord s_input_struct;
    TOutputWord s_output_struct;
    Quantizer quantizer;

    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      // Read the input data structure from the input streams.
      if (i_ch < IN_CH_A / CH_PAR) {
        s_input_struct = i_dataA[i_w_par].read();
      } else {
        s_input_struct = i_dataB[i_w_par].read();
      }

      for (size_t i_ch_par = 0; i_ch_par < CH_PAR; i_ch_par++) {
        // Extract the data for the current pixel channel.
        TInput s_input_data = s_input_struct[i_ch_par];

        // Quantize the sum.
        TOutput s_output_data = quantizer(s_input_data);

        // Store the quantized data in the output structure.
        s_output_struct[i_ch_par] = s_output_data;
      }
      o_data[i_w_par].write(s_output_struct);
    }
  }
};