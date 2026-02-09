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
          typename TOutput, size_t LUT_SIZE, size_t IN_HEIGHT, size_t IN_WIDTH,
          size_t IN_CH, size_t CH_PAR, size_t W_PAR>
class StreamingLUT {
public:
  static_assert(IN_CH % CH_PAR == 0, "IN_CH must be a multiple of CH_PAR");
  static_assert(CH_PAR > 0, "CH_PAR must be greater than 0");
  static_assert(IN_HEIGHT > 0 && IN_WIDTH > 0,
                "IN_HEIGHT and IN_WIDTH must be greater than 0");

  StreamingLUT() = default;

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
           const TOutput LUTmem[LUT_SIZE],
           hls::stream<TOutputWord> o_data[W_PAR]) {
    // Loop through the input height and width.
    for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH / W_PAR; i_hw++) {
    STREAMINGLUT_RUN_LOOP:
      for (size_t i_ch = 0; i_ch < IN_CH / CH_PAR; i_ch++) {
#pragma HLS pipeline II = 1
        StreamingLUT::pipeline_body(i_data, LUTmem, o_data);
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[W_PAR],
                   const TOutput LUTmem[LUT_SIZE],
                   hls::stream<TOutputWord> o_data[W_PAR]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;

    // Check non empty input streams.
    for (size_t i_in_stream = 0; i_in_stream < W_PAR; i_in_stream++) {
      if (i_data[i_in_stream].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      hls::stream<TOutputWord> instant_output_stream[W_PAR];
      StreamingLUT::pipeline_body(i_data, LUTmem, instant_output_stream);

      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t w_par = 0; w_par < W_PAR; w_par++) {
        st.delayed_output[w_par].push(instant_output_stream[w_par].read(),
                                      true);
      }

      // Update the counters.
      st.i_ch++;
      if (st.i_ch >= IN_CH / CH_PAR) {
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
      for (size_t i_w_par = 0; i_w_par < W_PAR; ++i_w_par) {
        st.delayed_output[i_w_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      if (st.delayed_output[i_w_par].pop(out)) {
        o_data[i_w_par].write(out);
      }
    }

    // Return the current actor status.
    return st.actor_status;
  }

private:
  static void pipeline_body(hls::stream<TInputWord> i_data[W_PAR],
                            const TOutput LUTmem[LUT_SIZE],
                            hls::stream<TOutputWord> o_data[W_PAR]) {
#pragma HLS inline
    for (size_t w_par = 0; w_par < W_PAR; w_par++) {
      TInputWord in_word = i_data[w_par].read();
      TOutputWord out_word;
      for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++) {
        ap_uint<TInput::width> address =
            in_word[ch_par].range(TInput::width - 1, 0);
        TOutput out_value = LUTmem[address];
        out_word[ch_par] = out_value;
      }
      o_data[w_par].write(out_word);
    }
  }
};
