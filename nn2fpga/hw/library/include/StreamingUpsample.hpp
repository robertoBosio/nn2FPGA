#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>

template <typename TInputWord, typename TOutputWord, typename Quantizer,
          size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH, size_t OUT_HEIGHT,
          size_t OUT_WIDTH, size_t SCALE_FACTOR, size_t CH_PAR, size_t IN_W_PAR,
          size_t OUT_W_PAR>
class StreamingUpsample {
public:
  static_assert(IN_CH % CH_PAR == 0, "IN_CH must be a multiple of CH_PAR");
  static_assert(CH_PAR > 0, "CH_PAR must be greater than 0");
  static_assert(IN_W_PAR > 0, "IN_W_PAR must be greater than 0");
  static_assert(IN_WIDTH % IN_W_PAR == 0,
                "IN_WIDTH must be a multiple of IN_W_PAR");
  static_assert(OUT_W_PAR > 0, "OUT_W_PAR must be greater than 0");
  static_assert(OUT_WIDTH % OUT_W_PAR == 0,
                "OUT_WIDTH must be a multiple of OUT_W_PAR");
  static_assert(IN_HEIGHT > 0 && IN_WIDTH > 0,
                "IN_HEIGHT and IN_WIDTH must be greater than 0");
  static_assert(SCALE_FACTOR > 1, "SCALE_FACTOR must be greater than 1");
  static_assert(OUT_HEIGHT == IN_HEIGHT * SCALE_FACTOR,
                "OUT_HEIGHT must be equal to IN_HEIGHT * SCALE_FACTOR");
  static_assert(OUT_WIDTH == IN_WIDTH * SCALE_FACTOR,
                "OUT_WIDTH must be equal to IN_WIDTH * SCALE_FACTOR");
  static_assert(OUT_W_PAR == IN_W_PAR * SCALE_FACTOR,
                "OUT_W_PAR must be equal to IN_W_PAR * SCALE_FACTOR");

  StreamingUpsample() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_h = 0, i_sf_h = 0, i_wch = 0;

    // Input buffer
    TOutputWord buffer[IN_WIDTH * IN_CH / (IN_W_PAR * CH_PAR)]
                        [IN_W_PAR];
    PipelineDelayBuffer<TOutputWord> delayed_output[OUT_W_PAR];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < OUT_W_PAR; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status =
          ActorStatus(depth, OUT_HEIGHT * OUT_WIDTH * IN_CH / (CH_PAR * OUT_W_PAR));
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

  ActorStatus step(hls::stream<TInputWord> i_data[IN_W_PAR],
                   hls::stream<TOutputWord> o_data[OUT_W_PAR]) {
    // Find the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    if (st.i_sf_h == 0) {
      for (size_t w_par = 0; w_par < IN_W_PAR; w_par++) {
        if (i_data[w_par].empty()) {
          firing_condition = false;
        }
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> instant_o_data[OUT_W_PAR];
      StreamingUpsample::pipeline_body(i_data, st.buffer, instant_o_data,
                                       st.i_sf_h,
                                       st.i_wch);
      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t w_par = 0; w_par < OUT_W_PAR; w_par++) {
        if (!instant_o_data[w_par].empty()) {
          st.delayed_output[w_par].push(instant_o_data[w_par].read(), true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output[w_par].push(TOutputWord(), false); 
        }
      }

      // Update the counters.
      st.i_wch++;
      if (st.i_wch >= IN_WIDTH * IN_CH / (IN_W_PAR * CH_PAR)) {
        st.i_wch = 0;
        st.i_sf_h++;
      }
      if (st.i_sf_h >= SCALE_FACTOR) {
        st.i_sf_h = 0;
        st.i_h++;
      }
      if (st.i_h >= IN_HEIGHT) {
        st.i_h = 0;
      }
    } else {
      // If not firing, just advance the delayed output buffers.
      for (size_t w_par = 0; w_par < OUT_W_PAR; w_par++) {
        st.delayed_output[w_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    for (size_t w_par = 0; w_par < OUT_W_PAR; w_par++) {
      TOutputWord out;
      if (st.delayed_output[w_par].pop(out)) {
        o_data[w_par].write(out);
      }
    }

    // Return the actor status.
    return st.actor_status;
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[IN_W_PAR],
           hls::stream<TOutputWord> o_data[OUT_W_PAR]) {
    TOutputWord buffer[IN_WIDTH * IN_CH / (IN_W_PAR * CH_PAR)][IN_W_PAR];
    for (size_t i_h = 0; i_h < IN_HEIGHT; i_h++) {
      for (size_t sf_h = 0; sf_h < SCALE_FACTOR; sf_h++) {
        for (size_t i_wch = 0; i_wch < IN_WIDTH * IN_CH / (IN_W_PAR * CH_PAR);
             i_wch++) {
        STREAMINGUPSAMPLE_RUN_LOOP:
#pragma HLS pipeline II = 1
          StreamingUpsample::pipeline_body(i_data, buffer, o_data, sf_h, i_wch);
        }
      }
    }
  }

private:
  static void
  pipeline_body(hls::stream<TInputWord> i_data[IN_W_PAR],
                TOutputWord linebuffer[IN_WIDTH * IN_CH / (IN_W_PAR * CH_PAR)]
                                [IN_W_PAR],
                hls::stream<TOutputWord> o_data[OUT_W_PAR], size_t sf_h,
                size_t i_wch) {
#pragma HLS inline

    Quantizer quantizer;
    if (sf_h == 0) {
      // Read new input data only on the first scale factor height iteration
      for (size_t w_par = 0; w_par < IN_W_PAR; w_par++) {
        TInputWord in_word = i_data[w_par].read();
        TOutputWord out_word;
        for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++) {
          out_word[ch_par] = quantizer(in_word[ch_par]);
        }
        linebuffer[i_wch][w_par] = out_word;
      }
    }

    // Write output data
    for (size_t w_par = 0; w_par < IN_W_PAR; w_par++) {
      TOutputWord out_word = linebuffer[i_wch][w_par];
      for (size_t sf_w_iter = 0; sf_w_iter < SCALE_FACTOR; sf_w_iter++) {
        size_t out_index = w_par * SCALE_FACTOR + sf_w_iter;
        o_data[out_index].write(out_word);
      }
    }
  }
};