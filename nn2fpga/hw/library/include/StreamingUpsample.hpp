#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>

template <typename TInputWord, typename TOutputWord, typename Quantizer,
          size_t IN_DIM0, size_t IN_DIM1, size_t IN_DIM2, size_t OUT_DIM0,
          size_t OUT_DIM1, size_t SCALE_FACTOR, size_t IN_DIM1_UNROLL,
          size_t OUT_DIM1_UNROLL, size_t DIM2_UNROLL>
class StreamingUpsampleDim01 {
public:
  static_assert(IN_DIM2 % DIM2_UNROLL == 0,
                "IN_DIM2 must be a multiple of DIM2_UNROLL");
  static_assert(DIM2_UNROLL > 0, "DIM2_UNROLL must be greater than 0");
  static_assert(IN_DIM1_UNROLL > 0, "IN_DIM1_UNROLL must be greater than 0");
  static_assert(IN_DIM1 % IN_DIM1_UNROLL == 0,
                "IN_DIM1 must be a multiple of IN_DIM1_UNROLL");
  static_assert(OUT_DIM1_UNROLL > 0, "OUT_DIM1_UNROLL must be greater than 0");
  static_assert(OUT_DIM1 % OUT_DIM1_UNROLL == 0,
                "OUT_DIM1 must be a multiple of OUT_DIM1_UNROLL");
  static_assert(IN_DIM0 > 0 && IN_DIM1 > 0,
                "IN_DIM0 and IN_DIM1 must be greater than 0");
  static_assert(SCALE_FACTOR > 1, "SCALE_FACTOR must be greater than 1");
  static_assert(OUT_DIM0 == IN_DIM0 * SCALE_FACTOR,
                "OUT_DIM0 must be equal to IN_DIM0 * SCALE_FACTOR");
  static_assert(OUT_DIM1 == IN_DIM1 * SCALE_FACTOR,
                "OUT_DIM1 must be equal to IN_DIM1 * SCALE_FACTOR");
  static_assert(
      OUT_DIM1_UNROLL == IN_DIM1_UNROLL * SCALE_FACTOR,
      "OUT_DIM1_UNROLL must be equal to IN_DIM1_UNROLL * SCALE_FACTOR");

  StreamingUpsampleDim01() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_dim0 = 0, i_sf_h = 0, i_dim12 = 0;

    // Input buffer
    TOutputWord buffer[IN_DIM1 * IN_DIM2 / (IN_DIM1_UNROLL * DIM2_UNROLL)]
                      [IN_DIM1_UNROLL];
    PipelineDelayBuffer<TOutputWord> delayed_output[OUT_DIM1_UNROLL];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < OUT_DIM1_UNROLL; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status = ActorStatus(depth, OUT_DIM0 * OUT_DIM1 * IN_DIM2 /
                                            (DIM2_UNROLL * OUT_DIM1_UNROLL));
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

  ActorStatus step(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL]) {
    // Find the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    if (st.i_sf_h == 0) {
      for (size_t i_dim1_par = 0; i_dim1_par < IN_DIM1_UNROLL; i_dim1_par++) {
        if (i_data[i_dim1_par].empty()) {
          firing_condition = false;
        }
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> instant_o_data[OUT_DIM1_UNROLL];
      StreamingUpsampleDim01::pipeline_body(i_data, st.buffer, instant_o_data,
                                            st.i_sf_h, st.i_dim12);
      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_dim1_par = 0; i_dim1_par < OUT_DIM1_UNROLL; i_dim1_par++) {
        if (!instant_o_data[i_dim1_par].empty()) {
          st.delayed_output[i_dim1_par].push(instant_o_data[i_dim1_par].read(),
                                             true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output[i_dim1_par].push(TOutputWord(), false);
        }
      }

      // Update the counters.
      st.i_dim12++;
      if (st.i_dim12 >= IN_DIM1 * IN_DIM2 / (IN_DIM1_UNROLL * DIM2_UNROLL)) {
        st.i_dim12 = 0;
        st.i_sf_h++;
      }
      if (st.i_sf_h >= SCALE_FACTOR) {
        st.i_sf_h = 0;
        st.i_dim0++;
      }
      if (st.i_dim0 >= IN_DIM0) {
        st.i_dim0 = 0;
      }
    } else {
      // If not firing, just advance the delayed output buffers.
      for (size_t i_dim1_par = 0; i_dim1_par < OUT_DIM1_UNROLL; i_dim1_par++) {
        st.delayed_output[i_dim1_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    for (size_t i_dim1_par = 0; i_dim1_par < OUT_DIM1_UNROLL; i_dim1_par++) {
      TOutputWord out;
      if (st.delayed_output[i_dim1_par].pop(out)) {
        o_data[i_dim1_par].write(out);
      }
    }

    // Return the actor status.
    return st.actor_status;
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
           hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL]) {
    TOutputWord buffer[IN_DIM1 * IN_DIM2 / (IN_DIM1_UNROLL * DIM2_UNROLL)]
                      [IN_DIM1_UNROLL];
    for (size_t i_dim0 = 0; i_dim0 < IN_DIM0; i_dim0++) {
      for (size_t sf_h = 0; sf_h < SCALE_FACTOR; sf_h++) {
      STREAMINGUPSAMPLE_RUN_LOOP:
        for (size_t i_dim12 = 0;
             i_dim12 < IN_DIM1 * IN_DIM2 / (IN_DIM1_UNROLL * DIM2_UNROLL);
             i_dim12++) {
#pragma HLS pipeline II = 1
          StreamingUpsampleDim01::pipeline_body(i_data, buffer, o_data, sf_h,
                                                i_dim12);
        }
      }
    }
  }

private:
  static void
  pipeline_body(hls::stream<TInputWord> i_data[IN_DIM1_UNROLL],
                TOutputWord linebuffer[IN_DIM1 * IN_DIM2 / (IN_DIM1_UNROLL * DIM2_UNROLL)]
                                [IN_DIM1_UNROLL],
                hls::stream<TOutputWord> o_data[OUT_DIM1_UNROLL], size_t sf_h,
                size_t i_dim12) {
#pragma HLS inline

    Quantizer quantizer;
    if (sf_h == 0) {
      // Read new input data only on the first scale factor height iteration
      for (size_t i_dim1_par = 0; i_dim1_par < IN_DIM1_UNROLL; i_dim1_par++) {
        TInputWord in_word = i_data[i_dim1_par].read();
        TOutputWord out_word;
        for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL; i_dim2_par++) {
          out_word[i_dim2_par] = quantizer(in_word[i_dim2_par]);
        }
        linebuffer[i_dim12][i_dim1_par] = out_word;
      }
    }

    // Write output data
    for (size_t i_dim1_par = 0; i_dim1_par < IN_DIM1_UNROLL; i_dim1_par++) {
      TOutputWord out_word = linebuffer[i_dim12][i_dim1_par];
      for (size_t sf_w_iter = 0; sf_w_iter < SCALE_FACTOR; sf_w_iter++) {
        size_t out_index = i_dim1_par * SCALE_FACTOR + sf_w_iter;
        o_data[out_index].write(out_word);
      }
    }
  }
};