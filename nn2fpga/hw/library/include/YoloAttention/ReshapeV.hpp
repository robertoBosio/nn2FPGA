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
          typename TOutput, typename Quantizer, size_t IN_HEADS, size_t IN_DIM,
          size_t IN_SEQ, size_t OUT_HEIGHT, size_t OUT_WIDTH, size_t OUT_CH,
          size_t REDUCE_PAR>
class ReshapeV {
  static_assert(OUT_CH % REDUCE_PAR == 0,
                "OUT_CH must be a multiple of REDUCE_PAR");
  static_assert(IN_DIM % REDUCE_PAR == 0,
                "IN_DIM must be a multiple of REDUCE_PAR");
  static_assert(REDUCE_PAR > 0, "REDUCE_PAR must be greater than 0");
  static_assert(OUT_HEIGHT > 0 && OUT_WIDTH > 0,
                "OUT_HEIGHT and OUT_WIDTH must be greater than 0");

  struct StepState {
    // Loop iteration indexes.
    size_t i_hw = 0, i_ch = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output;
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      delayed_output = PipelineDelayBuffer<TOutputWord>(depth);
      actor_status =
          ActorStatus(depth, OUT_HEIGHT * OUT_WIDTH * OUT_CH / REDUCE_PAR);
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

public:
  ReshapeV() = default;
  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  ActorStatus step(hls::stream<TInputWord> i_data[2],
                   hls::stream<TOutputWord> o_data[1]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    if (st.i_ch < IN_DIM) {
      if (i_data[0].empty()) {
        firing_condition = false;
      }
    } else {
      if (i_data[1].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      // If there is data in the input stream, process it.
      hls::stream<TOutputWord> instant_output_stream[1];
      ReshapeV::pipeline_body(i_data, instant_output_stream, st.i_ch);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      if (!instant_output_stream[0].empty()) {
        st.delayed_output.push(instant_output_stream[0].read(), true);
      } else {
        // If the output stream is empty, push a placeholder.
        st.delayed_output.push(TOutputWord(), false);
      }

      // Update the counters.
      st.i_ch += REDUCE_PAR;
      if (st.i_ch >= OUT_CH) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        st.i_ch = 0;
        st.i_hw++;
      }
      if (st.i_hw >= OUT_HEIGHT * OUT_WIDTH) {
        st.i_hw = 0; // Reset the height/width index if we have processed all
                     // iterations.
      }

    } else {
      // If there is no data in the input stream, push a delay slot.
      st.delayed_output.push(TOutputWord(), false);
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    if (st.delayed_output.pop(out)) {
      o_data[0].write(out);
    }

    // Return the actor status.
    return st.actor_status;
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[2],
           hls::stream<TOutputWord> o_data[1]) {
    for (size_t i_hw = 0; i_hw < OUT_HEIGHT * OUT_WIDTH; i_hw++) {
    RESHAPEV_RUN_LOOP:
      for (size_t i_ch = 0; i_ch < OUT_CH; i_ch += REDUCE_PAR) {
#pragma HLS PIPELINE II = 1
        pipeline_body(i_data, o_data, i_ch);
      }
    }
  }

private:
  void pipeline_body(hls::stream<TInputWord> i_data[2],
                     hls::stream<TOutputWord> o_data[1], size_t i_ch) {
#pragma HLS inline
    Quantizer quantizer;
    TInputWord in_word;
    TOutputWord out_word;

    if (i_ch < IN_DIM) {
      in_word = i_data[0].read();
    } else {
      in_word = i_data[1].read();
    }
    for (size_t ch_par = 0; ch_par < REDUCE_PAR; ch_par++) {
      out_word[ch_par] = quantizer(in_word[ch_par]);
    }
    o_data[0].write(out_word);
  }
};