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
           typename TOutput, typename Quantizer, size_t OUT_DIM0,
           size_t OUT_DIM1, size_t OUT_DIM2, size_t FH, size_t FW,
           size_t STRIDE_W, size_t DIM1_UNROLL, size_t DIM2_UNROLL>
class StreamingWindowToChannels {
    static constexpr size_t IN_DIM2 = OUT_DIM2 / (FH * FW);
    static constexpr size_t CHANNEL_GROUPS = IN_DIM2 / DIM2_UNROLL;
    static constexpr size_t FW_EXPAND = FW + (DIM1_UNROLL - 1) * STRIDE_W;
public:
  static_assert(FH > 0 && FW > 0, "Window dimensions must be greater than 0");
  static_assert(STRIDE_W > 0, "STRIDE_W must be greater than 0");
  static_assert(DIM1_UNROLL > 0 && DIM2_UNROLL > 0,
                "Unroll factors must be greater than 0");
  static_assert(OUT_DIM2 % (FH * FW) == 0,
                "Output channels must be divisible by the window area");
  static_assert(OUT_DIM1 % DIM1_UNROLL == 0,
                "Output width must be divisible by DIM1_UNROLL");
  static_assert(IN_DIM2 % DIM2_UNROLL == 0,
                "Input channels must be divisible by DIM2_UNROLL");

  StreamingWindowToChannels() = default;
  
  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
            hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {

    for (size_t i_dim0 = 0; i_dim0 < OUT_DIM0; i_dim0++) {
      for (size_t i_dim1 = 0; i_dim1 < OUT_DIM1 / DIM1_UNROLL; i_dim1++) {
      STREAMINGWINDOWTOCHANNELS_RUN_LOOP:
        for (size_t i_dim2 = 0; i_dim2 < OUT_DIM2 / DIM2_UNROLL; i_dim2++) {
#pragma HLS pipeline II = 1
          StreamingWindowToChannels::pipeline_body(i_data, o_data, i_dim2);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
                   hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    const size_t window_position = st.i_dim2 / CHANNEL_GROUPS;
    const size_t fh = window_position % FH;
    const size_t fw = window_position / FH;

    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
#pragma HLS unroll
      const size_t stream_index = fh * FW_EXPAND + i_dim1_par * STRIDE_W + fw;
      if (i_data[stream_index].empty()) {
        firing_condition = false;
        break;
      }
    }
    
    if (firing_condition) {
      hls::stream<TOutputWord> o_data_instant[DIM1_UNROLL];
      StreamingWindowToChannels::pipeline_body(i_data, o_data_instant,
                                               st.i_dim2);
      st.actor_status.fire();
      st.i_dim2++;
      if (st.i_dim2 == OUT_DIM2 / DIM2_UNROLL) {
        st.i_dim2 = 0;
        st.i_dim1++;
        if (st.i_dim1 == OUT_DIM1 / DIM1_UNROLL) {
          st.i_dim1 = 0;
          st.i_dim0++;
          if (st.i_dim0 == OUT_DIM0) {
            st.i_dim0 = 0;
          }
        }
      }

      // Add the output to the delayed output buffers
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        if (!o_data_instant[i_dim1_par].empty()) {
          st.delayed_output[i_dim1_par].push(o_data_instant[i_dim1_par].read(),
                                             true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output[i_dim1_par].push(TOutputWord(), false);
        }
      }
    } else {
      // If not firing, push invalid data to maintain pipeline timing
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        st.delayed_output[i_dim1_par].push(TOutputWord(), false);
      }
    }

    // Advance the actor status
    st.actor_status.advance();

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
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        delayed_output[i_dim1_par] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status = ActorStatus(depth, OUT_DIM0 * (OUT_DIM1 / DIM1_UNROLL) *
                                            (OUT_DIM2 / DIM2_UNROLL));
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

  static void pipeline_body(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
                            hls::stream<TOutputWord> o_data[DIM1_UNROLL],
                            size_t i_dim2) {
#pragma HLS inline
    // Channel groups are serialized after each window position.
    const size_t window_position = i_dim2 / CHANNEL_GROUPS;
    // Flatten columns before rows to preserve the YOLOX Focus channel order:
    // top-left, bottom-left, top-right, bottom-right.
    const size_t fh = window_position % FH;
    const size_t fw = window_position / FH;
    Quantizer quantizer;

    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL;
         i_dim1_par++) {
#pragma HLS unroll
      const size_t stream_index =
          fh * FW_EXPAND + i_dim1_par * STRIDE_W + fw;
      TInputWord input_word = i_data[stream_index].read();
      TOutputWord output_word;

      for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL;
           i_dim2_par++) {
#pragma HLS unroll
        output_word[i_dim2_par] = quantizer(input_word[i_dim2_par]);
      }
      o_data[i_dim1_par].write(output_word);
    }
  }
};
