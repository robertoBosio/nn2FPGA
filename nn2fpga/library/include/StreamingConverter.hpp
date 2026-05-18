#pragma once
#include "ap_int.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>
#include <unordered_map>

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH,
          size_t W_PAR, size_t CH_PAR>
class StreamingConverter {
  static_assert(IN_WIDTH % W_PAR == 0, "W_PAR must divide IN_WIDTH");
  static_assert(IN_CH % CH_PAR == 0, "CH_PAR must divide IN_CH");
  static_assert(IN_WIDTH % CH_PAR == 0,
                "CH_PAR must divide IN_WIDTH for correct mapping");
  static_assert(IN_CH % W_PAR == 0,
                "W_PAR must divide IN_CH for correct mapping");

  struct StepState {
    size_t i_k = 0;
    size_t i_w = 0;
    size_t i_ch = 0;

    ActorStatus actor_status{1, 1};
    bool initialized = false;
    size_t depth = 0;
    PipelineDelayBuffer<TOutputWord> delayed_output[CH_PAR];

    void init(size_t depth) {
      if (initialized)
        return;
      // one firing per (k, w, ch) step
      actor_status =
          ActorStatus(1, (IN_HEIGHT * (IN_WIDTH / W_PAR) * (IN_CH / CH_PAR)));
      initialized = true;
      this->depth = depth;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

public:
  StreamingConverter() = default;

  void step_init(size_t pipeline_depth = 1) {
    registry()[this].init(pipeline_depth);
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> in_data[W_PAR],
           hls::stream<TOutputWord> out_data[CH_PAR]) {
#pragma HLS INLINE off

    for (int k = 0; k < (int)IN_HEIGHT; k++) {
      for (int w = 0; w < (int)(IN_WIDTH / W_PAR); w++) {
        for (int ch = 0; ch < (int)(IN_CH / CH_PAR); ch++) {
#pragma HLS PIPELINE II = 1
          pipeline_body(in_data, out_data, k, w, ch);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> in_data[W_PAR],
                   hls::stream<TOutputWord> out_data[CH_PAR]) {
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;
    int k = (int)st.i_k;
    int w = (int)st.i_w;
    int ch = (int)st.i_ch;

    // ── Firing condition ──────────────────────────────────────────

    bool firing_condition = true;

    for (int w_i = 0; w_i < (int)W_PAR; w_i++) {
      if (in_data[w_i].empty()) {
        firing_condition = false;
        break;
      }
    }

    if (firing_condition) {

      hls::stream<TOutputWord> instant_out_data[CH_PAR];
      StreamingConverter::pipeline_body(in_data, out_data, k, w, ch);

      // Advance loop iterators: k → j → r → ch
      st.i_ch++;
      if (st.i_ch >= (int)(IN_CH / CH_PAR)) {
        st.i_ch = 0;
        st.i_w++;
      }

      if (st.i_w >= (int)(IN_WIDTH / W_PAR)) {
        st.i_w = 0;
        st.i_k++;
      }
      if (st.i_k >= (int)IN_HEIGHT) {
        st.i_k = 0;
      }

      st.actor_status.fire();

      for (int ch_par = 0; ch_par < (int)CH_PAR; ch_par++) {
        if (!out_data[ch_par].empty()) {
          st.delayed_output[ch_par].push(out_data[ch_par].read(), true);
        } else {
          st.delayed_output[ch_par].push(TOutputWord(), false);
        }
      }
    }

    st.actor_status.advance();

    TOutputWord out_word;
    for (int ch_par = 0; ch_par < (int)CH_PAR; ch_par++) {
      if (st.delayed_output[ch_par].pop(out_word)) {
        out_data[ch_par].write(out_word);
      }
    }
    return st.actor_status;
  }

private:
  static void pipeline_body(hls::stream<TInputWord> in_data[W_PAR],
                            hls::stream<TOutputWord> out_data[CH_PAR], int k,
                            int w, int ch) {
#pragma HLS INLINE

    TOutputWord word_ch[CH_PAR];
#pragma HLS ARRAY_PARTITION variable = word_ch complete

    for (int w_i = 0; w_i < (int)W_PAR; w_i++) {
#pragma HLS UNROLL
      TInputWord pkt = in_data[w_i].read();
      for (int ch_i = 0; ch_i < (int)CH_PAR; ch_i++) {
#pragma HLS UNROLL
        word_ch[ch_i][w_i] = pkt[ch_i];
      }
    }

    for (int ch_i = 0; ch_i < (int)CH_PAR; ch_i++) {
#pragma HLS UNROLL
      out_data[ch_i].write(word_ch[ch_i]);
    }
  }
};