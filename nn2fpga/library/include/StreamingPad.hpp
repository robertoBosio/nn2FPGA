#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>

template <typename TWord, typename TData, size_t IN_HEIGHT, size_t IN_WIDTH,
          size_t IN_CH, size_t FH, size_t FW, size_t STRIDE_H, size_t STRIDE_W,
          size_t DILATION_H, size_t DILATION_W, size_t PAD_T, size_t PAD_L,
          size_t PAD_B, size_t PAD_R, size_t W_PAR, size_t CH_PAR,
          int PAD_VALUE = 0>
class StreamingPad {
  static constexpr size_t FW_EXPAND = FW + (W_PAR - 1) * STRIDE_W;
  static constexpr size_t OUT_HEIGHT =
      (IN_HEIGHT + PAD_T + PAD_B - DILATION_H * (FH - 1) - 1) / STRIDE_H + 1;
  static constexpr size_t OUT_WIDTH =
      (IN_WIDTH + PAD_L + PAD_R - DILATION_W * (FW - 1) - 1) / STRIDE_W + 1;

public:
  static_assert(FH > 0 && FW > 0, "FH and FW must be greater than 0");
  static_assert(STRIDE_H > 0 && STRIDE_W > 0,
                "STRIDE_H and STRIDE_W must be greater than 0");
  static_assert(PAD_T >= 0 && PAD_L >= 0 && PAD_B >= 0 && PAD_R >= 0,
                "PAD_T, PAD_L, PAD_B and PAD_R must be non-negative");
  static_assert(W_PAR > 0, "W_PAR must be greater than 0");
  static_assert(CH_PAR > 0, "CH_PAR must be greater than 0");
  static_assert(FW_EXPAND > 0,
                "FW + (W_PAR-1)*STRIDE_W must be greater than 0");

  StreamingPad() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_h = 0, i_w = 0, i_ch = 0;
    PipelineDelayBuffer<TWord> delayed_output[FH * FW_EXPAND];
    ActorStatus actor_status{1, 1};
    bool initialized = false;
    size_t depth = 1;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < FH * FW_EXPAND; i++) {
        delayed_output[i] = PipelineDelayBuffer<TWord>(depth);
      }
      actor_status = ActorStatus(depth, OUT_HEIGHT * (OUT_WIDTH / W_PAR) *
                                            (IN_CH / CH_PAR));
      this->depth = depth;
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
  void run(hls::stream<TWord> i_data[FH * FW_EXPAND],
           hls::stream<TWord> o_data[FH * FW_EXPAND]) {
    for (size_t i_h = 0; i_h < OUT_HEIGHT; i_h++) {
      for (size_t i_w = 0; i_w < OUT_WIDTH; i_w += W_PAR) {
      STREAMINGPAD_RUN_LOOP:
        for (size_t i_ch = 0; i_ch < IN_CH; i_ch += CH_PAR) {
#pragma HLS pipeline II = 1
          StreamingPad::pipeline_body(i_data, o_data, i_h, i_w);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TWord> i_data[FH * FW_EXPAND],
                   hls::stream<TWord> o_data[FH * FW_EXPAND]) {
    // Find the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_fh = 0; i_fh < FH; i_fh++) {
      for (size_t i_fw = 0; i_fw < FW_EXPAND; i_fw++) {
        bool is_within_tensor = true;
        size_t in_h = st.i_h * STRIDE_H + i_fh - PAD_T;
        size_t in_w = st.i_w * STRIDE_W + i_fw - PAD_L;
        is_within_tensor &= (in_h < IN_HEIGHT && in_h >= 0);
        is_within_tensor &= (in_w < IN_WIDTH && in_w >= 0);
        if (is_within_tensor && i_data[i_fh * FW_EXPAND + i_fw].empty()) {
          firing_condition = false;
        }
      }
    }

    if (st.depth == 1) {
      for (size_t i_fh = 0; i_fh < FH; i_fh++) {
        for (size_t i_fw = 0; i_fw < FW_EXPAND; i_fw++) {
          if (o_data[i_fh * FW_EXPAND + i_fw].size() >= 2) {
            firing_condition = false;
          }
        }
      }
    } else {
      for (size_t i_fh = 0; i_fh < FH; i_fh++) {
        for (size_t i_fw = 0; i_fw < FW_EXPAND; i_fw++) {
          if (st.delayed_output[i_fh * FW_EXPAND + i_fw].peek() &&
              o_data[i_fh * FW_EXPAND + i_fw].size() >= 2) {
            firing_condition = false;
          }
        }
      }
    }

    if (firing_condition) {
      hls::stream<TWord> instant_o_data[FH * FW_EXPAND];
      StreamingPad::pipeline_body(i_data, instant_o_data, st.i_h, st.i_w);

      st.i_ch += CH_PAR;
      if (st.i_ch >= IN_CH) {
        st.i_ch = 0;
        st.i_w += W_PAR;
      }
      if (st.i_w >= OUT_WIDTH) {
        st.i_w = 0;
        st.i_h++;
      }
      if (st.i_h >= OUT_HEIGHT) {
        st.i_h = 0;
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_fh = 0; i_fh < FH; i_fh++) {
        for (size_t i_fw = 0; i_fw < FW_EXPAND; i_fw++) {
          st.delayed_output[i_fh * FW_EXPAND + i_fw].push(
              instant_o_data[i_fh * FW_EXPAND + i_fw].read(), true);
        }
      }
      // Advance the state of the actor firings.
      st.actor_status.advance();

      // Write the output data to the output stream.
      for (size_t i_fh = 0; i_fh < FH; i_fh++) {
        for (size_t i_fw = 0; i_fw < FW_EXPAND; i_fw++) {
          TWord out;
          if (st.delayed_output[i_fh * FW_EXPAND + i_fw].pop(out)) {
            o_data[i_fh * FW_EXPAND + i_fw].write(out);
          }
        }
      }
    }

    // Return the actor status.
    return st.actor_status;
  }

private:
  static void pipeline_body(hls::stream<TWord> i_data[FH * FW_EXPAND],
                            hls::stream<TWord> o_data[FH * FW_EXPAND],
                            size_t i_h, size_t i_w) {
#pragma HLS inline
    for (size_t i_fh = 0; i_fh < FH; i_fh++) {
      for (size_t i_fw = 0; i_fw < FW_EXPAND; i_fw++) {
        TWord in_word;
        bool is_within_tensor = true;
        size_t in_h = i_h * STRIDE_H + i_fh - PAD_T;
        size_t in_w = i_w * STRIDE_W + i_fw - PAD_L;
        is_within_tensor &= (in_h < IN_HEIGHT && in_h >= 0);
        is_within_tensor &= (in_w < IN_WIDTH && in_w >= 0);
        if (is_within_tensor) {
          in_word = i_data[i_fh * FW_EXPAND + i_fw].read();
        } else {
          for (size_t i_ch_par = 0; i_ch_par < CH_PAR; i_ch_par++) {
            in_word[i_ch_par] = TData(PAD_VALUE); // Padding with specified value
          }
        }
        o_data[i_fh * FW_EXPAND + i_fw].write(in_word);
      }
    }
  }
};