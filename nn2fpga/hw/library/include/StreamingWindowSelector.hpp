#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>

/**
 * @brief Implements a single pixel of the line buffer. Discard and shifts
 * data based on the window position, stride and padding.
 *
 * @tparam TWord         Data type of the input/output stream word.
 * @tparam IN_HEIGHT     Input feature map height.
 * @tparam IN_WIDTH      Input feature map width.
 * @tparam IN_CH         Number of input channels.
 * @tparam FH            Filter/window height.
 * @tparam FW            Filter/window width.
 * @tparam STRIDE_H      Stride in height direction.
 * @tparam STRIDE_W      Stride in width direction.
 * @tparam DILATION_H    Dilation in height direction (must be 1).
 * @tparam DILATION_W    Dilation in width direction (must be 1).
 * @tparam PAD_T         Padding at the top.
 * @tparam PAD_L         Padding on the left.
 * @tparam PAD_B         Padding at the bottom.
 * @tparam PAD_R         Padding on the right.
 * @tparam POS_H         Position in the window (height index).
 * @tparam POS_W         Position in the window (width index).
 * @tparam W_PAR         Parallelism factor in width (number of columns
 * processed in parallel).
 * @tparam CH_PAR        Parallelism factor in channels (number of channels
 * processed in parallel).
 *
 * @section Usage
 * - Use the run() method for functional verification and synthesis.
 * - Use the step() method for self-timed execution with actor status tracking,
 * which is needed for fifo depth estimation.
 *
 * @section Limitations
 * - DILATION_H and DILATION_W must be 1.
 *
 * @section Parallelism
 * The class supports parallel processing of channels and width, as specified by
 * CH_PAR and W_PAR.
 */

template <typename TWord, size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH,
          size_t FH, size_t FW, size_t STRIDE_H, size_t STRIDE_W,
          size_t DILATION_H, size_t DILATION_W, size_t PAD_T, size_t PAD_L,
          size_t PAD_B, size_t PAD_R, size_t POS_H, size_t POS_W, size_t W_PAR,
          size_t CH_PAR>
class StreamingWindowSelector {
  static constexpr size_t FW_EXPAND = FW + (W_PAR - 1) * STRIDE_W;

  static constexpr size_t TOP_BORDER = (PAD_T > POS_H) ? 0 : POS_H - PAD_T;
  static constexpr size_t LEFT_BORDER = (PAD_L > POS_W) ? 0 : POS_W - PAD_L;
  static constexpr size_t BOTTOM_BORDER =
      IN_HEIGHT - ((FH - 1 - POS_H) - PAD_B);
  static constexpr size_t RIGHT_BORDER =
      IN_WIDTH - ((FW_EXPAND - 1 - POS_W) - PAD_R);

  // Stream lane in W_PAR considered by this pixel.
  static constexpr size_t W_STREAM = (POS_W + PAD_L * (W_PAR - 1)) % W_PAR;

  // Row modulo selected by this pixel.
  static constexpr size_t H_ROW_MOD =
      (POS_H + PAD_T * (STRIDE_H - 1)) % STRIDE_H;

  // Word-position modulo selected by this pixel.
  static constexpr size_t W_COL_MOD =
      ((POS_W - PAD_L + (W_PAR * STRIDE_W)) / W_PAR) % STRIDE_W;

  // Number of words seen by this stream lane across the input row.
  // These correspond to columns:
  //   W_STREAM, W_STREAM + W_PAR, W_STREAM + 2*W_PAR, ...
  static constexpr size_t WORDS_THIS_STREAM =
      (IN_WIDTH <= W_STREAM) ? 0 : ((IN_WIDTH - 1 - W_STREAM) / W_PAR + 1);

  static_assert(IN_CH % CH_PAR == 0, "IN_CH must be a multiple of CH_PAR");
  static_assert(FH > 0 && FW > 0, "FH and FW must be greater than 0");
  static_assert(STRIDE_H > 0 && STRIDE_W > 0,
                "STRIDE_H and STRIDE_W must be greater than 0");
  static_assert(DILATION_H > 0 && DILATION_W > 0,
                "DILATION_H and DILATION_W must be greater than 0");
  static_assert(POS_H < FH, "POS_H must be less than FH");
  static_assert(POS_W < FW_EXPAND,
                "POS_W must be less than FW + (W_PAR-1)*STRIDE_W");
  static_assert(W_PAR > 0, "W_PAR must be greater than 0");
  static_assert(CH_PAR > 0, "CH_PAR must be greater than 0");
  static_assert(FW_EXPAND > 0,
                "FW + (W_PAR-1)*STRIDE_W must be greater than 0");

  // Current limitations
  static_assert(DILATION_H == 1 && DILATION_W == 1,
                "DILATION_H and DILATION_W must be 1");

private:
  static constexpr size_t col_from_word(size_t i_w_word) {
    return W_STREAM + i_w_word * W_PAR;
  }

  static bool match(size_t i_h, size_t i_w_word) {
#pragma HLS inline
    const size_t col = col_from_word(i_w_word);

    bool is_within_window = true;
    is_within_window &= (i_h >= TOP_BORDER && i_h < BOTTOM_BORDER);
    is_within_window &= (col >= LEFT_BORDER && col < RIGHT_BORDER);
    is_within_window &= ((i_h % STRIDE_H) == H_ROW_MOD);
    is_within_window &= ((i_w_word % STRIDE_W) == W_COL_MOD);
    return is_within_window;
  }

  static void pipeline_body(hls::stream<TWord> &i_data,
                            hls::stream<TWord> &o_data,
                            hls::stream<TWord> &o_shift_data, size_t i_h,
                            size_t i_w_word) {
#pragma HLS inline
    TWord in_word = i_data.read();

    if (match(i_h, i_w_word)) {
      o_data.write(in_word);
    }
    o_shift_data.write(in_word);
  }

  static void pipeline_body(hls::stream<TWord> &i_data,
                            hls::stream<TWord> &o_data, size_t i_h,
                            size_t i_w_word) {
#pragma HLS inline
    TWord in_word = i_data.read();

    if (match(i_h, i_w_word)) {
      o_data.write(in_word);
    }
  }

public:
  StreamingWindowSelector() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_h = 0;
    size_t i_w_word = 0;
    size_t i_ch = 0;

    PipelineDelayBuffer<TWord> delayed_output[2];
    ActorStatus actor_status{1, 1};
    bool initialized = false;
    size_t depth = 1;
    size_t shift_stream_depth = 1;

    void init(size_t depth, size_t shift_stream_depth) {
      if (initialized)
        return;
      delayed_output[0] = PipelineDelayBuffer<TWord>(depth);
      delayed_output[1] = PipelineDelayBuffer<TWord>(depth);
      actor_status =
          ActorStatus(depth, IN_HEIGHT * WORDS_THIS_STREAM * (IN_CH / CH_PAR));
      this->depth = depth;
      this->shift_stream_depth = shift_stream_depth;
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;

  static Registry &registry() {
    static Registry r;
    return r;
  }

  void step_init(size_t pipeline_depth = 1, size_t shift_stream_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth, shift_stream_depth);
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TWord> &i_data, hls::stream<TWord> &o_data,
           hls::stream<TWord> &o_shift_data) {
    for (size_t i_h = 0; i_h < IN_HEIGHT; i_h++) {
      for (size_t i_w_word = 0; i_w_word < WORDS_THIS_STREAM; i_w_word++) {
      WINDOWSELECTOR_RUN_LOOP:
        for (size_t i_ch = 0; i_ch < IN_CH / CH_PAR; i_ch++) {
#pragma HLS pipeline II = 1
          StreamingWindowSelector::pipeline_body(i_data, o_data,
                                                    o_shift_data, i_h,
                                                    i_w_word);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TWord> &i_data, hls::stream<TWord> &o_data,
                   hls::stream<TWord> &o_shift_data) {
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    bool firing_condition = true;
    if (i_data.size() < st.shift_stream_depth) {
      firing_condition = false;
    }

    if (firing_condition) {
      hls::stream<TWord> instant_o_data[1];
      hls::stream<TWord> instant_o_shift_data[1];

      StreamingWindowSelector::pipeline_body(
          i_data, instant_o_data[0], instant_o_shift_data[0], st.i_h,
          st.i_w_word);

      st.i_ch += CH_PAR;
      if (st.i_ch >= IN_CH) {
        st.i_ch = 0;
        st.i_w_word++;
      }
      if (st.i_w_word >= WORDS_THIS_STREAM) {
        st.i_w_word = 0;
        st.i_h++;
      }
      if (st.i_h >= IN_HEIGHT) {
        st.i_h = 0;
      }

      st.actor_status.fire();

      if (!instant_o_data[0].empty()) {
        st.delayed_output[0].push(instant_o_data[0].read(), true);
      } else {
        st.delayed_output[0].push(TWord(), false);
      }

      if (!instant_o_shift_data[0].empty()) {
        st.delayed_output[1].push(instant_o_shift_data[0].read(), true);
      } else {
        st.delayed_output[1].push(TWord(), false);
      }

      st.actor_status.advance();

      TWord out;
      if (st.delayed_output[0].pop(out)) {
        o_data.write(out);
      }
      if (st.delayed_output[1].pop(out)) {
        o_shift_data.write(out);
      }
    }

    return st.actor_status;
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TWord> &i_data, hls::stream<TWord> &o_data) {
    for (size_t i_h = 0; i_h < IN_HEIGHT; i_h++) {
      for (size_t i_w_word = 0; i_w_word < WORDS_THIS_STREAM; i_w_word++) {
      WINDOWSELECTOR_RUN_LOOP:
        for (size_t i_ch = 0; i_ch < IN_CH / CH_PAR; i_ch++) {
#pragma HLS pipeline II = 1
          StreamingWindowSelector::pipeline_body(i_data, o_data, i_h,
                                                    i_w_word);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TWord> &i_data, hls::stream<TWord> &o_data) {
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    bool firing_condition = true;
    if (i_data.empty()) {
      firing_condition = false;
    }

    if (firing_condition) {
      hls::stream<TWord> instant_o_data[1];

      StreamingWindowSelector::pipeline_body(i_data, instant_o_data[0],
                                                st.i_h, st.i_w_word);

      st.i_ch += CH_PAR;
      if (st.i_ch >= IN_CH) {
        st.i_ch = 0;
        st.i_w_word++;
      }
      if (st.i_w_word >= WORDS_THIS_STREAM) {
        st.i_w_word = 0;
        st.i_h++;
      }
      if (st.i_h >= IN_HEIGHT) {
        st.i_h = 0;
      }

      st.actor_status.fire();

      if (!instant_o_data[0].empty()) {
        st.delayed_output[0].push(instant_o_data[0].read(), true);
      } else {
        st.delayed_output[0].push(TWord(), false);
      }
    } else {
      st.delayed_output[0].push(TWord(), false);
    }

    st.actor_status.advance();

    TWord out;
    if (st.delayed_output[0].pop(out)) {
      o_data.write(out);
    }

    return st.actor_status;
  }
};