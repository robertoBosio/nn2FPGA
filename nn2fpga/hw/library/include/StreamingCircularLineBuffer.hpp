#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>
#include <unordered_map>

/**
 * @brief Single-actor circular-buffer linebuffer.
 *
 * This is an experimental replacement for the StreamingWindowSelector network.
 * It keeps the active expanded window partitioned over the spatial dimensions
 * and uses one circular row-delay buffer per previous row and width lane. The
 * row buffers are never shifted: each bank performs one read and one write per
 * input word.
 *
 * Padding is injected into the circular window state; only real tensor pixels are
 * consumed from the input streams.
 *
 * Current prototype limitations:
 * - no dilation;
 * - IN_WIDTH and OUT_WIDTH must be multiples of W_PAR.
 */
template <typename TWord, typename TData, size_t IN_HEIGHT, size_t IN_WIDTH,
          size_t IN_CH, size_t FH, size_t FW, size_t STRIDE_H,
          size_t STRIDE_W, size_t DILATION_H, size_t DILATION_W, size_t PAD_T,
          size_t PAD_L, size_t PAD_B, size_t PAD_R, size_t W_PAR,
          size_t CH_PAR, int PAD_VALUE = 0>
class StreamingCircularLineBuffer {
  static constexpr size_t FW_EXPAND = FW + (W_PAR - 1) * STRIDE_W;
  static constexpr size_t EMIT_LANE = (FW_EXPAND - 1) % W_PAR;
  static constexpr size_t PADDED_HEIGHT = IN_HEIGHT + PAD_T + PAD_B;
  static constexpr size_t PADDED_WIDTH = IN_WIDTH + PAD_L + PAD_R;
  static constexpr size_t OUT_HEIGHT =
      (IN_HEIGHT + PAD_T + PAD_B - DILATION_H * (FH - 1) - 1) / STRIDE_H + 1;
  static constexpr size_t OUT_WIDTH =
      (IN_WIDTH + PAD_L + PAD_R - DILATION_W * (FW - 1) - 1) / STRIDE_W + 1;
  static constexpr size_t PADDED_WORDS_PER_ROW = PADDED_WIDTH / W_PAR;
  static constexpr size_t CH_GROUPS = IN_CH / CH_PAR;
  static constexpr size_t ROW_DELAY_DEPTH = PADDED_WORDS_PER_ROW * CH_GROUPS;
  static constexpr size_t ROWBUF_ROWS = (FH > 1) ? (FH - 1) : 1;
  static constexpr bool IS_POINTWISE = (FH == 1) && (FW == 1);

  static bool should_load_input(size_t i_h, size_t i_w_word) {
#pragma HLS inline
    const size_t virt_col_start = i_w_word * W_PAR;
    const size_t input_load_lane = PAD_L % W_PAR;
    const size_t input_load_col = virt_col_start + input_load_lane;
    const bool real_row = (i_h >= PAD_T) && (i_h < PAD_T + IN_HEIGHT);
    return real_row && (input_load_col >= PAD_L) &&
           ((input_load_col - PAD_L) < IN_WIDTH);
  }

  static bool output_group_valid(size_t i_h, size_t i_w_word) {
#pragma HLS inline
    const size_t current_col = i_w_word * W_PAR + EMIT_LANE;
    const bool window_filled =
        (i_h >= FH - 1) && (current_col >= FW_EXPAND - 1);
    const size_t base_h = window_filled ? i_h - (FH - 1) : 0;
    const size_t base_w = window_filled ? current_col - (FW_EXPAND - 1) : 0;
    const bool stride_match = window_filled && ((base_h % STRIDE_H) == 0) &&
                              ((base_w % (STRIDE_W * W_PAR)) == 0);
    const size_t out_h = base_h / STRIDE_H;
    const size_t out_w = base_w / STRIDE_W;
    return stride_match && (out_h < OUT_HEIGHT) && (out_w + W_PAR <= OUT_WIDTH);
  }

  static void pipeline_body(hls::stream<TWord> i_data[W_PAR],
                            hls::stream<TWord> o_data[FH * FW_EXPAND],
                            TWord rowbuf[ROWBUF_ROWS][W_PAR][ROW_DELAY_DEPTH],
                            TWord window[CH_GROUPS][FH][FW_EXPAND],
                            TWord loaded_input[CH_GROUPS][W_PAR], size_t i_h,
                            size_t i_w_word, size_t i_ch) {
#pragma HLS inline
    const size_t rowbuf_index = i_w_word * CH_GROUPS + i_ch;
    const size_t virt_col_start = i_w_word * W_PAR;

    // The first real lane inside a virtual padded word is fixed by
    // PAD_L % W_PAR. When the scan reaches that lane, consume one full
    // real input word from all width-lane streams.
    const size_t input_load_lane = PAD_L % W_PAR;
    const bool real_row = (i_h >= PAD_T) && (i_h < PAD_T + IN_HEIGHT);
    const bool load_input = should_load_input(i_h, i_w_word);

    TWord next_input[W_PAR];
#pragma HLS ARRAY_PARTITION variable = next_input complete dim = 0

    if (load_input) {
      for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
#pragma HLS unroll
        next_input[i_w_par] = i_data[i_w_par].read();
      }
    }

    TWord new_cols[FH][W_PAR];
#pragma HLS ARRAY_PARTITION variable = new_cols complete dim = 0

    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
#pragma HLS unroll
      const size_t virt_col = virt_col_start + i_w_par;
      const bool real_col =
          (virt_col >= PAD_L) && (virt_col < PAD_L + IN_WIDTH);
      const bool real_pixel = real_row && real_col;
      const size_t real_lane = real_pixel ? (virt_col - PAD_L) % W_PAR : 0;

      // Lanes before input_load_lane still belong to the previous real word;
      // lanes at/after input_load_lane use the newly consumed word.
      const bool use_next_input = load_input && (i_w_par >= input_load_lane);
      TWord row_word = real_pixel
                           ? (use_next_input ? next_input[real_lane]
                                             : loaded_input[i_ch][real_lane])
                           : pad_word();
      new_cols[FH - 1][i_w_par] = row_word;

      // Cascade the new word through FH-1 one-row delay buffers. Each bank is
      // indexed by virtual column word and channel group, so no row data is
      // shifted in memory.
      for (size_t i_fh = 0; i_fh < FH - 1; i_fh++) {
#pragma HLS unroll
        TWord delayed_word = rowbuf[i_fh][i_w_par][rowbuf_index];
        rowbuf[i_fh][i_w_par][rowbuf_index] = row_word;
        new_cols[FH - 2 - i_fh][i_w_par] = delayed_word;
        row_word = delayed_word;
      }
    }

    // Commit the newly read real word only after all lanes have been formed;
    // otherwise mixed padding words would overwrite the previous word before
    // its tail lanes are used.
    if (load_input) {
      for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
#pragma HLS unroll
        loaded_input[i_ch][i_w_par] = next_input[i_w_par];
      }
    }

    bool output_valid_group = false;
    TWord output_window[FH][FW_EXPAND];
#pragma HLS ARRAY_PARTITION variable = output_window complete dim = 0

    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
#pragma HLS unroll
      // Shift the active horizontal window by one virtual pixel lane and append
      // the newly generated column for every window row.
      for (size_t i_fh = 0; i_fh < FH; i_fh++) {
#pragma HLS unroll
        for (size_t i_fw = 0; i_fw < FW_EXPAND - 1; i_fw++) {
#pragma HLS unroll
          window[i_ch][i_fh][i_fw] = window[i_ch][i_fh][i_fw + 1];
        }
        window[i_ch][i_fh][FW_EXPAND - 1] = new_cols[i_fh][i_w_par];
      }

      // Only one lane can complete a W_PAR-wide output group. Snapshot the
      // window here, but write the streams after the lane loop so HLS sees a
      // single write site per output FIFO and can keep II=1.
      if (i_w_par == EMIT_LANE) {
        output_valid_group = output_group_valid(i_h, i_w_word);
        for (size_t i_fh = 0; i_fh < FH; i_fh++) {
#pragma HLS unroll
          for (size_t i_fw = 0; i_fw < FW_EXPAND; i_fw++) {
#pragma HLS unroll
            output_window[i_fh][i_fw] = window[i_ch][i_fh][i_fw];
          }
        }
      }
    }

    if (output_valid_group) {
      for (size_t i_fh = 0; i_fh < FH; i_fh++) {
#pragma HLS unroll
        for (size_t i_fw = 0; i_fw < FW_EXPAND; i_fw++) {
#pragma HLS unroll
          o_data[i_fh * FW_EXPAND + i_fw].write(output_window[i_fh][i_fw]);
        }
      }
    }
  }

  static void pointwise_body(hls::stream<TWord> i_data[W_PAR],
                             hls::stream<TWord> o_data[FH * FW_EXPAND],
                             size_t i_h, size_t i_w_word) {
#pragma HLS inline
    TWord input_word = i_data[0].read();
    if ((i_h % STRIDE_H) == 0 && (i_w_word % STRIDE_W) == 0) {
      o_data[0].write(input_word);
    }
  }

public:
  static_assert(FH > 0 && FW > 0,
                "FH and FW must be greater than 0");
  static_assert(STRIDE_H > 0 && STRIDE_W > 0,
                "STRIDE_H and STRIDE_W must be greater than 0");
  static_assert(DILATION_H == 1 && DILATION_W == 1,
                "StreamingCircularLineBuffer currently supports dilation 1");
  static_assert(W_PAR > 0, "W_PAR must be greater than 0");
  static_assert(CH_PAR > 0, "CH_PAR must be greater than 0");
  static_assert(IN_CH % CH_PAR == 0, "IN_CH must be a multiple of CH_PAR");
  static_assert(IN_WIDTH % W_PAR == 0,
                "IN_WIDTH must be a multiple of W_PAR");
  static_assert(PADDED_WIDTH % W_PAR == 0,
                "PADDED_WIDTH must be a multiple of W_PAR");
  static_assert(OUT_WIDTH % W_PAR == 0,
                "OUT_WIDTH must be a multiple of W_PAR");
  static_assert(!IS_POINTWISE ||
                    (PAD_T == 0 && PAD_L == 0 && PAD_B == 0 && PAD_R == 0),
                "1x1 pointwise path does not support padding");
  static_assert(!IS_POINTWISE || W_PAR == 1,
                "1x1 pointwise path currently supports W_PAR=1");

  StreamingCircularLineBuffer() = default;

  static TWord pad_word() {
    TWord word;
    for (size_t i_ch_par = 0; i_ch_par < CH_PAR; i_ch_par++) {
#pragma HLS unroll
      word[i_ch_par] = TData(PAD_VALUE);
    }
    return word;
  }

  struct StepState {
    size_t i_h = 0;
    size_t i_w_word = 0;
    size_t i_ch = 0;

    TWord rowbuf[ROWBUF_ROWS][W_PAR][ROW_DELAY_DEPTH];
    TWord window[CH_GROUPS][FH][FW_EXPAND];
    TWord loaded_input[CH_GROUPS][W_PAR];
    PipelineDelayBuffer<TWord> delayed_output[FH * FW_EXPAND];
    ActorStatus actor_status{1, 1};
    bool initialized = false;
    size_t depth = 1;
    size_t output_fifo_depth = 2;

    void init(size_t depth, size_t output_fifo_depth = 2) {
      if (initialized)
        return;
      for (size_t i = 0; i < FH * FW_EXPAND; i++) {
        delayed_output[i] = PipelineDelayBuffer<TWord>(depth);
      }
      actor_status = ActorStatus(
          depth, PADDED_HEIGHT * PADDED_WORDS_PER_ROW * CH_GROUPS);
      this->depth = depth;
      this->output_fifo_depth = output_fifo_depth;
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

  void step_init(size_t pipeline_depth = 1, size_t output_fifo_depth = 2) {
    auto &st = registry()[this];
    st.init(pipeline_depth, output_fifo_depth);
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TWord> i_data[W_PAR],
           hls::stream<TWord> o_data[FH * FW_EXPAND]) {
    TWord rowbuf[ROWBUF_ROWS][W_PAR][ROW_DELAY_DEPTH];
#pragma HLS ARRAY_PARTITION variable = rowbuf complete dim = 1
#pragma HLS ARRAY_PARTITION variable = rowbuf complete dim = 2

    TWord window[CH_GROUPS][FH][FW_EXPAND];
#pragma HLS ARRAY_PARTITION variable = window complete dim = 2
#pragma HLS ARRAY_PARTITION variable = window complete dim = 3

    // Holds the most recently consumed real input word for each channel group.
    // It is needed when PAD_L is not aligned to W_PAR: one virtual padded word
    // can contain lanes from both the previous and the newly-read real word.
    TWord loaded_input[CH_GROUPS][W_PAR];
#pragma HLS ARRAY_PARTITION variable = loaded_input complete dim = 0

    // Iterate over the padded virtual image. Padding pixels are synthesized
    // locally; input streams are consumed only when this virtual scan reaches a
    // real input word boundary.
    for (size_t i_h = 0; i_h < PADDED_HEIGHT; i_h++) {
      for (size_t i_w_word = 0; i_w_word < PADDED_WORDS_PER_ROW; i_w_word++) {
      CIRCULAR_LINEBUFFER_RUN_LOOP:
        for (size_t i_ch = 0; i_ch < CH_GROUPS; i_ch++) {
#pragma HLS pipeline II = 1
          if constexpr (IS_POINTWISE) {
            StreamingCircularLineBuffer::pointwise_body(i_data, o_data, i_h,
                                                        i_w_word);
          } else {
            StreamingCircularLineBuffer::pipeline_body(
                i_data, o_data, rowbuf, window, loaded_input, i_h, i_w_word,
                i_ch);
          }
        }
      }
    }
  }

  ActorStatus step(hls::stream<TWord> i_data[W_PAR],
                   hls::stream<TWord> o_data[FH * FW_EXPAND]) {
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    bool firing_condition = true;
    bool blocked_by_input = false;
    bool blocked_by_output = false;
    if (IS_POINTWISE || should_load_input(st.i_h, st.i_w_word)) {
      for (size_t i = 0; i < W_PAR; i++) {
        if (i_data[i].empty()) {
          firing_condition = false;
          blocked_by_input = true;
        }
      }
    }

    if (st.depth == 1) {
      for (size_t i = 0; i < FH * FW_EXPAND; i++) {
        if (o_data[i].size() >= st.output_fifo_depth) {
          firing_condition = false;
          blocked_by_output = true;
        }
      }
    } else {
      for (size_t i = 0; i < FH * FW_EXPAND; i++) {
        if (st.delayed_output[i].peek() &&
            o_data[i].size() >= st.output_fifo_depth) {
          firing_condition = false;
          blocked_by_output = true;
        }
      }
    }

    if (firing_condition) {
      hls::stream<TWord> instant_o_data[FH * FW_EXPAND];
      if constexpr (IS_POINTWISE) {
        StreamingCircularLineBuffer::pointwise_body(i_data, instant_o_data,
                                                    st.i_h, st.i_w_word);
      } else {
        StreamingCircularLineBuffer::pipeline_body(
            i_data, instant_o_data, st.rowbuf, st.window, st.loaded_input,
            st.i_h, st.i_w_word, st.i_ch);
      }

      st.i_ch++;
      if (st.i_ch >= CH_GROUPS) {
        st.i_ch = 0;
        st.i_w_word++;
      }
      if (st.i_w_word >= PADDED_WORDS_PER_ROW) {
        st.i_w_word = 0;
        st.i_h++;
      }
      if (st.i_h >= PADDED_HEIGHT) {
        st.i_h = 0;
      }

      st.actor_status.fire();

      for (size_t i = 0; i < FH * FW_EXPAND; i++) {
        if (!instant_o_data[i].empty()) {
          st.delayed_output[i].push(instant_o_data[i].read(), true);
        } else {
          st.delayed_output[i].push(TWord(), false);
        }
      }

      st.actor_status.advance();

      for (size_t i = 0; i < FH * FW_EXPAND; i++) {
        TWord out;
        if (st.delayed_output[i].pop(out)) {
          o_data[i].write(out);
        }
      }
    }

    return st.actor_status;
  }
};
