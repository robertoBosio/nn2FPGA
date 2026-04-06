#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>
#include <cassert>

template <typename TQKInputWord, typename TQKInput, typename TVInputWord,
          typename TVInput, typename TOutputWord, typename TOutput,
          typename TLUT, typename TNum, typename TDen, typename TDiv,
          typename Quantizer, size_t LUT_SIZE, size_t DIM_HEADS, size_t DIM_V,
          size_t DIM_SEQ, size_t REDUCE_PAR>
class StreamingFusedSoftmaxMatmul {
  
  struct StepState {
    // Loop iteration indexes.
    size_t i_seq = 0, i_v_row = 0, i_head = 0, i_red_group = 0;
    TVInputWord v_matrix[DIM_HEADS][DIM_V][DIM_SEQ / REDUCE_PAR];
    TQKInputWord qk_row[DIM_SEQ / REDUCE_PAR];
    TQKInput max;
    TNum num;
    TDen den;

    PipelineDelayBuffer<TOutput> delayed_output[1];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < 1; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutput>(depth);
      }
      actor_status =
          ActorStatus(depth, DIM_SEQ * DIM_V * DIM_HEADS * DIM_SEQ / REDUCE_PAR);
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


public:
  StreamingFusedSoftmaxMatmul() = default;

  ActorStatus step(hls::stream<TQKInputWord> qk_data[1],
            hls::stream<TVInputWord> v_data[1], const TLUT lut_table[LUT_SIZE],
            hls::stream<TOutput> o_data[1]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    if (st.i_v_row == 0) {
      firing_condition = firing_condition && !qk_data[0].empty();
    }
    if (st.i_seq == 0) {
      firing_condition = firing_condition && !v_data[0].empty();
    }

    if (firing_condition) {
      if (st.i_red_group == 0) {
        st.max = LimitsImpl<TQKInput>::min();
        st.num = 0;
        st.den = 0;
      }
      hls::stream<TOutput> o_data_instant[1];
      StreamingFusedSoftmaxMatmul::pipeline_body(
          qk_data, v_data, lut_table, o_data_instant, st.i_seq, st.i_v_row,
          st.i_red_group, st.max, st.num, st.den,
          st.v_matrix[st.i_head][st.i_v_row], st.qk_row);
      st.i_red_group++;
      if (st.i_red_group == DIM_SEQ / REDUCE_PAR) {
        st.i_red_group = 0;
        st.i_v_row++;
        if (st.i_v_row == DIM_V) {
          st.i_v_row = 0;
          st.i_head++;
          if (st.i_head == DIM_HEADS) {
            st.i_head = 0;
            st.i_seq++;
            if (st.i_seq == DIM_SEQ) {
              st.i_seq = 0;
            }
          }
        }
      }

      st.actor_status.fire();
      TOutput out_value;
      if (o_data_instant[0].read_nb(out_value)) {
        st.delayed_output[0].push(out_value, true);
      } else {
        st.delayed_output[0].push(TOutput(), false);
      }
    } else {
      st.delayed_output[0].push(TOutput(), false);
    }

    st.actor_status.advance();

    TOutput out_value;
    if (st.delayed_output[0].pop(out_value)) {
      o_data[0].write(out_value);
    }

    return st.actor_status;
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TQKInputWord> qk_data[1],
           hls::stream<TVInputWord> v_data[1], const TLUT lut_table[LUT_SIZE],
           hls::stream<TOutput> o_data[1]) {

    TVInputWord v_matrix[DIM_HEADS][DIM_V][DIM_SEQ / REDUCE_PAR];
    TQKInputWord qk_row[DIM_SEQ / REDUCE_PAR];
#pragma HLS array_partition variable = v_matrix complete dim = 4

    // Loop over the columns of the output matrix.
    for (size_t i_seq = 0; i_seq < DIM_SEQ; i_seq++) {

      // Loop over heads.
      for (size_t i_head = 0; i_head < DIM_HEADS; i_head++) {

        // loop over rows of V. 
        for (size_t i_v_row = 0; i_v_row < DIM_V; i_v_row++) {

          // Tracking the max and sum for each lane in the group.
          TQKInput max = LimitsImpl<TQKInput>::min();
          TNum num = 0;
          TDen den = 0;

          // Loop over the groups of unrolled operations in the lane.
          for (size_t i_red_group = 0; i_red_group < DIM_SEQ / REDUCE_PAR;
               i_red_group++) {
          STREAMINGSOFTMAX_RUN_LOOP:
#pragma HLS PIPELINE II = 1

            StreamingFusedSoftmaxMatmul::pipeline_body(
                qk_data, v_data, lut_table, o_data, i_seq, i_v_row, i_red_group,
                max, num, den, v_matrix[i_head][i_v_row], qk_row);
          }
        }
      }
    }
  }

private:
  static void pipeline_body(hls::stream<TQKInputWord> qk_data[1],
                            hls::stream<TVInputWord> v_data[1],
                            const TLUT lut_table[LUT_SIZE],
                            hls::stream<TOutput> o_data[1], size_t i_seq,
                            size_t i_v_row, size_t i_red_group, TQKInput &max,
                            TNum &num, TDen &den,
                            TVInputWord v_row[DIM_SEQ / REDUCE_PAR],
                            TQKInputWord qk_row[DIM_SEQ / REDUCE_PAR]) {
#pragma HLS inline

    // Output quantizer
    Quantizer quantizer;

    // Precision of the exponantial function.
    const unsigned int exp_precision = TLUT::width;

    // Precision of the division result. It is used for shifting the numerator before division 
    // to have more precision in the mantissa of the result. 
    const unsigned int div_precision = TDiv::width - TNum::width;

    // For rescaling the den when the max is updated.
    typedef ap_uint<TDen::width + exp_precision> TRescaledDen;
    DequantQuantPo2<exp_precision, TRescaledDen, TDen> rescalerDen;

    // For rescaling the num when the max is updated.
    typedef ap_int<TNum::width + exp_precision> TRescaledNum;
    DequantQuantPo2<exp_precision, TRescaledNum, TNum> rescalerNum;

    // The maximum value stored in the LUT, representing 1.0 in the exponential
    // function.
    const unsigned int lut_max_input = lut_table[0]; // Assuming the first entry corresponds to exp(0) = 1.0

    // The address type for the LUT, which is based on the input type width
    // but it must be unsigned.
    typedef ap_uint<TQKInput::width> TAddress;

    // Read the v data from the stream only for the first time.
    if (i_seq == 0) {
      v_row[i_red_group] = v_data[0].read();
    }
    TVInputWord v_word = v_row[i_red_group];

    // Read the qk data from the stream at every new column of V.
    if (i_v_row == 0) {
      qk_row[i_red_group] = qk_data[0].read();
    }
    TQKInputWord x_word = qk_row[i_red_group];

    // Perform the softmax computation for the input value and write to output
    for (size_t i_red_par = 0; i_red_par < REDUCE_PAR; i_red_par++) {
      TVInput v_i = (TVInput)v_word[i_red_par];
      TQKInput x_i = (TQKInput)x_word[i_red_par];

        if (x_i > max) {
          // Compute the address for the LUT based on the difference between
          // the current max and the input value It is for sure positive since
          // x > max
          TAddress diff = x_i  - max;

          // Read the LUT value for the current difference
          TLUT alpha = lut_table[diff];

          // Rescale the denominator
          TRescaledDen scaled_den = (den * alpha);
          den = rescalerDen(scaled_den);
          den = den + lut_max_input;

          // Rescale the numerator
          TRescaledNum scaled_num = (num * alpha);
          num = rescalerNum(scaled_num);
          num = num + v_i * lut_max_input;
          
          // Update the max value
          max = x_i;

        } else {

          // Compute the address for the LUT based on the difference between
          // the current max and the input value
          TAddress diff = max - x_i;
          // Read the LUT value for the current difference and accumulate it
          // to the sum
          TLUT exp = lut_table[diff];
          // Accumulate the exponentials in Q0.16 format
          den = den + exp;
          num = num + exp * v_i;
        }
    }

    if (i_red_group == (DIM_SEQ / REDUCE_PAR) - 1) {
      TDiv div_result = ((TDiv)num << div_precision) / den;
      TOutput out_value = quantizer(div_result);
      o_data[0].write(out_value);
    }
  }
};