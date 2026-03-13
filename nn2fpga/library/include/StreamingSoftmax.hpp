#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>
#include <cassert>

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename TLUT, typename TAcc, typename TDiv,
          typename Quantizer, size_t LUT_SIZE, size_t DIM_LANES,
          size_t DIM_REDUCTION, size_t LANE_PAR, size_t REDUCE_PAR>
class StreamingOnlineSoftmax {

  struct StepState {
    // Loop iteration indexes.
    size_t i_lane_group = 0, i_red_group = 0, i_step = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output[LANE_PAR];
    TInputWord in_row[LANE_PAR][DIM_REDUCTION / REDUCE_PAR];
    TInput max[LANE_PAR];
    TAcc sum[LANE_PAR];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < LANE_PAR; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status = ActorStatus(depth, DIM_LANES * DIM_REDUCTION * 2 /
                                            (REDUCE_PAR * LANE_PAR));
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

  ActorStatus step(hls::stream<TInputWord> i_data[LANE_PAR],
                   const TLUT lut_table[LUT_SIZE],
                   hls::stream<TOutputWord> o_data[LANE_PAR]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_lane_par = 0; i_lane_par < LANE_PAR; i_lane_par++) {
      if (st.i_step == 0 && i_data[i_lane_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> o_data_instant[LANE_PAR];
      StreamingOnlineSoftmax::pipeline_body(i_data, lut_table, o_data_instant,
                                      st.i_red_group, st.i_step, st.max, st.sum,
                                      st.in_row);

      // Update iterators
      st.i_red_group++;
      if (st.i_red_group == DIM_REDUCTION / REDUCE_PAR) {
        st.i_red_group = 0;
        st.i_step++;
        if (st.i_step == 2) {
          st.i_step = 0;
          st.i_lane_group++;
          if (st.i_lane_group == DIM_LANES / LANE_PAR) {
            st.i_lane_group = 0;
          }
        }
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Mul the output to the delayed output buffers
      TOutputWord out_value;
      for (size_t i_lane_par = 0; i_lane_par < LANE_PAR; i_lane_par++) {
        if (!o_data_instant[i_lane_par].empty()) {
          out_value = o_data_instant[i_lane_par].read();
          st.delayed_output[i_lane_par].push(out_value, true);
        } else {
          st.delayed_output[i_lane_par].push(TOutputWord(), false);
        }
      }
    } else {
      // If not firing, push invalid data to maintain pipeline timing
      for (size_t i_lane_par = 0; i_lane_par < LANE_PAR; i_lane_par++) {
        st.delayed_output[i_lane_par].push(TOutputWord(), false);
      }
    }

    // Advance the actor status
    st.actor_status.advance();

    // Read from the delayed output buffers
    TOutputWord out_value;
    for (size_t i_lane_par = 0; i_lane_par < LANE_PAR; i_lane_par++) {
      if (st.delayed_output[i_lane_par].pop(out_value)) {
        o_data[i_lane_par].write(out_value);
      }
    }

    return st.actor_status;
  }

  StreamingOnlineSoftmax() = default;

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[LANE_PAR],
           const TLUT lut_table[LUT_SIZE],
           hls::stream<TOutputWord> o_data[LANE_PAR]) {
    
    
    TInputWord in_row[LANE_PAR][DIM_REDUCTION / REDUCE_PAR];

    // Loop over groups of lanes.
    for (size_t i_lane_group = 0;
         i_lane_group < DIM_LANES / LANE_PAR; i_lane_group++) {
      
      // Tracking the max and sum for each lane in the group.
      TInput max[LANE_PAR] = {LimitsImpl<TInput>::min()};
      TAcc sum[LANE_PAR] = {0};

      // Loop over the two steps of the softmax computation for each group of lanes.
      for (size_t i_step = 0; i_step < 2; i_step++) {

        // Loop over the groups of unrolled operations in the lane.
        for (size_t i_red_group = 0; i_red_group < DIM_REDUCTION / REDUCE_PAR;
             i_red_group++) {
        STREAMINGSOFTMAX_RUN_LOOP:
#pragma HLS PIPELINE II = 1
          StreamingOnlineSoftmax::pipeline_body(
              i_data, lut_table, o_data, i_red_group, i_step, max, sum, in_row);
        }
      }
    }
  }

private:
  static void
  pipeline_body(hls::stream<TInputWord> i_data[LANE_PAR],
                const TLUT lut_table[LUT_SIZE],
                hls::stream<TOutputWord> o_data[LANE_PAR], size_t i_red_group,
                size_t i_step, TInput max[LANE_PAR], TAcc sum[LANE_PAR],
                TInputWord in_row[LANE_PAR][DIM_REDUCTION / REDUCE_PAR]) {
#pragma HLS inline

    // Output quantizer
    Quantizer quantizer;

    // Precision of the exponantial function.
    const unsigned int exp_precision = TLUT::width; // in bits, e.g., 16 for Q0.16 format
    const unsigned int div_precision = TDiv::width - exp_precision;

    // For rescaling the sum when the max is updated. Q0.16 format for alpha,
    // Q0.32 format for sum, output in Q0.16 format
    typedef ap_uint<TAcc::width + exp_precision> TRescaledSum;
    DequantQuantPo2<exp_precision, TRescaledSum, TAcc> rescaler;

    // The maximum value stored in the LUT, representing 1.0 in the exponential function. 
    const unsigned int lut_max_input = (1U << exp_precision) - 1;

    // The address type for the LUT, which is based on the input type width
    // but it must be unsigned.
    typedef ap_uint<TInput::width> TAddress;

    for (size_t i_lane_par = 0; i_lane_par < LANE_PAR; i_lane_par++) {
    
      if (i_step == 0) {
        // Read input values for the current channel partition
        in_row[i_lane_par][i_red_group] = i_data[i_lane_par].read();
      }
      TInputWord in_value = in_row[i_lane_par][i_red_group];
      TOutputWord out_value;

      // Perform the softmax computation for the input value and write to output
      for (size_t i_red_par = 0; i_red_par < REDUCE_PAR; i_red_par++) {
        TInput x = (TInput)in_value[i_red_par];
        if (i_step == 0) {
          // Update max and sum for this line

          if (x > max[i_lane_par]) {
            // Compute the address for the LUT based on the difference between
            // the current max and the input value It is for sure positive since
            // x > max[i_lane_par]
            TAddress diff = x - max[i_lane_par];

            // Read the LUT value for the current difference
            TLUT alpha = lut_table[diff];

            // Rescale the accumulated sum with the new base.
            // Alpha is in Q0.16 format, sum in UQ16.16 format, so the result is in Q?.32
            // format. Sum is on 32 bits, multiplying by alpha (16 bits) gives
            // 48 bits
            TRescaledSum scaled_sum = (sum[i_lane_par] * alpha);

            // Rescale back to Q?.16 format
            sum[i_lane_par] = rescaler(scaled_sum);
            
            // Add 1 in Q0.16 format for the new max term.
            sum[i_lane_par] = sum[i_lane_par] + lut_max_input;

            // Update the max value
            max[i_lane_par] = x;

          } else {

            // Compute the address for the LUT based on the difference between
            // the current max and the input value
            TAddress diff = max[i_lane_par] - x;

            // Read the LUT value for the current difference and accumulate it
            // to the sum
            TLUT exp = lut_table[diff];

            // Accumulate the exponentials in Q0.16 format
            sum[i_lane_par] = sum[i_lane_par] + exp;
          }

        } else if (i_step == 1) {
          // Compute the address for the LUT based on the difference between
          // the current max and the input value
          TAddress diff = max[i_lane_par] - x;

          // Read the LUT value for the current difference and accumulate it
          // to the sum
          TLUT exp = lut_table[diff];

          // We scale the exponential by 2^32 to maintain precision during the
          // division, which gives us a Q0.48 fixed-point format for the
          // exponential. The sum is also in Q0.16 format, so the division
          // result is in Q0.32 format. The quantizer will then convert this to
          // the output format.
          TDiv div_result = ((TDiv)exp << div_precision) / sum[i_lane_par];
          out_value[i_red_par] = quantizer(div_result);
        }
      }
      if (i_step == 1) {
        o_data[i_lane_par].write(out_value);
      }
    }
  }
};

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename TLUT, typename TAcc, typename TDiv,
          typename Quantizer, size_t LUT_SIZE, size_t DIM_LANES,
          size_t DIM_REDUCTION, size_t LANE_PAR, size_t REDUCE_PAR>
class StreamingSoftmax {

  struct StepState {
    // Loop iteration indexes.
    size_t i_lane_group = 0, i_red_group = 0, i_step = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output[LANE_PAR];
    TInputWord in_row[LANE_PAR][DIM_REDUCTION / REDUCE_PAR];
    TInput max[LANE_PAR];
    TAcc sum[LANE_PAR];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < LANE_PAR; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status = ActorStatus(depth, DIM_LANES * DIM_REDUCTION * 3 /
                                            (REDUCE_PAR * LANE_PAR));
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

  ActorStatus step(hls::stream<TInputWord> i_data[LANE_PAR],
                   const TLUT lut_table[LUT_SIZE],
                   hls::stream<TOutputWord> o_data[LANE_PAR]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_lane_par = 0; i_lane_par < LANE_PAR; i_lane_par++) {
      if (st.i_step == 0 && i_data[i_lane_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> o_data_instant[LANE_PAR];
      StreamingSoftmax::pipeline_body(i_data, lut_table, o_data_instant,
                                      st.i_red_group, st.i_step, st.max, st.sum,
                                      st.in_row);

      // Update iterators
      st.i_red_group++;
      if (st.i_red_group == DIM_REDUCTION / REDUCE_PAR) {
        st.i_red_group = 0;
        st.i_step++;
        if (st.i_step == 3) {
          st.i_step = 0;
          st.i_lane_group++;
          if (st.i_lane_group == DIM_LANES / LANE_PAR) {
            st.i_lane_group = 0;
          }
        }
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Mul the output to the delayed output buffers
      TOutputWord out_value;
      for (size_t i_lane_par = 0; i_lane_par < LANE_PAR; i_lane_par++) {
        if (!o_data_instant[i_lane_par].empty()) {
          out_value = o_data_instant[i_lane_par].read();
          st.delayed_output[i_lane_par].push(out_value, true);
        } else {
          st.delayed_output[i_lane_par].push(TOutputWord(), false);
        }
      }
    } else {
      // If not firing, push invalid data to maintain pipeline timing
      for (size_t i_lane_par = 0; i_lane_par < LANE_PAR; i_lane_par++) {
        st.delayed_output[i_lane_par].push(TOutputWord(), false);
      }
    }

    // Advance the actor status
    st.actor_status.advance();

    // Read from the delayed output buffers
    TOutputWord out_value;
    for (size_t i_lane_par = 0; i_lane_par < LANE_PAR; i_lane_par++) {
      if (st.delayed_output[i_lane_par].pop(out_value)) {
        o_data[i_lane_par].write(out_value);
      }
    }

    return st.actor_status;
  }

  StreamingSoftmax() = default;

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[LANE_PAR],
           const TLUT lut_table[LUT_SIZE],
           hls::stream<TOutputWord> o_data[LANE_PAR]) {
    
    
    TInputWord in_row[LANE_PAR][DIM_REDUCTION / REDUCE_PAR];

    // Loop over groups of lanes.
    for (size_t i_lane_group = 0;
         i_lane_group < DIM_LANES / LANE_PAR; i_lane_group++) {
      
      // Tracking the max and sum for each lane in the group.
      TInput max[LANE_PAR] = {LimitsImpl<TInput>::min()};
      TAcc sum[LANE_PAR] = {0};

      // Loop over the three steps of the softmax computation for each group of lanes.
      for (size_t i_step = 0; i_step < 3; i_step++) {

        // Loop over the groups of unrolled operations in the lane.
        for (size_t i_red_group = 0; i_red_group < DIM_REDUCTION / REDUCE_PAR;
             i_red_group++) {
        STREAMINGSOFTMAX_RUN_LOOP:
#pragma HLS PIPELINE II = 1
          StreamingSoftmax::pipeline_body(
              i_data, lut_table, o_data, i_red_group, i_step, max, sum, in_row);
        }
      }
    }
  }

private:
  static void
  pipeline_body(hls::stream<TInputWord> i_data[LANE_PAR],
                const TLUT lut_table[LUT_SIZE],
                hls::stream<TOutputWord> o_data[LANE_PAR], size_t i_red_group,
                size_t i_step, TInput max[LANE_PAR], TAcc sum[LANE_PAR],
                TInputWord in_row[LANE_PAR][DIM_REDUCTION / REDUCE_PAR]) {
#pragma HLS inline

    // Output quantizer
    Quantizer quantizer;

    // Precision of the exponantial function.
    const unsigned int exp_precision = TLUT::width; // in bits, e.g., 16 for Q0.16 format
    const unsigned int div_precision = TDiv::width - exp_precision;

    // The address type for the LUT, which is based on the input type width
    // but it must be unsigned.
    typedef ap_uint<TInput::width> TAddress;

    for (size_t i_lane_par = 0; i_lane_par < LANE_PAR; i_lane_par++) {

      if (i_step == 0) {
        // Read input values for the current channel partition
        in_row[i_lane_par][i_red_group] = i_data[i_lane_par].read();
      }
      TInputWord in_value = in_row[i_lane_par][i_red_group];
      TOutputWord out_value;

      // Perform the softmax computation for the input value and write to output
      for (size_t i_red_par = 0; i_red_par < REDUCE_PAR; i_red_par++) {
        TInput x = (TInput)in_value[i_red_par];
        if (i_step == 0) {
          // Find the max value for this lane across the reduction dimension

          if (x > max[i_lane_par]) {
            max[i_lane_par] = x;
          }
        } else if (i_step == 1) {
          // Update sum for this lane

          // Compute the address for the LUT based on the difference between
          // the current max and the input value
          TAddress diff = max[i_lane_par] - x;

          // Read the LUT value for the current difference and accumulate it
          // to the sum
          TLUT exp = lut_table[diff];

          // Accumulate the exponentials in Q0.16 format
          sum[i_lane_par] = sum[i_lane_par] + exp;

        } else if (i_step == 2) {
          // Compute the address for the LUT based on the difference between
          // the current max and the input value
          TAddress diff = max[i_lane_par] - x;

          // Read the LUT value for the current difference and accumulate it
          // to the sum
          TLUT exp = lut_table[diff];

          // We scale the exponential by 2^32 to maintain precision during the
          // division, which gives us a Q0.48 fixed-point format for the
          // exponential. The sum is also in Q0.16 format, so the division
          // result is in Q0.32 format. The quantizer will then convert this to
          // the output format.
          TDiv div_result = ((TDiv)exp << div_precision) / sum[i_lane_par];
          out_value[i_red_par] = quantizer(div_result);
        }
      }
      if (i_step == 2) {
        o_data[i_lane_par].write(out_value);
      }
    }
  }
};