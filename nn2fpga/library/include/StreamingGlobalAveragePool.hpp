#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>

/**
 * @class StreamingGlobalAveragePool
 * @brief Implements a streaming global average pooling operation for
 * HWC-formatted data.
 *
 * This class performs global average pooling in a streaming fashion.
 * The input data is expected in HWC format. The pooling operation accumulates
 * values across the height and width dimensions for each channel, thus an
 * accumulator is needed for each output, then computes the average.
 *
 * @tparam TInputWord   Structure type for input data (vectorized
 * input).
 * @tparam TInput         Scalar type for input elements.
 * @tparam TOutputWord  Structure type for output data (vectorized
 * output).
 * @tparam TOutput        Scalar type for output elements.
 * @tparam TAcc           Accumulator type for intermediate sum.
 * @tparam TDiv           Type used for division in averaging.
 * @tparam Quantizer      Quantizer functor/class for output quantization.
 * @tparam IN_HEIGHT      Input height (number of rows).
 * @tparam IN_WIDTH       Input width (number of columns).
 * @tparam OUT_CH         Number of output channels.
 * @tparam OUT_CH_PAR     Number of output channels processed in parallel.
 *
 * @note
 * - OUT_CH must be a multiple of OUT_CH_PAR.
 * - IN_HEIGHT, IN_WIDTH, and OUT_CH_PAR must be greater than 0.
 *
 * @section Usage
 * - Use the run() method for functional verification and synthesis.
 * - Use the step() method for self-timed execution with actor status tracking,
 * which is needed for fifo depth estimation.
 *
 * @section Parallelism
 * The class supports parallel processing of output channels, as specified by
 * OUT_CH_PAR. It does not support parallel processing of width, so only one
 * stream is used for input/output.
 *
 * @section Quantization
 * The integer division is can introduce a small rounding error, since it
 * does not round ties to the nearest even number. This is a trade-off
 * between accuracy and performance, as rounding to the nearest even number
 * would require a modulo operation, which is expensive in hardware.
 */

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename TAcc, typename TDiv, typename Quantizer,
          size_t IN_HEIGHT, size_t IN_WIDTH, size_t OUT_CH, size_t OUT_CH_PAR>
class StreamingGlobalAveragePool {
public:
  static_assert(OUT_CH % OUT_CH_PAR == 0,
                "OUT_CH must be a multiple of OUT_CH_PAR");
  static_assert(OUT_CH_PAR > 0, "OUT_CH_PAR must be greater than 0");
  static_assert(IN_HEIGHT > 0 && IN_WIDTH > 0,
                "IN_HEIGHT and IN_WIDTH must be greater than 0");

  StreamingGlobalAveragePool() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_hw = 0, i_och = 0;

    // Accumulator buffer for each output channel.
    TAcc s_acc_buff[OUT_CH / OUT_CH_PAR][OUT_CH_PAR];

    PipelineDelayBuffer<TOutputWord> delayed_output;
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      delayed_output = PipelineDelayBuffer<TOutputWord>(depth);
      actor_status =
          ActorStatus(depth, IN_HEIGHT * IN_WIDTH * OUT_CH / OUT_CH_PAR);
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
  void run(hls::stream<TInputWord> i_data[1],
           hls::stream<TOutputWord> o_data[1]) {
    TAcc s_acc_buff[OUT_CH / OUT_CH_PAR]
                   [OUT_CH_PAR]; // Accumulator buffer for each output channel.
#pragma HLS array_partition variable = s_acc_buff dim = 2

    // Loop through the input height and width.
    for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH; i_hw++) {
      // Loop through the output channels, with a step size equal to the number
      // of channels processed in parallel.
    STREAMINGGLOBALAVERAGEPOOL_RUN_LOOP:
      for (size_t i_och = 0; i_och < OUT_CH / OUT_CH_PAR; i_och++) {
#pragma HLS pipeline II = 1
        StreamingGlobalAveragePool::pipeline_body(i_data, o_data,
                                                  s_acc_buff[i_och], i_hw);
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[1],
                   hls::stream<TOutputWord> o_data[1]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = !i_data[0].empty();

    if (firing_condition) {

      // If there is data in the input stream, process it.
      hls::stream<TOutputWord> instant_output_stream[1];
      StreamingGlobalAveragePool::pipeline_body(
          i_data, instant_output_stream, st.s_acc_buff[st.i_och], st.i_hw);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      if (!instant_output_stream[0].empty()) {
        st.delayed_output.push(instant_output_stream[0].read(), true);
      } else {
        st.delayed_output.push(TOutputWord(),
                               false); // Placeholder, ignored
      }

      // Update the counters.
      st.i_och++;
      if (st.i_och >= OUT_CH / OUT_CH_PAR) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        st.i_och = 0;
        st.i_hw++;
      }
      if (st.i_hw >= IN_HEIGHT * IN_WIDTH) {
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

private:
  static void pipeline_body(hls::stream<TInputWord> i_data[1],
                            hls::stream<TOutputWord> o_data[1],
                            TAcc s_acc_buff[OUT_CH_PAR], size_t i_hw) {
#pragma HLS inline
    TOutputWord s_output_struct; // Output structure to hold the results.
    TInputWord
        s_input_struct;  // Input structure to read data from the input stream.
    Quantizer quantizer; // Quantizer instance for quantization.

    // Loop through the channels processed in parallel.
    for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR; i_och_par++) {

      // Initializing the accumulator for each window.
      if (i_hw == 0) {
        s_acc_buff[i_och_par] = 0;
      }

      // Reading packets of OUT_CH_PAR channels.
      if (i_och_par == 0) {
        s_input_struct = i_data[0].read();
      }

      // Accumulating the input data for the current output channel.
      s_acc_buff[i_och_par] += s_input_struct[i_och_par];

      // Writing the output at the end of the window
      if (i_hw == (IN_HEIGHT * IN_WIDTH - 1)) {
        TDiv divisor =
            IN_HEIGHT * IN_WIDTH; // Divisor for the average calculation.

        // Round the accumulated value to the nearest integer.
        // This is not strictly correct, as ties should be rounded to the
        // nearest even number, but it requires the use of a modulo operation,
        // which is quite expensive. Instead, we are rounding ties up.
        // TAcc bias = (s_acc_buff[i_och_par] >= 0) ? (TAcc)(divisor >> 1)
        //                                          : (TAcc) - (divisor >> 1);
        // TAcc rounded_value = s_acc_buff[i_och_par] + bias;
        // TAcc result = rounded_value / divisor; // Calculate the average.

        // Potential logic for a rounding to the nearest even number
        TAcc quotient = s_acc_buff[i_och_par] / divisor;
        ap_int<TDiv::width + 1> remainder = s_acc_buff[i_och_par] % divisor;
        ap_int<TDiv::width + 2> double_remainder = remainder * 2;
        if (double_remainder > divisor || (double_remainder == divisor && (quotient & 1))) {
          quotient += 1;
        }
        if (double_remainder < -divisor || (double_remainder == -divisor && (quotient & 1))) {
          quotient -= 1;
        }
        TAcc result = quotient;

        s_output_struct[i_och_par] = quantizer(result);
        if (i_och_par == (OUT_CH_PAR - 1)) {
          o_data[0].write(s_output_struct);
        }
      }
    }
  }
};