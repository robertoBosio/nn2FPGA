#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
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
 * @tparam TInputStruct   Structure type for input data (vectorized
 * input).
 * @tparam TInput         Scalar type for input elements.
 * @tparam TOutputStruct  Structure type for output data (vectorized
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

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, typename TAcc, typename TDiv, typename Quantizer,
          size_t IN_HEIGHT, size_t IN_WIDTH, size_t OUT_CH, size_t OUT_CH_PAR>
class StreamingGlobalAveragePool {
public:
  static_assert(OUT_CH % OUT_CH_PAR == 0,
                "OUT_CH must be a multiple of OUT_CH_PAR");
  static_assert(OUT_CH_PAR > 0, "OUT_CH_PAR must be greater than 0");
  static_assert(IN_HEIGHT > 0 && IN_WIDTH > 0,
                "IN_HEIGHT and IN_WIDTH must be greater than 0");

  StreamingGlobalAveragePool() : StreamingGlobalAveragePool(1) {}

  StreamingGlobalAveragePool(size_t pipeline_depth)
      : STEP_pipeline_depth(pipeline_depth), STEP_i_hw(0), STEP_i_och(0),
        STEP_actor_status(pipeline_depth,
                          IN_HEIGHT * IN_WIDTH * OUT_CH / OUT_CH_PAR),
        STEP_delayed_output(pipeline_depth) {}

  void run(hls::stream<TInputStruct> i_data[1],
           hls::stream<TOutputStruct> o_data[1]) {
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
        StreamingGlobalAveragePool::pipeline_body(i_data, o_data, s_acc_buff[i_och],
                                                  i_hw, i_och);
      }
    }
  }

  ActorStatus
  step(hls::stream<TInputStruct> i_data[1],
       hls::stream<TOutputStruct> o_data[1]) {

    if (!i_data[0].empty()) {

      // If there is data in the input stream, process it.
      hls::stream<TOutputStruct> output_stream[1];
      StreamingGlobalAveragePool::pipeline_body(
          i_data, output_stream, STEP_s_acc_buff[STEP_i_och], STEP_i_hw, STEP_i_och);

      // Insert new firing status into the multiset.
      STEP_actor_status.fire();

      // Add the output to the delayed output stream.
      if (!output_stream[0].empty()) {
        STEP_delayed_output.push(output_stream[0].read(), true);
      } else {
        STEP_delayed_output.push(TOutputStruct(),
                                 false); // Placeholder, ignored
      }

      // Update the counters.
      STEP_i_och++;
      if (STEP_i_och >= OUT_CH / OUT_CH_PAR) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        STEP_i_och = 0;
        STEP_i_hw++;
      }
      if (STEP_i_hw >= IN_HEIGHT * IN_WIDTH) {
        STEP_i_hw = 0; // Reset the height/width index if we have processed all
                       // iterations.
      }

    } else {
      // If there is no data in the input stream, push a delay slot.
      STEP_delayed_output.push(TOutputStruct(), false);
    }

    // Advance the state of the actor firings.
    STEP_actor_status.advance();

    // Write the output data to the output stream.
    TOutputStruct out;
    if (STEP_delayed_output.pop(out)) {
      o_data[0].write(out);
    }

    // Return the actor status.
    return STEP_actor_status;
  }

private:
  // State variables for step execution
  TAcc STEP_s_acc_buff[OUT_CH / OUT_CH_PAR][OUT_CH_PAR];   // Accumulator buffer for each output channel
  size_t STEP_i_hw;               // Current height and width index
  size_t STEP_i_och;              // Current output channel index
  size_t STEP_pipeline_depth = 1; // Pipeline depth for the actor

  // CSDFG actor state variables
  ActorStatus STEP_actor_status;

  // Pipeline state variables
  // Delayed output buffer to maintain pipeline depth
  PipelineDelayBuffer<TOutputStruct> STEP_delayed_output;

  static void pipeline_body(hls::stream<TInputStruct> i_data[1],
                            hls::stream<TOutputStruct> o_data[1],
                            TAcc s_acc_buff[OUT_CH_PAR], size_t i_hw,
                            size_t i_och) {
#pragma HLS inline
    TOutputStruct s_output_struct; // Output structure to hold the results.
    TInputStruct
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
        TAcc bias = (s_acc_buff[i_och_par] >= 0) ? (TAcc)(divisor >> 1) : (TAcc)-(divisor >> 1);
        TAcc rounded_value = s_acc_buff[i_och_par] + bias;
        TAcc result = rounded_value / divisor; // Calculate the average.

        // Potential logic for a rounding to the nearest even number
        // TAcc quotient = acc / div;
        // TDiv remainder = acc % div;
        // TDiv half = div >> 1;
        // if (remainder > half || (remainder == half && (quotient & 1))) {
        //   quotient += 1;
        // }
        // if (remainder < -half || (remainder == -half && (quotient & 1))) {
        //   quotient -= 1;
        // }
        // TAcc result = quotient;

        s_output_struct[i_och_par] = quantizer(result);
        if (i_och_par == (OUT_CH_PAR - 1)) {
          o_data[0].write(s_output_struct);
        }
      }
    }
  }
};