#pragma once
#include "ap_int.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>

/**
 * @class StreamToNHWC
 * @brief StreamToNHWC consumes input data streams, quantizes the data, and
 * packs it into words for an AXI stream.
 *
 * This class is designed to handle the consumption of nn2FPGA input data
 * streams and convert them into an AXI stream format. It supports parallel
 * processing of input channels and width, as specified by IN_CH_PAR and
 * IN_W_PAR, respectively.
 *
 * @tparam TInputWord     The type of the input data stream.
 * @tparam TInput         The data type of the input elements.
 * @tparam TOutputWord    The type of the output data stream.
 * @tparam TOutput        The data type of the output elements.
 * @tparam Quantizer      The quantizer functor/class used to quantize input
 * data.
 * @tparam DATA_PER_WORD  Number of data elements packed into a single output
 * word.
 * @tparam BITS_PER_DATA  Number of bits used to represent each data element.
 * @tparam HEIGHT         Height of the input tensor.
 * @tparam WIDTH          Width of the input tensor.
 * @tparam CH             Number of input channels.
 * @tparam IN_W_PAR       Number of input width elements processed in parallel.
 * @tparam IN_CH_PAR      Number of input channels processed in parallel.
 *
 * @note
 * - DATA_PER_WORD must be a multiple of IN_CH_PAR * IN_W_PAR.
 * - If IN_W_PAR > 1, CH must be equal to IN_CH_PAR, this is to preserve
 * the correct order of the data flowing.
 * - CH must be a multiple of IN_CH_PAR.
 * - WIDTH must be a multiple of IN_W_PAR.
 *
 * @section Usage
 * - Use the run() method for functional verification and synthesis.
 * - Use the step() method for self-timed execution with actor status tracking,
 * which is needed for fifo depth estimation.
 *
 * @section Parallelism
 * The class supports parallel processing of input channels and width, as
 * specified by IN_CH_PAR and IN_W_PAR, respectively.
 *
 * @section Quantization
 * The Quantizer template parameter is used to quantize the extracted data
 * before writing to the output stream.
 */

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t DATA_PER_WORD,
          size_t HEIGHT, size_t WIDTH, size_t CH, size_t IN_W_PAR,
          size_t IN_CH_PAR>
class StreamToNHWC {
  static constexpr size_t ITER = HEIGHT * WIDTH * CH;

public:
  static_assert(
      DATA_PER_WORD >= (IN_W_PAR * IN_CH_PAR),
      "DATA_PER_WORD must be bigger or equal to IN_CH_PAR * IN_W_PAR");
  static_assert(IN_W_PAR == 1 || CH == IN_CH_PAR,
                "CH must be equal to IN_CH_PAR when IN_W_PAR > 1");
  static_assert(CH % IN_CH_PAR == 0, "CH must be a multiple of IN_CH_PAR");
  static_assert(WIDTH % IN_W_PAR == 0, "WIDTH must be a multiple of IN_W_PAR");

  StreamToNHWC() : StreamToNHWC(1) {}

  StreamToNHWC(size_t pipeline_depth)
      : STEP_pipeline_depth(pipeline_depth), STEP_i_input_word(0), STEP_head(0),
        STEP_tail(0), STEP_size(0),
        STEP_actor_status(pipeline_depth, ITER / (IN_CH_PAR * IN_W_PAR)),
        STEP_delayed_output(pipeline_depth) {}

  void run(hls::stream<TInputWord> input_data_stream[IN_W_PAR],
           hls::stream<TOutputWord> &output_data_stream) {
    TInput circular_buffer[DATA_PER_WORD * 2];
    char head = 0;
    char tail = 0;
    char size = 0;

    // Loop through the input height and width.
STREAM_TO_NHWC_MAINLOOP:
    for (size_t i_input_word = 0; i_input_word < ITER;
         i_input_word += IN_CH_PAR * IN_W_PAR) {
#pragma HLS pipeline II = 1
      StreamToNHWC::pipeline_body(input_data_stream, output_data_stream,
                                  circular_buffer, head, size, tail,
                                  i_input_word);
    }
  }

  ActorStatus step(hls::stream<TInputWord> input_data_stream[IN_W_PAR],
                   hls::stream<TOutputWord> &output_data_stream) {

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_w_par = 0; i_w_par < IN_W_PAR; i_w_par++) {
      if (input_data_stream[i_w_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> instant_output_stream;
      StreamToNHWC::pipeline_body(input_data_stream, instant_output_stream,
                                  circular_buffer, STEP_head, STEP_size,
                                  STEP_tail, STEP_i_input_word);
      STEP_i_input_word += IN_CH_PAR * IN_W_PAR;
      if (STEP_i_input_word >= ITER) {
        STEP_i_input_word = 0;
      }

      STEP_actor_status.fire(); // Fire the actor status.

      // Add the output to the delayed output stream.
      if (!instant_output_stream.empty()) {
        STEP_delayed_output.push(instant_output_stream.read(), true);
      } else {
        STEP_delayed_output.push(TOutputWord(),
                                 false); // Placeholder, ignored
      }
    } else {
      // If the firing condition is not met, push a placeholder to maintain the
      // pipeline depth.
      STEP_delayed_output.push(TOutputWord(), false);
    }

    // Advance the actor status.
    STEP_actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    if (STEP_delayed_output.pop(out)) {
      output_data_stream.write(out);
    }

    return STEP_actor_status; // Return the current actor status.
  }

private:
  // State variables for step execution
  size_t STEP_i_input_word;           // Current word index
  size_t STEP_pipeline_depth;   // Pipeline depth for the step
  char STEP_head;            // Head index for circular buffer
  char STEP_tail;            // Tail index for circular buffer
  char STEP_size;            // Current size of data in circular buffer
  TInput circular_buffer[DATA_PER_WORD * 2]; // Circular buffer for input data

  // CSDFG state variables
  ActorStatus STEP_actor_status;
  PipelineDelayBuffer<TOutputWord> STEP_delayed_output;

  static void pipeline_body(hls::stream<TInputWord> input_data_stream[IN_W_PAR],
                            hls::stream<TOutputWord> &output_data_stream,
                            TInput circular_buffer[DATA_PER_WORD * 2],
                            char &head, char &size, char &tail,
                            size_t i_input_word) {
#pragma HLS inline
    Quantizer quantizer; // Quantizer instance for quantization.

    // Loop through the pixels processed in parallel.
    for (size_t i_w_par = 0; i_w_par < IN_W_PAR; i_w_par++) {
      TInputWord s_input_struct = input_data_stream[i_w_par].read();
      for (size_t i_och_par = 0; i_och_par < IN_CH_PAR; i_och_par++) {
        circular_buffer[head] = s_input_struct[i_och_par];
        head = (head + 1) % (DATA_PER_WORD * 2);
      }
    }
    size += IN_W_PAR * IN_CH_PAR;

    // Check if we have enough data to form an output word or if we are at the
    // end of the tensor.
    if (size >= DATA_PER_WORD || i_input_word == ITER - (IN_W_PAR * IN_CH_PAR)) {

      // If we have enough data to form an output word, proceed with packing.
      TOutputWord output_data;
      for (size_t i = 0; i < DATA_PER_WORD; i++) {
        output_data.data.range((i + 1) * TInput::width - 1,
                               i * TInput::width) =
            quantizer(circular_buffer[tail + i]);
      }
      tail = (tail + DATA_PER_WORD) % (DATA_PER_WORD * 2);
      size -= DATA_PER_WORD;

      if (i_input_word == ITER - (IN_W_PAR * IN_CH_PAR)) {
        // If we are at the end of the tensor, assert the last.
        output_data.last = true;
      } else {
        // Otherwise, set the last signal to false.
        output_data.last = false;
      }

      // Write the output data structure to the output stream.
      output_data.keep = ~0; // Set all bytes as valid.
      output_data.strb = output_data.keep;
      output_data_stream.write(output_data);
    }
  }
};