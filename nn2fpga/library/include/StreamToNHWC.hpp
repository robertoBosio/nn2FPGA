#pragma once
#include "ap_int.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>
#include <cassert>

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

  StreamToNHWC() = default;

  template <size_t HLS_TAG>
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

  struct StepState {
    // Circular buffer to hold output data for processing.
    TInput circular_buffer[DATA_PER_WORD * 2];

    // Indexes and size for the circular buffer.
    char head = 0, tail = 0, size = 0;

    // Loop iteration index for the input word.
    size_t i_input_word = 0;
    ActorStatus actor_status{1, 1};
    PipelineDelayBuffer<TOutputWord> delayed_output;
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      delayed_output = PipelineDelayBuffer<TOutputWord>(depth);
      actor_status = ActorStatus(depth, ITER / (IN_W_PAR * IN_CH_PAR));
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

  ActorStatus step(hls::stream<TInputWord> input_data_stream[IN_W_PAR],
                   hls::stream<TOutputWord> &output_data_stream) {

    // Find the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() &&
           "step_init() must be called before step()");
    StepState &st = it->second;

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
                                  st.circular_buffer, st.head, st.size, st.tail,
                                  st.i_input_word);
      st.i_input_word += IN_CH_PAR * IN_W_PAR;
      if (st.i_input_word >= ITER) {
        st.i_input_word = 0;
      }

      st.actor_status.fire(); // Fire the actor status.

      // Add the output to the delayed output stream.
      if (!instant_output_stream.empty()) {
        st.delayed_output.push(instant_output_stream.read(), true);
      } else {
        st.delayed_output.push(TOutputWord(),
                               false); // Placeholder, ignored
      }
    } else {
      // If the firing condition is not met, push a placeholder to maintain the
      // pipeline depth.
      st.delayed_output.push(TOutputWord(), false);
    }

    // Advance the actor status.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    if (st.delayed_output.pop(out)) {
      output_data_stream.write(out);
    }

    return st.actor_status; // Return the current actor status.
  }

private:

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
    const bool end_of_tensor = (i_input_word == ITER - (IN_W_PAR * IN_CH_PAR));
    if (size >= DATA_PER_WORD || end_of_tensor) {

      // If we have enough data to form an output word, proceed with packing.
      TOutputWord output_data;
      for (size_t i = 0; i < DATA_PER_WORD; i++) {
        output_data.data.range((i + 1) * TInput::width - 1,
                               i * TInput::width) =
            quantizer(circular_buffer[tail + i]);
      }

      if (end_of_tensor) {
        size_t valid_bytes = size * TInput::width / 8;
        output_data.keep = (1 << valid_bytes) - 1;
        tail = 0; // Reset the tail at the end of the tensor.
        size = 0; // Reset the size at the end of the tensor.
        head = 0; // Reset the head at the end of the tensor.
        output_data.last = true;
      } else {
        tail = (tail + DATA_PER_WORD) % (DATA_PER_WORD * 2);
        size -= DATA_PER_WORD;
        output_data.last = false;
        output_data.keep = ~0; // Set all bytes as valid.
      }

      output_data.strb = output_data.keep;
      output_data_stream.write(output_data);
    }
  }
};