#pragma once
#include "ap_int.h"
#include "ap_float.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include "utils/HLS_utils.hpp"
#include <cassert>
#include <cstddef>

/**
 * @class StreamToAXI
 * @brief StreamToAXI consumes input data streams, quantizes the data, and
 * packs it into words for an AXI stream.
 *
 * This class is designed to handle the consumption of nn2FPGA input data
 * streams and convert them into an AXI stream format. It supports parallel
 * processing of input channels and width, as specified by DIM2_UNROLL and
 * DIM1_UNROLL, respectively.
 *
 * @tparam TInputWord     The type of the input data stream.
 * @tparam TInput         The data type of the input elements.
 * @tparam TOutputWord    The type of the output data stream.
 * @tparam Quantizer      The quantizer functor/class used to quantize input
 * data.
 * @tparam ITER           Number of input data elements to process, rounded up
 * to the nearest multiple of DATA_PER_WORD.
 * @tparam DATA_PER_WORD  Number of data elements packed into a single output
 * word.
 * @tparam BITS_PER_DATA  Number of bits used to represent each data element.
 * @tparam DIM0         Height of the input tensor.
 * @tparam DIM1          Width of the input tensor.
 * @tparam DIM2             Number of input channels.
 * @tparam DIM1_UNROLL       Number of input width elements processed in
 * parallel.
 * @tparam DIM2_UNROLL      Number of input channels processed in parallel.
 *
 * @note
 * - DATA_PER_WORD must be a multiple of DIM2_UNROLL * DIM1_UNROLL.
 * - If DIM1_UNROLL > 1, DIM2 must be equal to DIM2_UNROLL, this is to preserve
 * the correct order of the data flowing.
 * - DIM2 must be a multiple of DIM2_UNROLL.
 * - DIM1 must be a multiple of DIM1_UNROLL.
 *
 * @section Usage
 * - Use the run() method for functional verification and synthesis.
 * - Use the step() method for self-timed execution with actor status tracking,
 * which is needed for fifo depth estimation.
 *
 * @section Parallelism
 * The class supports parallel processing of input channels and width, as
 * specified by DIM2_UNROLL and DIM1_UNROLL, respectively.
 *
 * @section Quantization
 * The Quantizer template parameter is used to quantize the extracted data
 * before writing to the output stream.
 */

template <typename TInputWord, typename TInput,
          typename TOutputWord, typename Quantizer, size_t ITER,
          size_t DATA_PER_WORD, size_t DIM0, size_t DIM1, size_t DIM2,
          size_t DIM1_UNROLL, size_t DIM2_UNROLL>
class StreamToAXI {
  static constexpr size_t READS =
      DIM0 * DIM1 * DIM2 / (DIM1_UNROLL * DIM2_UNROLL);
  static_assert(
      DATA_PER_WORD >= (DIM1_UNROLL * DIM2_UNROLL),
      "DATA_PER_WORD must be bigger or equal to DIM2_UNROLL * DIM1_UNROLL");
  static_assert(DIM1_UNROLL == 1 || DIM2 == DIM2_UNROLL,
                "DIM2 must be equal to DIM2_UNROLL when DIM1_UNROLL > 1");
  static_assert(DIM2 % DIM2_UNROLL == 0,
                "DIM2 must be a multiple of DIM2_UNROLL");
  static_assert(DIM1 % DIM1_UNROLL == 0,
                "DIM1 must be a multiple of DIM1_UNROLL");

  struct StepState {
    // Circular buffer to hold output data for processing.
    TInput circular_buffer[DATA_PER_WORD * 2];

    // Indexes and size for the circular buffer.
    ap_uint<bits_for(DATA_PER_WORD * 2)> head = 0;
    ap_uint<1> tail = 0;
    ap_uint<bits_for(DATA_PER_WORD * 2 + 1)> size = 0;

    // Loop iteration index for the input word.
    size_t i_input_word = 0;
    ActorStatus actor_status{1, 1};
    PipelineDelayBuffer<TOutputWord> delayed_output;
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      delayed_output = PipelineDelayBuffer<TOutputWord>(depth);
      actor_status = ActorStatus(depth, ITER);
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

public:
  StreamToAXI() = default;

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> input_data_stream[DIM1_UNROLL],
           hls::stream<TOutputWord> &output_data_stream) {
    TInput circular_buffer[DATA_PER_WORD * 2];
    ap_uint<bits_for(DATA_PER_WORD * 2)> head = 0;
    ap_uint<1> tail = 0;
    ap_uint<bits_for((DATA_PER_WORD * 2) + 1)> size = 0;

    // Loop through the input height and width.
  STREAM_TO_NHWC_MAINLOOP:
    for (size_t i_input_word = 0; i_input_word < ITER; i_input_word++) {
#pragma HLS pipeline II = 1
      StreamToAXI::pipeline_body(input_data_stream, output_data_stream,
                                 circular_buffer, head, size, tail,
                                 i_input_word);
    }
  }

  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  ActorStatus step(hls::stream<TInputWord> input_data_stream[DIM1_UNROLL],
                   hls::stream<TOutputWord> &output_data_stream) {

    // Find the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() &&
           "step_init() must be called before step()");
    StepState &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    if (st.i_input_word < READS) {
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        if (input_data_stream[i_dim1_par].empty()) {
          firing_condition = false;
        }
      }
    }

    if (firing_condition) {
      hls::stream<TOutputWord> instant_output_stream;
      StreamToAXI::pipeline_body(input_data_stream, instant_output_stream,
                                 st.circular_buffer, st.head, st.size, st.tail,
                                 st.i_input_word);
      st.i_input_word++;
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
  static void
  pipeline_body(hls::stream<TInputWord> input_data_stream[DIM1_UNROLL],
                hls::stream<TOutputWord> &output_data_stream,
                TInput circular_buffer[DATA_PER_WORD * 2],
                ap_uint<bits_for(DATA_PER_WORD * 2)> &head,
                ap_uint<bits_for((DATA_PER_WORD * 2) + 1)> &size,
                ap_uint<1> &tail_bank, size_t i_input_word) {
#pragma HLS inline
    Quantizer quantizer; // Quantizer instance for quantization.

    // Loop through the pixels processed in parallel.
    const bool end_of_tensor = (i_input_word >= READS);
    if (!end_of_tensor) {
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        TInputWord s_input_struct = input_data_stream[i_dim1_par].read();
        for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL; i_dim2_par++) {
          circular_buffer[head] = s_input_struct[i_dim2_par];
          head = (head + 1) % (DATA_PER_WORD * 2);
        }
      }
      size += DIM1_UNROLL * DIM2_UNROLL;
    }

    // Check if we have enough data to form an output word or if we are at the
    // end of the tensor.
    if (size >= DATA_PER_WORD || end_of_tensor) {
      ap_uint<bits_for(DATA_PER_WORD * 2)> tail = tail_bank ? DATA_PER_WORD : 0;

      // If we have enough data to form an output word, proceed with packing.
      TOutputWord output_data;
      for (size_t i = 0; i < DATA_PER_WORD; i++) {
        output_data.data.range((i + 1) * data_width_v<TInput> - 1, i * data_width_v<TInput>) =
            get_raw_bits(quantizer(circular_buffer[tail + i]));
      }

      if (end_of_tensor) {
        size_t valid_bytes = size * data_width_v<TInput> / 8;
        output_data.keep = (1 << valid_bytes) - 1;
        // tail_bank = 0; // Reset the tail bank at the end of the tensor.
        // size = 0; // Reset the size at the end of the tensor.
        // head = 0; // Reset the head at the end of the tensor.
        output_data.last = true;
      } else {
        tail_bank ^= ap_uint<1>(1);
        size -= DATA_PER_WORD;
        output_data.last = false;
        output_data.keep = ~0; // Set all bytes as valid.
      }

      output_data.strb = output_data.keep;
      output_data_stream.write(output_data);
    }
  }
};