#pragma once
#include "ap_int.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include "utils/HLS_utils.hpp"
#include <cassert>
#include <cstddef>

/**
 * @brief AXIToStream is a templated class for converting AXI streams to the
 * nn2FPGA data format based on the parallelism selected.
 *
 * This class reads AXI input words from an input stream, processes
 * them in parallel according to the specified template parameters, and writes
 * the results to output streams. The main point is to transform a AXI stream
 * into the nn2FPGA data format.
 *
 * @tparam TInputWord      The type of the input word (AXI).
 * @tparam TInput          The type of the input data (AXI).
 * @tparam TOutputWord     The type of the output word.
 * @tparam TOutput         The type of the output data.
 * @tparam Quantizer       The quantizer functor used for quantization.
 * @tparam DATA_PER_WORD   Number of data elements per input word.
 * @tparam DIM0            Input tensor dimension 0 (e.g., height).
 * @tparam DIM1            Input tensor dimension 1 (e.g., width).
 * @tparam DIM2            Input tensor dimension 2 (e.g., channels).
 * @tparam DIM1_UNROLL     Output DIM1 parallelism (number of parallel output
 * streams).
 * @tparam DIM2_UNROLL     Output DIM2 parallelism (number of data in an output
 * word).
 *
 * @note
 * - DATA_PER_WORD must be a multiple of DIM2_UNROLL * DIM1_UNROLL.
 * - If DIM1_UNROLL > 1, DIM2 must be equal to DIM2_UNROLL, this is to preserve
 * the correct order of the data flowing.
 * - DIM2 must be a multiple of DIM2_UNROLL.
 * - DIM1 must be a multiple of DIM1_UNROLL.
 *
 * @section Usage
 * - Use the run() method for functional verification and for synthesis.
 * - Use the step() and step_fixedthroughput() method for self-timed execution
 * with actor status tracking, which is needed for fifo depth estimation.
 *
 * @section Pipeline
 * The class manages pipeline depth and delayed output buffers to ensure correct
 * data propagation and timing in hardware pipelines. This information is only
 * required for self-timed execution and is not used in hardware synthesis.
 *
 * @section Parallelism
 * The class supports parallel processing of output dimensions, as
 * specified by DIM2_UNROLL and DIM1_UNROLL, respectively.
 *
 * @section Quantization
 * The Quantizer template parameter is used to quantize the extracted data
 * before writing to the output stream.
 */

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t DATA_PER_WORD,
          size_t DIM0, size_t DIM1, size_t DIM2, size_t DIM1_UNROLL,
          size_t DIM2_UNROLL, bool ONLY_ONCE = false>
class AXIToStream {
  static constexpr size_t ITER =
      DIM0 * DIM1 * DIM2 / (DIM1_UNROLL * DIM2_UNROLL);

public:
  static_assert(DIM1_UNROLL == 1 || DIM2 == DIM2_UNROLL,
                "DIM2 must be equal to DIM2_UNROLL when DIM1_UNROLL > 1");
  static_assert(DIM2 % DIM2_UNROLL == 0,
                "DIM2 must be a multiple of DIM2_UNROLL");
  static_assert(DIM1 % DIM1_UNROLL == 0,
                "DIM1 must be a multiple of DIM1_UNROLL");

  AXIToStream() = default;

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> &i_data,
           hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
    TOutput circular_buffer[DATA_PER_WORD * 2];
    ap_uint<1> head = 0; // Head index for circular buffer
    ap_uint<bits_for(DATA_PER_WORD * 2)> tail =
        0; // Tail index for circular buffer
    ap_uint<bits_for(DATA_PER_WORD * 2 + 1)> size =
        0; // Current size of the circular buffer
    bool static only_once_flag = false;
    if (!only_once_flag) {

      // Loop through the word packets of the output tensor.
    NHWC_TO_STREAM_MAINLOOP:
      for (size_t i_output_word = 0; i_output_word < ITER; i_output_word++) {
#pragma HLS pipeline II = 1
        AXIToStream::pipeline_body(i_data, o_data, circular_buffer, head, size,
                                   tail);
      }
    }
    if (ONLY_ONCE)
      only_once_flag = true;
  }

  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  ActorStatus step(hls::stream<TInputWord> &i_data,
                   hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {

    // Find the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() &&
           "step_init() must be called before step()");
    StepState &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    if (st.size < DIM2_UNROLL * DIM1_UNROLL && i_data.empty()) {
      firing_condition = false;
    }

    if (firing_condition) {
      hls::stream<TOutputWord> instant_output_stream[DIM1_UNROLL];
      AXIToStream::pipeline_body(i_data, instant_output_stream,
                                 st.circular_buffer, st.head, st.size, st.tail);
      st.i_output_word++;
      if (st.i_output_word >= ITER) {
        st.i_output_word = 0;
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        if (!instant_output_stream[i_dim1_par].empty()) {
          st.delayed_output[i_dim1_par].push(
              instant_output_stream[i_dim1_par].read(), true);
        } else {
          st.delayed_output[i_dim1_par].push(TOutputWord(),
                                             false); // Placeholder, ignored
        }
      }
    } else {

      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; ++i_dim1_par) {
        st.delayed_output[i_dim1_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (st.delayed_output[i_dim1_par].pop(out)) {
        o_data[i_dim1_par].write(out);
      }
    }

    // Return the current firing iteration index.
    return st.actor_status;
  }

private:

  struct StepState {
    // Circular buffer to hold input data for processing.
    TOutput circular_buffer[DATA_PER_WORD * 2];

    // Indexes and size for the circular buffer.
    ap_uint<1> head = 0; // Head index for circular buffer
    ap_uint<bits_for(DATA_PER_WORD * 2)> tail = 0;
    ap_uint<bits_for(DATA_PER_WORD * 2 + 1)> size = 0;

    // Loop iteration index for the output word.
    size_t i_output_word = 0;
    ActorStatus actor_status{1, 1};
    PipelineDelayBuffer<TOutputWord> delayed_output[DIM1_UNROLL];
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      actor_status = ActorStatus(depth, ITER);
      for (size_t i = 0; i < DIM1_UNROLL; ++i)
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

  static void pipeline_body(hls::stream<TInputWord> &i_data,
                            hls::stream<TOutputWord> o_data[DIM1_UNROLL],
                            TOutput circular_buffer[DATA_PER_WORD * 2],
                            ap_uint<1> &head_bank,
                            ap_uint<bits_for(DATA_PER_WORD * 2 + 1)> &size,
                            ap_uint<bits_for(DATA_PER_WORD * 2)> &tail) {
#pragma HLS inline
    Quantizer quantizer; // Quantizer instance for quantization.

    // Read a new input data word if there is not enough data in the circular
    // buffer to output the required parallel data.
    if (size < DIM2_UNROLL * DIM1_UNROLL) {
      TInputWord input_data = i_data.read();
      ap_uint<bits_for(DATA_PER_WORD * 2)> head = head_bank ? DATA_PER_WORD : 0;
      for (size_t i = 0; i < DATA_PER_WORD; i++) {
        circular_buffer[head + i] = input_data.data.range(
            TOutput::width * (i + 1) - 1, TOutput::width * i);
      }
      head_bank ^= ap_uint<1>(1);
      size += DATA_PER_WORD;
    }

    // Loop through the pixels processed in parallel.
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      TOutputWord out_word; // Output word to hold the results.
      for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL; i_dim2_par++) {
        out_word[i_dim2_par] = quantizer(circular_buffer[tail]);
        tail = (tail + 1) % (DATA_PER_WORD * 2);
      }
      // Write the output word to the output stream.
      o_data[i_dim1_par].write(out_word);
    }
    size -= DIM2_UNROLL * DIM1_UNROLL;
  }
};