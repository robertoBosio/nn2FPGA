#pragma once
#include "ap_int.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>
#include <cassert>

/**
 * @brief NHWCToStream is a templated class for streaming data into neural
 * network data, converting AXI streams to the nn2FPGA data format based on the
 * parallelism selected.
 *
 * This class reads AXI input data structures from an input stream, processes
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
 * @tparam HEIGHT          Input tensor height.
 * @tparam WIDTH           Input tensor width.
 * @tparam CH              Input tensor channels.
 * @tparam OUT_W_PAR       Output width parallelism (number of parallel outputs
 * in width).
 * @tparam OUT_CH_PAR      Output channel parallelism (number of parallel
 * outputs in channels).
 *
 * @note
 * - DATA_PER_WORD must be a multiple of OUT_CH_PAR * OUT_W_PAR.
 * - If OUT_W_PAR > 1, CH must be equal to OUT_CH_PAR, this is to preserve
 * the correct order of the data flowing.
 * - CH must be a multiple of OUT_CH_PAR.
 * - WIDTH must be a multiple of OUT_W_PAR.
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
 * The class supports parallel processing of output channels and width, as
 * specified by OUT_CH_PAR and OUT_W_PAR, respectively.
 *
 * @section Quantization
 * The Quantizer template parameter is used to quantize the extracted data
 * before writing to the output stream.
 */

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, size_t DATA_PER_WORD,
          size_t HEIGHT, size_t WIDTH, size_t CH, size_t OUT_W_PAR,
          size_t OUT_CH_PAR, bool ONLY_ONCE = false>
class NHWCToStream {
  static constexpr size_t ITER = HEIGHT * WIDTH * CH / (OUT_W_PAR * OUT_CH_PAR);

public:
  static_assert(OUT_W_PAR == 1 || CH == OUT_CH_PAR,
                "CH must be equal to OUT_CH_PAR when OUT_W_PAR > 1");
  static_assert(CH % OUT_CH_PAR == 0, "CH must be a multiple of OUT_CH_PAR");
  static_assert(WIDTH % OUT_W_PAR == 0,
                "WIDTH must be a multiple of OUT_W_PAR");

  NHWCToStream() = default;

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> &input_data_stream,
           hls::stream<TOutputWord> output_data_stream[OUT_W_PAR]) {
    TOutput circular_buffer[DATA_PER_WORD * 2];
    char head = 0; // Head index for circular buffer
    char tail = 0; // Tail index for circular buffer
    char size = 0; // Current size of the circular buffer
    bool static only_once_flag = false;
    if (!only_once_flag) {

      // Loop through the word packets of the output tensor.
    NHWC_TO_STREAM_MAINLOOP:
      for (size_t i_output_word = 0; i_output_word < ITER; i_output_word++) {
#pragma HLS pipeline II = 1
        NHWCToStream::pipeline_body(input_data_stream, output_data_stream,
                                    circular_buffer, head, size, tail);
      }
    }
    if (ONLY_ONCE)
      only_once_flag = true;
  }

  struct StepState {
    // Circular buffer to hold input data for processing.
    TOutput circular_buffer[DATA_PER_WORD * 2];

    // Indexes and size for the circular buffer.
    char head = 0, tail = 0, size = 0;

    // Loop iteration index for the output word.
    size_t i_output_word = 0;
    ActorStatus actor_status{1, 1};
    PipelineDelayBuffer<TOutputWord> delayed_output[OUT_W_PAR];
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      actor_status = ActorStatus(depth, ITER);
      for (size_t i = 0; i < OUT_W_PAR; ++i)
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
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

  ActorStatus step(hls::stream<TInputWord> &input_data_stream,
                   hls::stream<TOutputWord> output_data_stream[OUT_W_PAR]) {

    // Find the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() &&
           "step_init() must be called before step()");
    StepState &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    if (st.size < OUT_CH_PAR * OUT_W_PAR && input_data_stream.empty()) {
      firing_condition = false;
    }

    if (firing_condition) {
      hls::stream<TOutputWord> instant_output_stream[OUT_W_PAR];
      NHWCToStream::pipeline_body(input_data_stream, instant_output_stream,
                                  st.circular_buffer, st.head, st.size,
                                  st.tail);
      st.i_output_word++;
      if (st.i_output_word >= ITER) {
        st.i_output_word = 0;
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++) {
        if (!instant_output_stream[i_w_par].empty()) {
          st.delayed_output[i_w_par].push(instant_output_stream[i_w_par].read(),
                                          true);
        } else {
          st.delayed_output[i_w_par].push(TOutputWord(),
                                          false); // Placeholder, ignored
        }
      }
    } else {

      for (size_t i = 0; i < OUT_W_PAR; ++i) {
        st.delayed_output[i].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++) {
      if (st.delayed_output[i_w_par].pop(out)) {
        output_data_stream[i_w_par].write(out);
      }
    }

    // Return the current firing iteration index.
    return st.actor_status;
  }

private:
  static void
  pipeline_body(hls::stream<TInputWord> &input_data_stream,
                hls::stream<TOutputWord> output_data_stream[OUT_W_PAR],
                TOutput circular_buffer[DATA_PER_WORD * 2], char &head,
                char &size, char &tail) {
#pragma HLS inline
    Quantizer quantizer; // Quantizer instance for quantization.

    // Read a new input data word if there is not enough data in the circular
    // buffer to output the required parallel data.
    if (size < OUT_CH_PAR * OUT_W_PAR) {
      TInputWord input_data = input_data_stream.read();
      for (size_t i = 0; i < DATA_PER_WORD; i++) {
        circular_buffer[head + i] = input_data.data.range(
            TOutput::width * (i + 1) - 1, TOutput::width * i);
      }
      head = (head + DATA_PER_WORD) % (DATA_PER_WORD * 2);
      size += DATA_PER_WORD;
    }

    // Loop through the pixels processed in parallel.
    for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++) {
      TOutputWord s_output_struct; // Output structure to hold the results.
      for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR; i_och_par++) {
        s_output_struct[i_och_par] = quantizer(circular_buffer[tail]);
        tail = (tail + 1) % (DATA_PER_WORD * 2);
      }
      // Write the output structure to the output stream.
      output_data_stream[i_w_par].write(s_output_struct);
    }
    size -= OUT_CH_PAR * OUT_W_PAR;
  }
};