#pragma once
#include "hls_stream.h"
#include "ap_int.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>

/**
 * @brief StreamingMemory implements a memory as a stream of data.
 *
 * This class is designated to follow the StreamingConv reading pattern,
 * i.e. blocks of IN_CH_PAR channels of OUT_CH_PAR filters at a time.
 * It supports initialization from a stream of packed input data.
 * Multiple StreamingMemory instances can be concatenated to load all the
 * memory with a single chain.
 *
 * @tparam TInput         Type of input data word (encoded stream).
 * @tparam TOutput        Type of output data element.
 * @tparam TOutputWord    Type of output data word (vector of TOutput).
 * @tparam DATA_PER_WORD  Number of data elements per input word.
 * @tparam DATA_TO_SHIFT  Number of input words to shift for downstream nodes.
 * @tparam TIMES          Number of times to repeat the streaming operation.
 * @tparam OUT_CH         Total number of output channels.
 * @tparam IN_CH          Total number of input channels.
 * @tparam FH             Filter height (spatial dimension).
 * @tparam FW             Filter width (spatial dimension).
 * @tparam OUT_CH_PAR     Output channel parallelism factor.
 * @tparam IN_CH_PAR      Input channel parallelism factor.
 *
 * @section Implementation Details
 * - Memory initialization is done only once, on the first call to run()
 * - It supports packed data that does not fit perfectly into input words.
 */

template <typename TInput, typename TOutput, typename TOutputWord,
          size_t DATA_PER_WORD, size_t DATA_TO_SHIFT, size_t TIMES,
          size_t OUT_CH, size_t IN_CH, size_t FH, size_t FW, size_t OUT_CH_PAR,
          size_t IN_CH_PAR>
class StreamingMemory {
private:
  static constexpr size_t CH_GROUPS = OUT_CH * IN_CH / (OUT_CH_PAR * IN_CH_PAR);

  size_t STEP_i_ch_groups;
  size_t STEP_i_hw;
  size_t STEP_pipeline_depth;
  ActorStatus STEP_actor_status;
  PipelineDelayBuffer<TOutputWord> STEP_delayed_output[FH * FW];

  static void
  pipeline_body(hls::stream<TOutputWord> o_data[FH * FW],
                TOutput mem[OUT_CH_PAR * IN_CH_PAR][FH * FW]) {
#pragma HLS inline
    for (size_t i_fhw = 0; i_fhw < FH * FW; i_fhw++) {
      TOutputWord out_word;
      for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR; i_och_par++) {
        for (size_t i_ich_par = 0; i_ich_par < IN_CH_PAR; i_ich_par++) {
          out_word[i_och_par * IN_CH_PAR + i_ich_par] =
              mem[i_och_par * IN_CH_PAR + i_ich_par][i_fhw];
        }
      }
      o_data[i_fhw].write(out_word);
    }
  }

  void
  initialize_memory(hls::stream<TInput> i_shift_data[1],
                    TOutput mem[CH_GROUPS][OUT_CH_PAR * IN_CH_PAR][FH * FW]) {
#pragma HLS inline
    auto i_fhw = 0;
    auto i_ch_groups = 0;
    auto i_par = 0;
    for (size_t i_word = 0; i_word < FH * FW * IN_CH * OUT_CH;
         i_word += DATA_PER_WORD) {
#pragma HLS pipeline off
      TInput in_word = i_shift_data[0].read();

      // Loop until all data inside the word are processed (excluding padding).
      for (size_t i_data = 0;
           i_data < DATA_PER_WORD && i_word + i_data < FH * FW * IN_CH * OUT_CH;
           i_data++) {
#pragma HLS pipeline off
        TOutput in_data = in_word.range(TOutput::width - 1, 0);
        in_word >>= TOutput::width;
        mem[i_ch_groups][i_par][i_fhw] = in_data;
        i_fhw++;
        if (i_fhw == FH * FW) {
          i_fhw = 0;
          i_par++;
        }
        if (i_par == IN_CH_PAR * OUT_CH_PAR) {
          i_par = 0;
          i_ch_groups++;
        }
      }
    }
  }

public:
  StreamingMemory() : StreamingMemory(1) {}
  StreamingMemory(size_t pipeline_depth)
      : STEP_i_ch_groups(0), STEP_i_hw(0), STEP_pipeline_depth(pipeline_depth),
        STEP_actor_status(pipeline_depth, CH_GROUPS * TIMES) {
    for (size_t i = 0; i < FH * FW; ++i) {
      STEP_delayed_output[i] = PipelineDelayBuffer<TOutputWord>(pipeline_depth);
    }
  }

  void run(hls::stream<TInput> i_shift_data[1],
           hls::stream<TOutputWord> o_data[FH * FW],
           hls::stream<TInput> o_shift_data[1]) {

    // Initialize memory from input stream on first run
    static TOutput mem[CH_GROUPS][OUT_CH_PAR * IN_CH_PAR][FH * FW];
    static bool initialized_flag = false;
    if (!initialized_flag) {
      initialized_flag = true;
      initialize_memory(i_shift_data, mem);

      // Shift data for following nodes.
      for (size_t i = 0; i < DATA_TO_SHIFT; i++) {
        TInput in_word = i_shift_data[0].read();
        o_shift_data[0].write(in_word);
      }
    }

    for (size_t i_hw = 0; i_hw < TIMES; i_hw++) {
      for (size_t i_ch_groups = 0; i_ch_groups < CH_GROUPS; i_ch_groups++) {
#pragma HLS pipeline II = 1
        StreamingMemory::pipeline_body(o_data, mem[i_ch_groups]);
      }
    }
  }

  ActorStatus step(hls::stream<TInput> i_shift_data[1],
                   hls::stream<TOutputWord> o_data[FH * FW],
                   hls::stream<TInput> o_shift_data[1]) {
    (void)o_shift_data;
    return step(i_shift_data, o_data);
  }

  void run(hls::stream<TInput> i_shift_data[1],
           hls::stream<TOutputWord> o_data[FH * FW]) {
    static TOutput mem[CH_GROUPS][OUT_CH_PAR * IN_CH_PAR][FH * FW];
    static bool initialized_flag = false;
    
    // Initialize memory from input stream on first run
    if (!initialized_flag) {
      initialized_flag = true;
      initialize_memory(i_shift_data, mem);
    }

    for (size_t i_hw = 0; i_hw < TIMES; i_hw++) {
      for (size_t i_ch_groups = 0; i_ch_groups < CH_GROUPS; i_ch_groups++) {
#pragma HLS pipeline II = 1
        StreamingMemory::pipeline_body(o_data, mem[i_ch_groups]);
      }
    }
  }

  ActorStatus step(hls::stream<TInput> i_shift_data[1],
                   hls::stream<TOutputWord> o_data[FH * FW]) {
    (void)i_shift_data;
    static TOutput mem[CH_GROUPS][OUT_CH_PAR * IN_CH_PAR][FH * FW];
    hls::stream<TOutputWord> instant_output_stream[FH * FW];
    StreamingMemory::pipeline_body(instant_output_stream,
                                   mem[STEP_i_ch_groups]);
    STEP_i_ch_groups++;
    if (STEP_i_ch_groups >= CH_GROUPS) {
      STEP_i_ch_groups = 0;
      STEP_i_hw++;
    }
    if (STEP_i_hw >= TIMES) {
      STEP_i_hw = 0;
    }
    STEP_actor_status.fire();
    STEP_actor_status.advance();
    for (size_t i_fhw = 0; i_fhw < FH * FW; ++i_fhw) {
      if (!instant_output_stream[i_fhw].empty()) {
        STEP_delayed_output[i_fhw].push(instant_output_stream[i_fhw].read(),
                                        true);
      } else {
        // If the output stream is empty, push a placeholder.
        STEP_delayed_output[i_fhw].push(TOutputWord(), false);
      }
    }
    for (size_t i_fhw = 0; i_fhw < FH * FW; i_fhw++) {
      TOutputWord out;
      if (STEP_delayed_output[i_fhw].pop(out)) {
        o_data[i_fhw].write(out);
      }
    }
    return STEP_actor_status;
  }
};
