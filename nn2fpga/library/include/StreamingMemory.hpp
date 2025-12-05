#pragma once
#include "ap_int.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
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
 * @tparam TInputWord     Type of input data word (encoded stream).
 * @tparam TOutput        Type of output data element.
 * @tparam TOutputWord    Type of output data word (vector of TOutput).
 * @tparam DATA_PER_WORD  Number of data elements per input word.
 * @tparam DATA_TO_SHIFT  Number of input words to shift for downstream nodes.
 * @tparam TIMES          Number of times to repeat the streaming operation.
 * @tparam WORDS          Number of words stored in the memory.
 * @tparam ARRAY_PAR      Number of streams written in parallel.
 * @tparam WORD_PAR       Number of words packed in a single stream packet.
 *
 * @section Implementation Details
 * - Memory initialization is done only once, on the first call to run()
 * - It supports packed data that does not fit perfectly into input words.
 */

template <typename TInputWord, typename TOutput, typename TOutputWord,
          size_t DATA_PER_WORD, size_t DATA_TO_SHIFT, size_t TIMES,
          size_t WORDS, size_t ARRAY_PAR, size_t WORD_PAR>
class StreamingMemory {
private:
  static constexpr size_t CH_GROUPS = WORDS / (ARRAY_PAR * WORD_PAR);

  static void pipeline_body(hls::stream<TOutputWord> o_data[ARRAY_PAR],
                            TOutput mem[WORD_PAR][ARRAY_PAR]) {
#pragma HLS inline
    for (size_t i_fhw = 0; i_fhw < ARRAY_PAR; i_fhw++) {
      TOutputWord out_word;
      for (size_t i_word_par = 0; i_word_par < WORD_PAR; i_word_par++) {
        out_word[i_word_par] = mem[i_word_par][i_fhw];
      }
      o_data[i_fhw].write(out_word);
    }
  }

  void initialize_memory(hls::stream<TInputWord> i_shift_data[1],
                         TOutput mem[CH_GROUPS][WORD_PAR][ARRAY_PAR]) {
#pragma HLS inline
    auto i_array_par = 0;
    auto i_ch_groups = 0;
    auto i_word_par = 0;
    TOutput word[WORD_PAR][ARRAY_PAR];
  MEMORY_INIT_LOOP:
    for (size_t i_word = 0; i_word < WORDS; i_word += DATA_PER_WORD) {
      TInputWord in_word = i_shift_data[0].read();

      // Loop until all data inside the word are processed (excluding padding).
    MEMORY_INIT_DATA_LOOP:
      for (size_t i_data = 0; i_data < DATA_PER_WORD && i_word + i_data < WORDS;
           i_data++) {
        TOutput in_data = in_word[0].range(TOutput::width - 1, 0);
        in_word[0] >>= TOutput::width;
        word[i_word_par][i_array_par] = in_data;
        i_array_par++;
        if (i_array_par == ARRAY_PAR) {
          i_array_par = 0;
          i_word_par++;
        }
        if (i_word_par == WORD_PAR) {
          for (size_t i = 0; i < WORD_PAR; i++) {
            #pragma HLS unroll
            for (size_t j = 0; j < ARRAY_PAR; j++) {
              mem[i_ch_groups][i][j] = word[i][j];
            }
          }
          i_word_par = 0;
          i_ch_groups++;
        }
      }
    }
  }

public:
  StreamingMemory() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_ch_groups = 0, i_hw = 0;
    PipelineDelayBuffer<TOutputWord> delayed_output[ARRAY_PAR];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < ARRAY_PAR; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status = ActorStatus(depth, CH_GROUPS * TIMES);
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
  void run(hls::stream<TInputWord> i_shift_data[1],
           hls::stream<TOutputWord> o_data[ARRAY_PAR],
           hls::stream<TInputWord> o_shift_data[1]) {

    // Initialize memory from input stream on first run
    static TOutput mem[CH_GROUPS][WORD_PAR][ARRAY_PAR];
#pragma HLS array_reshape variable = mem dim = 3 complete
#pragma HLS array_reshape variable = mem dim = 2 complete
    static bool initialized_flag = false;
    if (!initialized_flag) {
      initialize_memory(i_shift_data, mem);

      // Shift data for following nodes.
    SHIFT_LOOP:
      for (size_t i = 0; i < DATA_TO_SHIFT; i++) {
#pragma HLS pipeline off
        TInputWord in_word = i_shift_data[0].read();
        o_shift_data[0].write(in_word);
      }
    }
    initialized_flag = true;

    for (size_t i_hw = 0; i_hw < TIMES; i_hw++) {
    STREAMINGMEMORY_RUN_LOOP:
      for (size_t i_ch_groups = 0; i_ch_groups < CH_GROUPS; i_ch_groups++) {
#pragma HLS pipeline II = 1
        StreamingMemory::pipeline_body(o_data, mem[i_ch_groups]);
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_shift_data[1],
                   hls::stream<TOutputWord> o_data[ARRAY_PAR],
                   hls::stream<TInputWord> o_shift_data[1]) {
    (void)o_shift_data;
    (void)i_shift_data;
    return step(o_data);
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_shift_data[1],
           hls::stream<TOutputWord> o_data[ARRAY_PAR]) {
    static TOutput mem[CH_GROUPS][WORD_PAR][ARRAY_PAR];
#pragma HLS array_reshape variable = mem dim = 3 complete
#pragma HLS array_reshape variable = mem dim = 2 complete
    static bool initialized_flag = false;

    // Initialize memory from input stream on first run
    if (!initialized_flag) {
      initialize_memory(i_shift_data, mem);
    }
    initialized_flag = true;

    for (size_t i_hw = 0; i_hw < TIMES; i_hw++) {
    STREAMINGMEMORY_RUN_LOOP:
      for (size_t i_ch_groups = 0; i_ch_groups < CH_GROUPS; i_ch_groups++) {
#pragma HLS pipeline II = 1
        StreamingMemory::pipeline_body(o_data, mem[i_ch_groups]);
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_shift_data[1],
                   hls::stream<TOutputWord> o_data[ARRAY_PAR]) {
    (void)i_shift_data;
    return step(o_data);
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TOutputWord> o_data[ARRAY_PAR]) {
    static TOutput mem[CH_GROUPS][WORD_PAR][ARRAY_PAR];
#pragma HLS array_reshape variable = mem dim = 3 complete
#pragma HLS array_reshape variable = mem dim = 2 complete

    for (size_t i_hw = 0; i_hw < TIMES; i_hw++) {
    STREAMINGMEMORY_RUN_LOOP:
      for (size_t i_ch_groups = 0; i_ch_groups < CH_GROUPS; i_ch_groups++) {
#pragma HLS pipeline II = 1
        StreamingMemory::pipeline_body(o_data, mem[i_ch_groups]);
      }
    }
  }

  ActorStatus step(hls::stream<TOutputWord> o_data[ARRAY_PAR]) {
    static TOutput mem[CH_GROUPS][WORD_PAR][ARRAY_PAR];
    // Find the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    hls::stream<TOutputWord> instant_output_stream[ARRAY_PAR];
    StreamingMemory::pipeline_body(instant_output_stream, mem[st.i_ch_groups]);
    st.i_ch_groups++;
    if (st.i_ch_groups >= CH_GROUPS) {
      st.i_ch_groups = 0;
      st.i_hw++;
    }
    if (st.i_hw >= TIMES) {
      st.i_hw = 0;
    }
    st.actor_status.fire();
    st.actor_status.advance();
    for (size_t i_fhw = 0; i_fhw < ARRAY_PAR; ++i_fhw) {
      if (!instant_output_stream[i_fhw].empty()) {
        st.delayed_output[i_fhw].push(instant_output_stream[i_fhw].read(),
                                      true);
      } else {
        // If the output stream is empty, push a placeholder.
        st.delayed_output[i_fhw].push(TOutputWord(), false);
      }
    }
    for (size_t i_fhw = 0; i_fhw < ARRAY_PAR; i_fhw++) {
      TOutputWord out;
      if (st.delayed_output[i_fhw].pop(out)) {
        o_data[i_fhw].write(out);
      }
    }
    return st.actor_status;
  }
};
