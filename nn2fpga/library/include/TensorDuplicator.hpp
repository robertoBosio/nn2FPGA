#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>

/**
 * @class TensorDuplicator
 * @brief Implements a tensor duplicator for HWC-formatted data.
 *
 * This class duplicates a tensor in a streaming fashion.
 * The input data is expected in HWC format. The duplication operation creates
 * multiple copies of the input tensor across the height and width dimensions
 * for each channel.
 *
 * @tparam TWord          Structure type for input/output data.
 * @tparam IN_HEIGHT      Input height (number of rows).
 * @tparam IN_WIDTH       Input width (number of columns).
 * @tparam IN_CH          Number of input channels.
 * @tparam CH_PAR         Number of output channels processed in parallel.
 * @tparam W_PAR          Number of output width processed in parallel.
 *
 * @note
 * - IN_CH must be a multiple of CH_PAR.
 * - IN_HEIGHT, IN_WIDTH, and CH_PAR must be greater than 0.
 *
 * @section Usage
 * - Use the run() method for functional verification and synthesis.
 * - Use the step() method for self-timed execution with actor status tracking,
 * which is needed for fifo depth estimation.
 *
 * @section Parallelism
 * The class supports parallel processing of output channels, as specified by
 * CH_PAR.
 */

template <typename TWord, size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH,
          size_t CH_PAR, size_t W_PAR>
class TensorDuplicator {
public:
  static_assert(IN_CH % CH_PAR == 0, "IN_CH must be a multiple of CH_PAR");
  static_assert(CH_PAR > 0, "CH_PAR must be greater than 0");
  static_assert(IN_HEIGHT > 0 && IN_WIDTH > 0,
                "IN_HEIGHT and IN_WIDTH must be greater than 0");

  TensorDuplicator() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_hw = 0, i_ch = 0;

    PipelineDelayBuffer<TWord> delayed_output[W_PAR * 2];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < W_PAR * 2; i++) {
        delayed_output[i] = PipelineDelayBuffer<TWord>(depth);
      }
      actor_status =
          ActorStatus(depth, IN_HEIGHT * IN_WIDTH * IN_CH / (CH_PAR * W_PAR));
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
  void run(hls::stream<TWord> i_data[W_PAR], hls::stream<TWord> o_data0[W_PAR],
           hls::stream<TWord> o_data1[W_PAR]) {

    // Loop through the input height and width.
    for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH / W_PAR; i_hw++) {
    TENSORDUPLICATOR_RUN_LOOP:
      for (size_t i_ch = 0; i_ch < IN_CH / CH_PAR; i_ch++) {
#pragma HLS pipeline II = 1
        TensorDuplicator::pipeline_body(i_data, o_data0, o_data1);
      }
    }
  }

  ActorStatus step(hls::stream<TWord> i_data[W_PAR],
                   hls::stream<TWord> o_data0[W_PAR],
                   hls::stream<TWord> o_data1[W_PAR]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t w_par = 0; w_par < W_PAR; w_par++) {
      if (i_data[w_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      // If there is data in the input stream, process it.
      hls::stream<TWord> instant_output_stream0[W_PAR];
      hls::stream<TWord> instant_output_stream1[W_PAR];
      TensorDuplicator::pipeline_body(i_data, instant_output_stream0,
                                      instant_output_stream1);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t w_par = 0; w_par < W_PAR; w_par++) {
        st.delayed_output[w_par].push(instant_output_stream0[w_par].read(),
                                      true);
        st.delayed_output[w_par + W_PAR].push(
            instant_output_stream1[w_par].read(), true);
      }

      // Update the counters.
      st.i_ch++;
      if (st.i_ch >= IN_CH / CH_PAR) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        st.i_ch = 0;
        st.i_hw++;
      }
      if (st.i_hw >= IN_HEIGHT * IN_WIDTH / W_PAR) {
        st.i_hw = 0; // Reset the height/width index if we have processed all
                     // iterations.
      }

    } else {
      // If there is no data in the input stream, push a delay slot.
      for (size_t w_par = 0; w_par < W_PAR * 2; w_par++) {
        st.delayed_output[w_par].push(TWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TWord out;
    for (size_t w_par = 0; w_par < W_PAR; w_par++) {
      if (st.delayed_output[w_par].pop(out)) {
        o_data0[w_par].write(out);
      }
      if (st.delayed_output[w_par + W_PAR].pop(out)) {
        o_data1[w_par].write(out);
      }
    }

    // Return the actor status.
    return st.actor_status;
  }

private:
  static void pipeline_body(hls::stream<TWord> i_data[W_PAR],
                            hls::stream<TWord> o_data0[W_PAR],
                            hls::stream<TWord> o_data1[W_PAR]) {
#pragma HLS inline
    for (size_t w_par = 0; w_par < W_PAR; w_par++) {
      TWord in_word = i_data[w_par].read();
      o_data0[w_par].write(in_word);
      o_data1[w_par].write(in_word);
    }
  }
};