#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>

/**
 * @class StreamingTensorDuplicator
 * @brief Implements a tensor duplicator for HWC-formatted data.
 *
 * This class duplicates a tensor in a streaming fashion.
 * The input data is expected in HWC format. The duplication operation creates
 * multiple copies of the input tensor across the height and width dimensions
 * for each channel.
 *
 * @tparam TWord          Structure type for input/output data.
 * @tparam DIM0      Input height (number of rows).
 * @tparam DIM1       Input width (number of columns).
 * @tparam DIM2          Number of input channels.
 * @tparam DIM2_UNROLL         Number of output channels processed in parallel.
 * @tparam DIM1_UNROLL          Number of output width processed in parallel.
 *
 * @note
 * - DIM2 must be a multiple of DIM2_UNROLL.
 * - DIM0, DIM1, and DIM2_UNROLL must be greater than 0.
 *
 * @section Usage
 * - Use the run() method for functional verification and synthesis.
 * - Use the step() method for self-timed execution with actor status tracking,
 * which is needed for fifo depth estimation.
 *
 * @section Parallelism
 * The class supports parallel processing of output channels, as specified by
 * DIM2_UNROLL.
 */

template <typename TWord, size_t DIM0, size_t DIM1, size_t DIM2,
          size_t DIM1_UNROLL, size_t DIM2_UNROLL>
class StreamingTensorDuplicator {
public:
  static_assert(DIM2 % DIM2_UNROLL == 0,
                "DIM2 must be a multiple of DIM2_UNROLL");
  static_assert(DIM1 % DIM1_UNROLL == 0,
                "DIM1 must be a multiple of DIM1_UNROLL");
  static_assert(DIM2_UNROLL > 0, "DIM2_UNROLL must be greater than 0");
  static_assert(DIM1_UNROLL > 0, "DIM1_UNROLL must be greater than 0");
  static_assert(DIM0 > 0 && DIM1 > 0, "DIM0 and DIM1 must be greater than 0");

  StreamingTensorDuplicator() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_dim01 = 0, i_dim2 = 0;

    PipelineDelayBuffer<TWord> delayed_output[DIM1_UNROLL * 2];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < DIM1_UNROLL * 2; i++) {
        delayed_output[i] = PipelineDelayBuffer<TWord>(depth);
      }
      actor_status =
          ActorStatus(depth, DIM0 * DIM1 * DIM2 / (DIM2_UNROLL * DIM1_UNROLL));
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
  void run(hls::stream<TWord> i_data[DIM1_UNROLL],
           hls::stream<TWord> o_data0[DIM1_UNROLL],
           hls::stream<TWord> o_data1[DIM1_UNROLL]) {

    // Loop through the input height and width.
    for (size_t i_dim01 = 0; i_dim01 < DIM0 * DIM1 / DIM1_UNROLL; i_dim01++) {
    TENSORDUPLICATOR_RUN_LOOP:
      for (size_t i_dim2 = 0; i_dim2 < DIM2 / DIM2_UNROLL; i_dim2++) {
#pragma HLS pipeline II = 1
        StreamingTensorDuplicator::pipeline_body(i_data, o_data0, o_data1);
      }
    }
  }

  ActorStatus step(hls::stream<TWord> i_data[DIM1_UNROLL],
                   hls::stream<TWord> o_data0[DIM1_UNROLL],
                   hls::stream<TWord> o_data1[DIM1_UNROLL]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (i_data[i_dim1_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      // If there is data in the input stream, process it.
      hls::stream<TWord> instant_output_stream0[DIM1_UNROLL];
      hls::stream<TWord> instant_output_stream1[DIM1_UNROLL];
      StreamingTensorDuplicator::pipeline_body(i_data, instant_output_stream0,
                                                instant_output_stream1);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        st.delayed_output[i_dim1_par].push(
            instant_output_stream0[i_dim1_par].read(), true);
        st.delayed_output[i_dim1_par + DIM1_UNROLL].push(
            instant_output_stream1[i_dim1_par].read(), true);
      }

      // Update the counters.
      st.i_dim2++;
      if (st.i_dim2 >= DIM2 / DIM2_UNROLL) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        st.i_dim2 = 0;
        st.i_dim01++;
      }
      if (st.i_dim01 >= DIM0 * DIM1 / DIM1_UNROLL) {
        st.i_dim01 = 0; // Reset the height/width index if we have processed all
                        // iterations.
      }

    } else {
      // If there is no data in the input stream, push a delay slot.
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL * 2; i_dim1_par++) {
        st.delayed_output[i_dim1_par].push(TWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TWord out;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      if (st.delayed_output[i_dim1_par].pop(out)) {
        o_data0[i_dim1_par].write(out);
      }
      if (st.delayed_output[i_dim1_par + DIM1_UNROLL].pop(out)) {
        o_data1[i_dim1_par].write(out);
      }
    }

    // Return the actor status.
    return st.actor_status;
  }

private:
  static void pipeline_body(hls::stream<TWord> i_data[DIM1_UNROLL],
                            hls::stream<TWord> o_data0[DIM1_UNROLL],
                            hls::stream<TWord> o_data1[DIM1_UNROLL]) {
#pragma HLS inline
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      TWord in_word = i_data[i_dim1_par].read();
      o_data0[i_dim1_par].write(in_word);
      o_data1[i_dim1_par].write(in_word);
    }
  }
};
