#pragma once
#include "DequantQuant.hpp"
#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <unordered_map>

constexpr int leakyrelu_abs(int x) { return x < 0 ? -x : x; }

constexpr int leakyrelu_unsigned_bits(int x) {
  return x <= 1 ? 1 : 1 + leakyrelu_unsigned_bits(x >> 1);
}

template <int Shift, int AlphaNum, int AlphaDenShift, typename TAcc,
          typename TOut>
struct LeakyReLUQuantPo2 {
  TOut operator()(TAcc acc) const {
#pragma HLS inline
    if (acc < 0) {
      using TWide =
          ap_int<TAcc::width + leakyrelu_unsigned_bits(leakyrelu_abs(AlphaNum)) +
                 1>;
      TWide scaled = TWide(acc) * AlphaNum;
      DequantQuantPo2<Shift + AlphaDenShift, TWide, TOut> quantizer;
      return quantizer(scaled);
    } else {
      DequantQuantPo2<Shift, TAcc, TOut> quantizer;
      return quantizer(acc);
    }
  }
};

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename OutputTransform,
          size_t DIM0, size_t DIM1, size_t DIM2, size_t DIM1_UNROLL,
          size_t DIM2_UNROLL>
class StreamingLeakyReLU {
public:
  static_assert(DIM2 % DIM2_UNROLL == 0,
                "DIM2 must be a multiple of DIM2_UNROLL");
  static_assert(DIM2_UNROLL > 0, "DIM2_UNROLL must be greater than 0");
  static_assert(DIM1 % DIM1_UNROLL == 0,
                "DIM1 must be a multiple of DIM1_UNROLL");
  static_assert(DIM1_UNROLL > 0, "DIM1_UNROLL must be greater than 0");
  static_assert(DIM0 > 0 && DIM1 > 0, "DIM0 and DIM1 must be greater than 0");
  StreamingLeakyReLU() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_dim01 = 0, i_dim2 = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output[DIM1_UNROLL];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < DIM1_UNROLL; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
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
  void run(hls::stream<TInputWord> i_data[DIM1_UNROLL],
           hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
    // Loop through the input height and width.
    for (size_t i_hw = 0; i_hw < DIM0 * DIM1 / DIM1_UNROLL; i_hw++) {
    STREAMINGLEAKYRELU_RUN_LOOP:
      for (size_t i_ch = 0; i_ch < DIM2 / DIM2_UNROLL; i_ch++) {
#pragma HLS pipeline II = 1
        StreamingLeakyReLU::pipeline_body(i_data, o_data);
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[DIM1_UNROLL],
                   hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;

    // Check non empty input streams.
    for (size_t i_in_stream = 0; i_in_stream < DIM1_UNROLL; i_in_stream++) {
      if (i_data[i_in_stream].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      hls::stream<TOutputWord> instant_output_stream[DIM1_UNROLL];
      StreamingLeakyReLU::pipeline_body(i_data, instant_output_stream);

      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        st.delayed_output[i_dim1_par].push(
            instant_output_stream[i_dim1_par].read(), true);
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

    // Return the current actor status.
    return st.actor_status;
  }

private:
  static void pipeline_body(hls::stream<TInputWord> i_data[DIM1_UNROLL],
                            hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
#pragma HLS inline
    OutputTransform output_transform;
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      TInputWord in_word = i_data[i_dim1_par].read();
      TOutputWord out_word;
      for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL; i_dim2_par++) {
        out_word[i_dim2_par] = output_transform(in_word[i_dim2_par]);
      }
      o_data[i_dim1_par].write(out_word);
    }
  }
};
