#pragma once
#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <unordered_map>

template <typename TVInputWord, typename TVInput, typename TPInputWord,
          typename TPInput, typename TOutputWord, typename TOutput,
          typename TAcc, typename Quantizer, size_t DIM_HEADS, size_t DIM_V,
          size_t DIM_P, size_t DIM_SEQ, size_t REDUCE_PAR>
class VPMatMul {
  struct StepState {
    // Loop iteration indexes.
    size_t i_vrow = 0, i_heads = 0, i_pcol = 0, i_group_reduce = 0;

    TVInputWord v_matrix[DIM_HEADS][DIM_V][DIM_SEQ / REDUCE_PAR];
    TPInputWord p_row[DIM_SEQ / REDUCE_PAR];
    TAcc acc;
    PipelineDelayBuffer<TOutputWord> delayed_output;
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      delayed_output = PipelineDelayBuffer<TOutputWord>(depth);
      actor_status = ActorStatus(depth, DIM_HEADS * DIM_V * DIM_P * DIM_SEQ /
                                            (REDUCE_PAR));
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

public:
  VPMatMul() = default;

  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  ActorStatus step(hls::stream<TVInputWord> i_data_v[DIM_HEADS],
                   hls::stream<TPInputWord> i_data_p[DIM_HEADS],
                   hls::stream<TOutputWord> o_data_vp[DIM_HEADS]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    if (st.i_vrow == 0) {
      if (i_data_p[st.i_heads].empty()) {
        firing_condition = false;
      }
    }

    if (st.i_pcol == 0) {
      if (i_data_v[st.i_heads].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      if (st.i_group_reduce == 0) {
        st.acc = 0;
      }

      // If there is data in the input stream, process it.
      hls::stream<TOutputWord> instant_output_stream[DIM_HEADS];
      VPMatMul::pipeline_body(
          i_data_v[st.i_heads], i_data_p[st.i_heads], instant_output_stream[st.i_heads],
          st.i_pcol, st.i_vrow, st.i_group_reduce, st.acc,
          st.v_matrix[st.i_heads][st.i_vrow][st.i_group_reduce],
          st.p_row[st.i_group_reduce]);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      if (!instant_output_stream[st.i_heads].empty()) {
        st.delayed_output.push(instant_output_stream[st.i_heads].read(), true);
      } else {
        // If the output stream is empty, push a placeholder.
        st.delayed_output.push(TOutputWord(), false);
      }

      // Update the counters.
      st.i_group_reduce++;
      if (st.i_group_reduce >= DIM_SEQ / REDUCE_PAR) {
        st.i_group_reduce = 0;
        st.i_vrow++;
      }
      if (st.i_vrow >= DIM_V) {
        st.i_vrow = 0;
        st.i_heads++;
      }
      if (st.i_heads >= DIM_HEADS) {
        st.i_heads = 0;
        st.i_pcol++;
      }
      if (st.i_pcol >= DIM_P) {
        st.i_pcol = 0;
      }
    } else {
      // If there is no data in the input stream, push a delay slot.
      st.delayed_output.push(TOutputWord(), false);
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    if (st.delayed_output.pop(out)) {
      o_data_vp[st.i_heads].write(out);
    }

    // Return the actor status.
    return st.actor_status;
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TVInputWord> i_data_v[DIM_HEADS],
           hls::stream<TPInputWord> i_data_p[DIM_HEADS],
           hls::stream<TOutputWord> o_data_vp[DIM_HEADS]) {

    TVInputWord v_matrix[DIM_HEADS][DIM_V][DIM_SEQ / REDUCE_PAR];
#pragma HLS array_partition variable = v_matrix complete dim = 4
    TPInputWord p_row[DIM_SEQ / REDUCE_PAR];
#pragma HLS array_partition variable = p_row complete dim = 2
    TAcc acc;

    for (size_t i_pcol = 0; i_pcol < DIM_P; i_pcol++) {
      for (size_t i_heads = 0; i_heads < DIM_HEADS; i_heads++) {
        for (size_t i_vrow = 0; i_vrow < DIM_V; i_vrow++) {
          for (size_t i_group_reduce = 0; i_group_reduce < DIM_SEQ / REDUCE_PAR;
               i_group_reduce++) {
#pragma HLS loop_flatten
#pragma HLS PIPELINE II = 1
            VPMatMul::pipeline_body(i_data_v[i_heads], i_data_p[i_heads],
                                    o_data_vp[i_heads], i_pcol, i_vrow,
                                    i_group_reduce, acc,
                                    v_matrix[i_heads][i_vrow][i_group_reduce],
                                    p_row[i_group_reduce]);
          }
        }
      }
    }
  }

private:
  static void pipeline_body(hls::stream<TVInputWord> &i_data_v,
                            hls::stream<TPInputWord> &i_data_p,
                            hls::stream<TOutputWord> &o_data_vp, size_t i_pcol,
                            size_t i_vrow, size_t i_group_reduce, TAcc &acc,
                            TVInputWord &v_matrix, TPInputWord &p_row) {
#pragma HLS inline
    Quantizer quantizer;

    if (i_pcol == 0) {
      v_matrix = i_data_v.read();
    }
    if (i_vrow == 0) {
      p_row = i_data_p.read();
    }
    if (i_group_reduce == 0) {
      acc = 0;
    }

    TVInputWord v_word = v_matrix;
    TPInputWord p_word = p_row;
    for (size_t i_reduce = 0; i_reduce < REDUCE_PAR; i_reduce++) {
      TVInput v_i = (TVInput)v_word[i_reduce];
      TPInput p_i = (TPInput)p_word[i_reduce];
      acc += v_i * p_i;
    }

    if (i_group_reduce == (DIM_SEQ / REDUCE_PAR) - 1) {
      TOutput out_value = quantizer(acc);
      TOutputWord out_word;
      out_word[0] = out_value;
      o_data_vp.write(out_word);
    }
  }
};