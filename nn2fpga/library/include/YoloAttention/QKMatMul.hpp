#pragma once
#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <unordered_map>

template <typename TQInputWord, typename TQInput, typename TKInputWord,
          typename TKInput, typename TOutputWord, typename TOutput,
          typename TAcc, typename Quantizer, size_t DIM_HEADS, size_t DIM_Q,
          size_t DIM_K, size_t DIM_SEQ, size_t REDUCE_PAR>
class QKMatMul {
  struct StepState {
    // Loop iteration indexes.
    size_t i_qrow = 0, i_heads = 0, i_kcol = 0, i_group_reduce = 0;

    TKInputWord k_matrix[DIM_HEADS][DIM_K][DIM_SEQ / REDUCE_PAR];
    TQInputWord q_row[DIM_SEQ / REDUCE_PAR];
    TAcc acc;
    PipelineDelayBuffer<TOutputWord> delayed_output;
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      delayed_output = PipelineDelayBuffer<TOutputWord>(depth);
      actor_status = ActorStatus(depth, DIM_HEADS * DIM_Q * DIM_K * DIM_SEQ /
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
  QKMatMul() = default;

  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  ActorStatus step(hls::stream<TQInputWord> i_data_q[DIM_HEADS],
                   hls::stream<TKInputWord> i_data_k[DIM_HEADS],
                   hls::stream<TOutputWord> o_data_qk[DIM_HEADS]) {
    // Retrieve the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;
    if (st.i_kcol == 0) {
      if (i_data_q[st.i_heads].empty()) {
        firing_condition = false;
      }
    }

    if (st.i_qrow == 0) {
      if (i_data_k[st.i_heads].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      if (st.i_group_reduce == 0) {
        st.acc = 0;
      }

      // If there is data in the input stream, process it.
      hls::stream<TOutputWord> instant_output_stream[DIM_HEADS];
      QKMatMul::pipeline_body(
          i_data_q[st.i_heads], i_data_k[st.i_heads], instant_output_stream[st.i_heads],
          st.i_qrow, st.i_kcol, st.i_group_reduce, st.acc,
          st.k_matrix[st.i_heads][st.i_kcol][st.i_group_reduce],
          st.q_row[st.i_group_reduce]);

      // Insert new firing status into the multiset.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      if (!instant_output_stream[st.i_heads].empty()) {
        st.delayed_output.push(
            instant_output_stream[st.i_heads].read(), true);
      } else {
        // If the output stream is empty, push a placeholder.
        st.delayed_output.push(TOutputWord(), false);
      }

      // Update the counters.
      st.i_group_reduce++;
      if (st.i_group_reduce >= DIM_SEQ / REDUCE_PAR) {
        st.i_group_reduce = 0;
        st.i_kcol++;
      }
      if (st.i_kcol >= DIM_K) {
        st.i_kcol = 0;
        st.i_heads++;
      }
      if (st.i_heads >= DIM_HEADS) {
        st.i_heads = 0;
        st.i_qrow++;
      }
      if (st.i_qrow >= DIM_Q) {
        st.i_qrow = 0;
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
      o_data_qk[st.i_heads].write(out);
    }

    // Return the actor status.
    return st.actor_status;
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TQInputWord> i_data_q[DIM_HEADS],
           hls::stream<TKInputWord> i_data_k[DIM_HEADS],
           hls::stream<TOutputWord> o_data_qk[DIM_HEADS]) {

    TKInputWord k_matrix[DIM_HEADS][DIM_K][DIM_SEQ / REDUCE_PAR];
#pragma HLS array_partition variable = k_matrix complete dim = 4
    TQInputWord q_row[DIM_SEQ / REDUCE_PAR];
#pragma HLS array_partition variable = q_row complete dim = 2
    TAcc acc;

    for (size_t i_qrow = 0; i_qrow < DIM_Q; i_qrow++) {
      for (size_t i_heads = 0; i_heads < DIM_HEADS; i_heads++) {
        for (size_t i_kcol = 0; i_kcol < DIM_K; i_kcol++) {
          for (size_t i_group_reduce = 0; i_group_reduce < DIM_SEQ / REDUCE_PAR;
               i_group_reduce++) {
#pragma HLS PIPELINE II = 1
            QKMatMul::pipeline_body(i_data_q[i_heads], i_data_k[i_heads],
                                    o_data_qk[i_heads], i_qrow, i_kcol,
                                    i_group_reduce, acc,
                                    k_matrix[i_heads][i_kcol][i_group_reduce],
                                    q_row[i_group_reduce]);
          }
        }
      }
    }
  }

private:
  static void pipeline_body(hls::stream<TQInputWord> &i_data_q,
                            hls::stream<TKInputWord> &i_data_k,
                            hls::stream<TOutputWord> &o_data_qk,
                            size_t i_qrow, size_t i_kcol, size_t i_group_reduce,
                            TAcc &acc,
                            TKInputWord &k_matrix,
                            TQInputWord &q_row) {
#pragma HLS inline
    Quantizer quantizer;

    if (i_qrow == 0) {
      k_matrix = i_data_k.read();
    }
    if (i_kcol == 0) {
      q_row = i_data_q.read();
    }
    if (i_group_reduce == 0) {
      acc = 0;
    }

    TKInputWord k_word = k_matrix;
    TQInputWord q_word = q_row;
    for (size_t i_reduce = 0; i_reduce < REDUCE_PAR; i_reduce++) {
      TQInput q_i = (TQInput)q_word[i_reduce];
      TKInput k_i = (TKInput)k_word[i_reduce];
      acc += q_i * k_i;
    }

    if (i_group_reduce == (DIM_SEQ / REDUCE_PAR) - 1) {
      TOutput out_value = quantizer(acc);
      TOutputWord out_word;
      out_word[0] = out_value;
      o_data_qk.write(out_word);
    }
  }
};