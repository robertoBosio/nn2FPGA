#pragma once
#include "hls_stream.h"
#include "ap_int.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>
#include <cassert>

/**
 * @brief StreamingDepthwiseConv implements a quantized convolution with only streaming
 * in input and output. Works only with NHWC data layout.
 *
 * @tparam TInputWord      Data type for input word (packed input channels).
 * @tparam TInput          Data type for individual input elements.
 * @tparam TWeightWord     Data type for weight word (packed weights).
 * @tparam TWeight         Data type for individual weight elements.
 * @tparam TBiasWord       Data type for bias word (packed biases).
 * @tparam TBias          Data type for individual bias elements.
 * @tparam TOutputWord     Data type for output word (packed output channels).
 * @tparam TOutput         Data type for individual output elements.
 * @tparam TSum            Data type for accumulator considered the bias.
 * @tparam TPartialSum     Data type for partial sum accumulator.
 * @tparam Activation      Activation functor type for output activation.
 * @tparam Quantizer       Quantizer functor type for output quantization.
 * @tparam OUT_CH          Number of output channels.
 * @tparam IN_CH           Number of input channels.
 * @tparam OUT_HEIGHT      Output feature map height.
 * @tparam OUT_WIDTH       Output feature map width.
 * @tparam FH              Filter height.
 * @tparam FW              Filter width.
 * @tparam STRIDE_H        Stride along height.
 * @tparam STRIDE_W        Stride along width.
 * @tparam CH_PAR          Parallelism factor for channels.
 * @tparam W_PAR           Parallelism factor for output width (pixels).
 *
 * @note
 * - The class provides two main interfaces:
 *   - run(): Processes the entire convolution in a blocking fashion.
 *   - step(): Processes one pipeline step, suitable for CSDFG (Cyclo-Static
 * Data Flow Graph) scheduling.
 *
 * @section Implementation Details
 * - Weights are packed into words of IN_CH_PAR channels of OUT_CH_PAR filters.
 * Since we need to read FH*FW weights at each cycle, the weight input stream
 * is an array of FH*FW streams. Each stream provides IN_CH_PAR*OUT_CH_PAR
 * weights. Weights are read at each step. The only filter reuse is due to the
 * W_PAR factor, which allows to reuse the same weights for W_PAR adjacent
 * output pixels.
 * - Input data are packed into words of IN_CH_PAR channels. The window is
 * expanded to account for the stride and the width parallelism factor W_PAR,
 * such that no data is duplicated. The input stream is an array of
 * FH*(FW+(W_PAR-1)*STRIDE_W) streams, each providing IN_CH_PAR input channels.
 * The input window is completely reused.
 * - Biases are packed into words of OUT_CH_PAR channels. The bias stream
 * provides OUT_CH_PAR biases.
 * - Accumulators and input buffers are partitioned for parallel access.
 *
 */

template <typename TInputWord, typename TInput, typename TWeightWord,
          typename TWeight, typename TBiasWord, typename TBias,
          typename TOutputWord, typename TOutput, typename TSum,
          typename TPartialSum, typename Activation, typename Quantizer,
          size_t OUT_CH, size_t IN_CH, size_t OUT_HEIGHT, size_t OUT_WIDTH,
          size_t FH, size_t FW, size_t STRIDE_H, size_t STRIDE_W, size_t CH_PAR,
          size_t W_PAR>
class StreamingDepthwiseConv {
  static constexpr size_t FW_EXPAND = FW + (W_PAR - 1) * STRIDE_W;
  static constexpr size_t CH_GROUPS = OUT_CH / CH_PAR;

public:
  static_assert(OUT_HEIGHT > 0 && OUT_WIDTH > 0,
                "OUT_HEIGHT and OUT_WIDTH must be greater than 0");
  static_assert(W_PAR > 0, "W_PAR must be greater than 0");
  static_assert(CH_PAR > 0, "CH_PAR must be greater than 0");
  static_assert(FH > 0 && FW > 0, "FH and FW must be greater than 0");
  static_assert(STRIDE_H > 0 && STRIDE_W > 0, "STRIDE must be greater than 0");
  static_assert(CH_PAR > 0, "CH_PAR must be greater than 0");
  static_assert(W_PAR > 0, "W_PAR must be greater than 0");
  static_assert(OUT_CH % CH_PAR == 0,
                "OUT_CH must be a multiple of CH_PAR");
  static_assert(IN_CH % CH_PAR == 0,
                "IN_CH must be a multiple of CH_PAR");
  static_assert(OUT_WIDTH % W_PAR == 0,
                "OUT_WIDTH must be a multiple of W_PAR");
  static_assert(IN_CH == OUT_CH, "IN_CH must be equal to OUT_CH");

  StreamingDepthwiseConv() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_hw = 0, i_ch = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output[W_PAR];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < W_PAR; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status = ActorStatus(depth, OUT_HEIGHT * (OUT_WIDTH / W_PAR) *
                                            (OUT_CH / CH_PAR));
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
  void run(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
           hls::stream<TWeightWord> i_weights[FH * FW],
           hls::stream<TBiasWord> i_biases[1],
           hls::stream<TOutputWord> o_data[W_PAR]) {

    for (size_t i_hw = 0; i_hw < OUT_HEIGHT * OUT_WIDTH / W_PAR; i_hw++) {
    STREAMINGDEPTHWISECONV_RUN_LOOP:
      for (size_t i_ch = 0; i_ch < IN_CH; i_ch += CH_PAR) {
#pragma HLS pipeline II = 1
        StreamingDepthwiseConv::pipeline_body(
            i_data, i_weights, i_biases, o_data);
      }
    }
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
           TWeight i_weights[CH_GROUPS][CH_PAR][FH * FW],
           TBias i_biases[OUT_CH / CH_PAR][CH_PAR][1],
           hls::stream<TOutputWord> o_data[W_PAR]) {

    for (size_t i_hw = 0; i_hw < OUT_HEIGHT * OUT_WIDTH / W_PAR; i_hw++) {
      for (size_t i_ch = 0, weight_group = 0; i_ch < IN_CH;
           i_ch += CH_PAR, weight_group++) {
#pragma HLS pipeline II = 1
        StreamingDepthwiseConv::pipeline_body(i_data, i_weights[weight_group],
                                              i_biases[weight_group], o_data);
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
                   hls::stream<TWeightWord> i_weights[FH * FW],
                   hls::stream<TBiasWord> i_biases[1],
                   hls::stream<TOutputWord> o_data[W_PAR]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;

    // Check non empty input streams. Input data are read only at the
    // beginning of the computation of the output channels.
    for (size_t i_in_stream = 0; i_in_stream < FH * FW_EXPAND; i_in_stream++) {
      if (i_data[i_in_stream].empty()) {
        firing_condition = false;
      }
    }

    // Check non empty weight streams. Weights are read at each step.
    for (size_t i_weight_stream = 0; i_weight_stream < FH * FW;
         i_weight_stream++) {
      if (i_weights[i_weight_stream].empty()) {
        firing_condition = false;
      }
    }

    // Check non empty bias stream. Biases are read only at the end of the
    // computation of the output.
    if (i_biases[0].empty()) {
      firing_condition = false;
    }

    if (firing_condition) {

      hls::stream<TOutputWord> instant_output_stream[W_PAR];
      StreamingDepthwiseConv::pipeline_body(
          i_data, i_weights, i_biases, instant_output_stream);

      st.i_ch += CH_PAR;
      if (st.i_ch >= OUT_CH) {
        // If we have processed all output channels, reset the index and
        // increment the input channels index.
        st.i_ch = 0;
        st.i_hw++;
      }
      if (st.i_hw >= OUT_HEIGHT * OUT_WIDTH / W_PAR) {
        st.i_hw = 0;
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_w_par = 0; i_w_par < W_PAR; ++i_w_par) {
        if (!instant_output_stream[i_w_par].empty()) {
          st.delayed_output[i_w_par].push(instant_output_stream[i_w_par].read(),
                                          true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output[i_w_par].push(TOutputWord(), false);
        }
      }
    } else {
      // If no data is available, push empty outputs.
      for (size_t i_w_par = 0; i_w_par < W_PAR; ++i_w_par) {
        st.delayed_output[i_w_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      if (st.delayed_output[i_w_par].pop(out)) {
        o_data[i_w_par].write(out);
      }
    }

    // Return the current actor status.
    return st.actor_status;
  }

  ActorStatus step(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
                   TWeight i_weights[CH_GROUPS][CH_PAR][FH * FW],
                   TBias i_biases[OUT_CH / CH_PAR][CH_PAR][1],
                   hls::stream<TOutputWord> o_data[W_PAR]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;

    // Check non empty input streams. Input data are read only at the
    // beginning of the computation of the output channels.
    for (size_t i_in_stream = 0; i_in_stream < FH * FW_EXPAND; i_in_stream++) {
      if (i_data[i_in_stream].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      hls::stream<TOutputWord> instant_output_stream[W_PAR];
      StreamingDepthwiseConv::pipeline_body(
          i_data, i_weights[st.i_ch / CH_PAR], i_biases[st.i_ch / CH_PAR], instant_output_stream);

      st.i_ch += CH_PAR;
      if (st.i_ch >= OUT_CH) {
        // If we have processed all output channels, reset the index and
        // increment the input channels index.
        st.i_ch = 0;
        st.i_hw++;
      }
      if (st.i_hw >= OUT_HEIGHT * OUT_WIDTH / W_PAR) {
        st.i_hw = 0;
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_w_par = 0; i_w_par < W_PAR; ++i_w_par) {
        if (!instant_output_stream[i_w_par].empty()) {
          st.delayed_output[i_w_par].push(instant_output_stream[i_w_par].read(),
                                          true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output[i_w_par].push(TOutputWord(), false);
        }
      }
    } else {
      // If no data is available, push empty outputs.
      for (size_t i_w_par = 0; i_w_par < W_PAR; ++i_w_par) {
        st.delayed_output[i_w_par].push(TOutputWord(), false);
      }
    }

    // Advance the state of the actor firings.
    st.actor_status.advance();

    // Write the output data to the output stream.
    TOutputWord out;
    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      if (st.delayed_output[i_w_par].pop(out)) {
        o_data[i_w_par].write(out);
      }
    }

    // Return the current actor status.
    return st.actor_status;
  }

private:
  static void pipeline_body(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
                            hls::stream<TWeightWord> i_weights[FH * FW],
                            hls::stream<TBiasWord> i_biases[1],
                            hls::stream<TOutputWord> o_data[W_PAR]) {
#pragma HLS inline

    Quantizer quantizer;
    Activation activation;
    // Output structure to hold the results.
    TOutputWord output_data;
    // Weight structure to hold the weights.
    TWeightWord weight_data[FH][FW];
    // Bias structure to hold the biases.
    TBiasWord bias_data;
    // Input structure to hold the input data.
    TInputWord input_data[FH][FW_EXPAND];
    // Accumulator buffer.
    TPartialSum acc_buff_par[CH_PAR * W_PAR];

    // Read the input data for the current expanded window.
    for (size_t fh = 0; fh < FH; fh++) {
      for (size_t fw = 0; fw < FW_EXPAND; fw++) {
        input_data[fh][fw] = i_data[fh * FW_EXPAND + fw].read();
      }
    }

    // Read the weight data for the current filter.
    for (size_t fh = 0; fh < FH; fh++) {
      for (size_t fw = 0; fw < FW; fw++) {
        weight_data[fh][fw] = i_weights[fh * FW + fw].read();
      }
    }

    // Read the bias data.
    bias_data = i_biases[0].read();

    // Initialize the accumulator buffer for the current block of output
    // channels and pixels.
    for (size_t i = 0; i < CH_PAR * W_PAR; i++) {
      acc_buff_par[i] = 0;
    }

    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      for (size_t i_ch_par = 0; i_ch_par < CH_PAR; i_ch_par++) {

        // Compute the index of the accumulator.
        size_t acc_index = i_w_par * CH_PAR + i_ch_par;

        for (size_t i_fh = 0; i_fh < FH; i_fh++) {
          for (size_t i_fw = 0; i_fw < FW; i_fw++) {

            // Compute the filter width index inside the expanded input window.
            size_t i_fw_expanded = i_fw + i_w_par * STRIDE_W;

            acc_buff_par[acc_index] +=
                input_data[i_fh][i_fw_expanded][i_ch_par] *
                weight_data[i_fh][i_fw][i_ch_par];
          }
        }

        // Finalize the output.
        TSum wide_acc = acc_buff_par[acc_index] + bias_data[i_ch_par];
        wide_acc = activation(wide_acc);
        TOutput output_value = quantizer(wide_acc);
        output_data[i_ch_par] = output_value;

        // Write the output data only after the computation of all
        // output channels for the current pixels.
        if (i_ch_par == CH_PAR - 1) {
          o_data[i_w_par].write(output_data);
        }
      }
    }
  }

  static void pipeline_body(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
                            TWeight i_weights[CH_PAR][FH * FW],
                            TBias i_biases[CH_PAR][1],
                            hls::stream<TOutputWord> o_data[W_PAR]) {
#pragma HLS inline

    Quantizer quantizer;
    Activation activation;
    // Output structure to hold the results.
    TOutputWord output_data;
    // Weight structure to hold the weights.
    TWeightWord weight_data[FH][FW];
    // Bias structure to hold the biases.
    TBiasWord bias_data;
    // Input structure to hold the input data.
    TInputWord input_data[FH][FW_EXPAND];
    // Accumulator buffer.
    TPartialSum acc_buff_par[CH_PAR * W_PAR];

    // Read the input data for the current expanded window.
    for (size_t fh = 0; fh < FH; fh++) {
      for (size_t fw = 0; fw < FW_EXPAND; fw++) {
        input_data[fh][fw] = i_data[fh * FW_EXPAND + fw].read();
      }
    }

    // Read the weight data for the current filter.
    for (size_t fh = 0; fh < FH; fh++) {
      for (size_t fw = 0; fw < FW; fw++) {
        for (size_t i_ch_par = 0; i_ch_par < CH_PAR; i_ch_par++) {
          weight_data[fh][fw][i_ch_par] = i_weights[i_ch_par][fh * FW + fw];
        }
      }
    }

    // Read the bias data.
    for (size_t i_ch_par = 0; i_ch_par < CH_PAR; i_ch_par++) {
      bias_data[i_ch_par] = i_biases[i_ch_par][0];
    }

    // Initialize the accumulator buffer for the current block of output
    // channels and pixels.
    for (size_t i = 0; i < CH_PAR * W_PAR; i++) {
      acc_buff_par[i] = 0;
    }

    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      for (size_t i_ch_par = 0; i_ch_par < CH_PAR; i_ch_par++) {

        // Compute the index of the accumulator.
        size_t acc_index = i_w_par * CH_PAR + i_ch_par;

        for (size_t i_fh = 0; i_fh < FH; i_fh++) {
          for (size_t i_fw = 0; i_fw < FW; i_fw++) {

            // Compute the filter width index inside the expanded input window.
            size_t i_fw_expanded = i_fw + i_w_par * STRIDE_W;

            acc_buff_par[acc_index] +=
                input_data[i_fh][i_fw_expanded][i_ch_par] *
                weight_data[i_fh][i_fw][i_ch_par];
          }
        }

        // Finalize the output.
        TSum wide_acc = acc_buff_par[acc_index] + bias_data[i_ch_par];
        wide_acc = activation(wide_acc);
        TOutput output_value = quantizer(wide_acc);
        output_data[i_ch_par] = output_value;

        // Write the output data only after the computation of all
        // output channels for the current pixels.
        if (i_ch_par == CH_PAR - 1) {
          o_data[i_w_par].write(output_data);
        }
      }
    }
  }
};