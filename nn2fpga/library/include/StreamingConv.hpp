#pragma once
#include "hls_stream.h"
#include "ap_int.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>
#include <cassert>

/**
 * @brief StreamingConv implements a quantized convolution with only streaming
 * in input and output. Works only with NHWC data layout.
 *
 * @tparam TInputWord      Data type for input word (packed input channels).
 * @tparam TInput          Data type for individual input elements.
 * @tparam TWeightWord     Data type for weight word (packed weights).
 * @tparam TWeight         Data type for individual weight elements.
 * @tparam TBiasWord       Data type for bias word (packed biases).
 * @tparam TBias           Data type for individual bias elements.
 * @tparam TOutputWord     Data type for output word (packed output channels).
 * @tparam TOutput         Data type for individual output elements.
 * @tparam TSum            Data type for accumulator considered the bias.
 * @tparam TPartialSum     Data type for partial sum accumulator.
 * @tparam Quantizer       Quantizer functor type for output quantization.
 * @tparam Activation      Activation functor type for output activation.
 * @tparam OUT_CH          Number of output channels.
 * @tparam IN_CH           Number of input channels.
 * @tparam OUT_HEIGHT      Output feature map height.
 * @tparam OUT_WIDTH       Output feature map width.
 * @tparam GROUP           Number of groups for grouped convolution.
 * @tparam FH              Filter height.
 * @tparam FW              Filter width.
 * @tparam STRIDE_H        Stride along height.
 * @tparam STRIDE_W        Stride along width.
 * @tparam IN_CH_PAR       Parallelism factor for input channels.
 * @tparam OUT_CH_PAR      Parallelism factor for output channels.
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
 * @todo
 * - Add support for grouped convolutions.
 *
 */

template <typename TInputWord, typename TInput, typename TWeightWord,
          typename TWeight, typename TBiasWord, typename TBias,
          typename TOutputWord, typename TOutput, typename TSum,
          typename TPartialSum, typename Activation, typename Quantizer,
          size_t OUT_CH, size_t IN_CH, size_t OUT_HEIGHT, size_t OUT_WIDTH,
          size_t GROUP, size_t FH, size_t FW, size_t STRIDE_H, size_t STRIDE_W,
          size_t IN_CH_PAR, size_t OUT_CH_PAR, size_t W_PAR>
class StreamingConv {
  static constexpr size_t FW_EXPAND = FW + (W_PAR - 1) * STRIDE_W;
  static constexpr size_t CH_GROUPS = IN_CH * OUT_CH / (IN_CH_PAR * OUT_CH_PAR);

public:
  static_assert(OUT_HEIGHT > 0 && OUT_WIDTH > 0,
                "OUT_HEIGHT and OUT_WIDTH must be greater than 0");
  static_assert(W_PAR > 0, "W_PAR must be greater than 0");
  static_assert(OUT_CH_PAR > 0, "OUT_CH_PAR must be greater than 0");
  static_assert(IN_CH_PAR > 0, "IN_CH_PAR must be greater than 0");
  static_assert(FH > 0 && FW > 0, "FH and FW must be greater than 0");
  static_assert(STRIDE_H > 0 && STRIDE_W > 0, "STRIDE must be greater than 0");
  static_assert(GROUP > 0 && GROUP <= IN_CH,
                "GROUP must be between 1 and IN_CH");
  static_assert(IN_CH % GROUP == 0, "IN_CH must be a multiple of GROUP");
  static_assert(IN_CH_PAR > 0, "IN_CH_PAR must be greater than 0");
  static_assert(OUT_CH_PAR > 0, "OUT_CH_PAR must be greater than 0");
  static_assert(W_PAR > 0, "W_PAR must be greater than 0");
  static_assert(OUT_CH % OUT_CH_PAR == 0,
                "OUT_CH must be a multiple of OUT_CH_PAR");
  static_assert(IN_CH % IN_CH_PAR == 0,
                "IN_CH must be a multiple of IN_CH_PAR");
  static_assert(OUT_WIDTH % W_PAR == 0,
                "OUT_WIDTH must be a multiple of W_PAR");
  static_assert(GROUP == 1, "Grouped convolution not supported yet");

  StreamingConv() = default;

  struct StepState {
    // Loop iteration indexes.
    size_t i_hw = 0, i_ich = 0, i_och = 0;

    // Accumulator buffers.
    TPartialSum acc_buff[OUT_CH / OUT_CH_PAR][OUT_CH_PAR * W_PAR];

    // Input buffer to hold the input data.
    TInputWord input_data[FH][FW_EXPAND];

    PipelineDelayBuffer<TOutputWord> delayed_output[W_PAR];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i = 0; i < W_PAR; i++) {
        delayed_output[i] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status =
          ActorStatus(depth, OUT_HEIGHT * (OUT_WIDTH / W_PAR) *
                                 (OUT_CH / OUT_CH_PAR) * (IN_CH / IN_CH_PAR));
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

    // Accumulator buffer.
    // The order of the loops impose that for each input window, we process
    // all the output channels, thus we need to store an accumulator for
    // each output channel.
    // The number of accumulators used in parallel (i.e. the partitioning of the
    // memory) are determined by OUT_CH_PAR and W_PAR. This means that at each
    // clock cycle, the convolution will process OUT_CH_PAR output channels and
    // W_PAR input windows.
    TPartialSum acc_buff[OUT_CH / OUT_CH_PAR][OUT_CH_PAR * W_PAR];
#pragma HLS ARRAY_PARTITION variable = acc_buff dim = 2 complete

    // Input structure to hold the input data.
    TInputWord input_data[FH][FW_EXPAND];
#pragma HLS ARRAY_PARTITION variable = input_data dim = 0

    for (size_t i_hw = 0; i_hw < OUT_HEIGHT * OUT_WIDTH / W_PAR; i_hw++) {
      for (size_t i_ich = 0; i_ich < IN_CH; i_ich += IN_CH_PAR) {
      STREAMINGCONV_RUN_LOOP:
        for (size_t i_och = 0; i_och < OUT_CH; i_och += OUT_CH_PAR) {
#pragma HLS pipeline II = 1
          StreamingConv::pipeline_body(i_data, i_weights, i_biases, o_data,
                                       input_data, acc_buff[i_och / OUT_CH_PAR],
                                       i_ich, i_och);
        }
      }
    }
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
           TWeight i_weights[CH_GROUPS][OUT_CH_PAR * IN_CH_PAR][FH * FW],
           TBias i_biases[OUT_CH / OUT_CH_PAR][OUT_CH_PAR][1],
           hls::stream<TOutputWord> o_data[W_PAR]) {

    // Accumulator buffer.
    // The order of the loops impose that for each input window, we process
    // all the output channels, thus we need to store an accumulator for
    // each output channel.
    // The number of accumulators used in parallel (i.e. the partitioning of the
    // memory) are determined by OUT_CH_PAR and W_PAR. This means that at each
    // clock cycle, the convolution will process OUT_CH_PAR output channels and
    // W_PAR input windows.
    TPartialSum acc_buff[OUT_CH / OUT_CH_PAR][OUT_CH_PAR * W_PAR];
#pragma HLS ARRAY_PARTITION variable = acc_buff dim = 2 complete

    // Input structure to hold the input data.
    TInputWord input_data[FH][FW_EXPAND];
#pragma HLS ARRAY_PARTITION variable = input_data dim = 0

    for (size_t i_hw = 0; i_hw < OUT_HEIGHT * OUT_WIDTH / W_PAR; i_hw++) {
      for (size_t i_ich = 0, weight_group = 0; i_ich < IN_CH; i_ich += IN_CH_PAR) {
      STREAMINGCONV_RUN_LOOP:
        for (size_t i_och = 0, bias_group = 0; i_och < OUT_CH; i_och += OUT_CH_PAR, bias_group++, weight_group++) {
#pragma HLS pipeline II = 1
          StreamingConv::pipeline_body(i_data, i_weights[weight_group], i_biases[bias_group], o_data,
                                       input_data, acc_buff[i_och / OUT_CH_PAR],
                                       i_ich, i_och);
        }
      }
    }
  }

  template <size_t HLS_TAG>
  void run_allpartitioned(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
                          hls::stream<TWeightWord> i_weights[FH * FW],
                          hls::stream<TBiasWord> i_biases[1],
                          hls::stream<TOutputWord> o_data[W_PAR]) {

    // Accumulator buffer.
    // The order of the loops impose that for each input window, we process
    // all the output channels, thus we need to store an accumulator for
    // each output channel.
    // The number of accumulators used in parallel (i.e. the partitioning of the
    // memory) are determined by OUT_CH_PAR and W_PAR. This means that at each
    // clock cycle, the convolution will process OUT_CH_PAR output channels and
    // W_PAR input windows.
    TPartialSum acc_buff[OUT_CH / OUT_CH_PAR][OUT_CH_PAR * W_PAR];
#pragma HLS ARRAY_PARTITION variable = acc_buff dim = 0 complete

    // Input structure to hold the input data.
    TInputWord input_data[FH][FW_EXPAND];
#pragma HLS ARRAY_PARTITION variable = input_data dim = 0

    for (size_t i_hw = 0; i_hw < OUT_HEIGHT * OUT_WIDTH / W_PAR; i_hw++) {
      for (size_t i_ich = 0; i_ich < IN_CH; i_ich += IN_CH_PAR) {
      STREAMINGCONV_RUN_LOOP:
        for (size_t i_och = 0; i_och < OUT_CH; i_och += OUT_CH_PAR) {
#pragma HLS pipeline II = 1
          StreamingConv::pipeline_body(i_data, i_weights, i_biases, o_data,
                                       input_data, acc_buff[i_och / OUT_CH_PAR],
                                       i_ich, i_och);
        }
      }
    }
  }
  
  template <size_t HLS_TAG>
  void run_allpartitioned(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
                          TWeight i_weights[CH_GROUPS][OUT_CH_PAR * IN_CH_PAR][FH * FW],
                          TBias i_biases[OUT_CH / OUT_CH_PAR][OUT_CH_PAR][1],
                          hls::stream<TOutputWord> o_data[W_PAR]) {

    // Accumulator buffer.
    // The order of the loops impose that for each input window, we process
    // all the output channels, thus we need to store an accumulator for
    // each output channel.
    // The number of accumulators used in parallel (i.e. the partitioning of the
    // memory) are determined by OUT_CH_PAR and W_PAR. This means that at each
    // clock cycle, the convolution will process OUT_CH_PAR output channels and
    // W_PAR input windows.
    TPartialSum acc_buff[OUT_CH / OUT_CH_PAR][OUT_CH_PAR * W_PAR];
#pragma HLS ARRAY_PARTITION variable = acc_buff dim = 0 complete

    // Input structure to hold the input data.
    TInputWord input_data[FH][FW_EXPAND];
#pragma HLS ARRAY_PARTITION variable = input_data dim = 0

    for (size_t i_hw = 0; i_hw < OUT_HEIGHT * OUT_WIDTH / W_PAR; i_hw++) {
      for (size_t i_ich = 0, weight_group = 0; i_ich < IN_CH; i_ich += IN_CH_PAR) {
      STREAMINGCONV_RUN_LOOP:
        for (size_t i_och = 0, bias_group = 0; i_och < OUT_CH; i_och += OUT_CH_PAR, bias_group++, weight_group++) {
#pragma HLS pipeline II = 1
          StreamingConv::pipeline_body(i_data, i_weights[weight_group], i_biases[bias_group], o_data,
                                       input_data, acc_buff[i_och / OUT_CH_PAR],
                                       i_ich, i_och);
        }
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
    if (st.i_och == 0) {
      for (size_t i_in_stream = 0; i_in_stream < FH * FW_EXPAND;
           i_in_stream++) {
        if (i_data[i_in_stream].empty()) {
          firing_condition = false;
        }
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
    if (st.i_ich == IN_CH - IN_CH_PAR) {
      if (i_biases[0].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      hls::stream<TOutputWord> instant_output_stream[W_PAR];
      StreamingConv::pipeline_body(
          i_data, i_weights, i_biases, instant_output_stream, st.input_data,
          st.acc_buff[st.i_och / OUT_CH_PAR], st.i_ich, st.i_och);

      st.i_och += OUT_CH_PAR;
      if (st.i_och >= OUT_CH) {
        // If we have processed all output channels, reset the index and
        // increment the input channels index.
        st.i_och = 0;
        st.i_ich += IN_CH_PAR;
      }
      if (st.i_ich >= IN_CH) {
        // Reset input channel index if we have processed all
        // input channels and increment the pixel index.
        st.i_ich = 0;
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

  ActorStatus
  step(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
       TWeight i_weights[CH_GROUPS][OUT_CH_PAR * IN_CH_PAR][FH * FW],
       TBias i_biases[OUT_CH / OUT_CH_PAR][OUT_CH_PAR][1],
       hls::stream<TOutputWord> o_data[W_PAR]) {
    // Get the state for this instance.
    auto it = registry().find(this);
    assert(it != registry().end() && "Instance not initialized");
    auto &st = it->second;

    // Compute firing condition.
    bool firing_condition = true;

    // Check non empty input streams. Input data are read only at the
    // beginning of the computation of the output channels.
    if (st.i_och == 0) {
      for (size_t i_in_stream = 0; i_in_stream < FH * FW_EXPAND;
           i_in_stream++) {
        if (i_data[i_in_stream].empty()) {
          firing_condition = false;
        }
      }
    }

    if (firing_condition) {

      hls::stream<TOutputWord> instant_output_stream[W_PAR];
      StreamingConv::pipeline_body(
          i_data, i_weights[st.i_ich / IN_CH_PAR], i_biases[st.i_och / OUT_CH_PAR], instant_output_stream, st.input_data,
          st.acc_buff[st.i_och / OUT_CH_PAR], st.i_ich, st.i_och);

      st.i_och += OUT_CH_PAR;
      if (st.i_och >= OUT_CH) {
        // If we have processed all output channels, reset the index and
        // increment the input channels index.
        st.i_och = 0;
        st.i_ich += IN_CH_PAR;
      }
      if (st.i_ich >= IN_CH) {
        // Reset input channel index if we have processed all
        // input channels and increment the pixel index.
        st.i_ich = 0;
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
                            hls::stream<TOutputWord> o_data[W_PAR],
                            TInputWord input_data[FH][FW_EXPAND],
                            TPartialSum acc_buff_par[OUT_CH_PAR * W_PAR], size_t i_ich,
                            size_t i_och) {
#pragma HLS inline

    // Quantizer instance.
    Quantizer quantizer;
    // Activation instance.
    Activation activation;
    // Output structure to hold the results.
    TOutputWord output_data;
    // Weight structure to hold the weights.
    TWeightWord weight_data[FH][FW];
    // Bias structure to hold the biases.
    TBiasWord bias_data;

    // Read the input data for the current expanded window.
    if (i_och == 0) {
      for (size_t fh = 0; fh < FH; fh++) {
        for (size_t fw = 0; fw < FW_EXPAND; fw++) {
          input_data[fh][fw] = i_data[fh * FW_EXPAND + fw].read();
        }
      }
    }

    // Read the weight data for the current filter.
    for (size_t fh = 0; fh < FH; fh++) {
      for (size_t fw = 0; fw < FW; fw++) {
        weight_data[fh][fw] = i_weights[fh * FW + fw].read();
      }
    }

    // Read the bias data only at the end of the computation of the output.
    if (i_ich == IN_CH - IN_CH_PAR) {
      bias_data = i_biases[0].read();
    }

    // Initialize the accumulator buffer for the current block of output
    // channels and pixels.
    if (i_ich == 0) {
      for (size_t i = 0; i < OUT_CH_PAR * W_PAR; i++) {
        acc_buff_par[i] = 0;
      }
    }

    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR; i_och_par++) {

        // Compute the index of the accumulator.
        size_t acc_index = i_w_par * OUT_CH_PAR + i_och_par;

        for (size_t i_fh = 0; i_fh < FH; i_fh++) {
          for (size_t i_fw = 0; i_fw < FW; i_fw++) {

            // Compute the filter width index inside the expanded input window.
            size_t i_fw_expanded = i_fw + i_w_par * STRIDE_W;

            for (size_t i_ich_par = 0; i_ich_par < IN_CH_PAR; i_ich_par++) {
              acc_buff_par[acc_index] +=
                  input_data[i_fh][i_fw_expanded][i_ich_par] *
                  weight_data[i_fh][i_fw][i_och_par * IN_CH_PAR + i_ich_par];
            }
          }
        }

        // If we are at the last block of input channels, read the bias and
        // finalize the output.
        if (i_ich == IN_CH - IN_CH_PAR) {
          TSum wide_acc = acc_buff_par[acc_index] + bias_data[i_och_par];
          wide_acc = activation(wide_acc);
          TOutput output_value = quantizer(wide_acc);
          output_data[i_och_par] = output_value;

          // If we are at the last output channel of the block, write the output
          // data to the output stream.
          if (i_och_par == OUT_CH_PAR - 1) {
            o_data[i_w_par].write(output_data);
          }
        }
      }
    }
  }

  static void pipeline_body(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
                            TWeight i_weights[OUT_CH_PAR * IN_CH_PAR][FH * FW],
                            TBias i_biases[OUT_CH_PAR][1],
                            hls::stream<TOutputWord> o_data[W_PAR],
                            TInputWord input_data[FH][FW_EXPAND],
                            TPartialSum acc_buff_par[OUT_CH_PAR * W_PAR], size_t i_ich,
                            size_t i_och) {
#pragma HLS inline

    // Quantizer instance.
    Quantizer quantizer;
    // Activation instance.
    Activation activation;
    // Output structure to hold the results.
    TOutputWord output_data;
    // Weight structure to hold the weights.
    TWeightWord weight_data[FH][FW];
    // Bias structure to hold the biases.
    TBiasWord bias_data;

    // Read the input data for the current expanded window.
    if (i_och == 0) {
      for (size_t fh = 0; fh < FH; fh++) {
        for (size_t fw = 0; fw < FW_EXPAND; fw++) {
          input_data[fh][fw] = i_data[fh * FW_EXPAND + fw].read();
        }
      }
    }

    // Read the weight data for the current filter.
    for (size_t fh = 0; fh < FH; fh++) {
      for (size_t fw = 0; fw < FW; fw++) {
        for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR; i_och_par++) {
          for (size_t i_ich_par = 0; i_ich_par < IN_CH_PAR; i_ich_par++) {
            weight_data[fh][fw][i_och_par * IN_CH_PAR + i_ich_par] =
                i_weights[i_och_par * IN_CH_PAR + i_ich_par][fh * FW + fw];
          }
        }
      }
    }

    // Read the bias data only at the end of the computation of the output.
    if (i_ich == IN_CH - IN_CH_PAR) {
      for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR; i_och_par++) {
        bias_data[i_och_par] = i_biases[i_och_par][0];
      }
    }

    // Initialize the accumulator buffer for the current block of output
    // channels and pixels.
    if (i_ich == 0) {
      for (size_t i = 0; i < OUT_CH_PAR * W_PAR; i++) {
        acc_buff_par[i] = 0;
      }
    }

    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR; i_och_par++) {

        // Compute the index of the accumulator.
        size_t acc_index = i_w_par * OUT_CH_PAR + i_och_par;

        for (size_t i_fh = 0; i_fh < FH; i_fh++) {
          for (size_t i_fw = 0; i_fw < FW; i_fw++) {

            // Compute the filter width index inside the expanded input window.
            size_t i_fw_expanded = i_fw + i_w_par * STRIDE_W;

            for (size_t i_ich_par = 0; i_ich_par < IN_CH_PAR; i_ich_par++) {
              acc_buff_par[acc_index] +=
                  input_data[i_fh][i_fw_expanded][i_ich_par] *
                  weight_data[i_fh][i_fw][i_och_par * IN_CH_PAR + i_ich_par];
            }
          }
        }

        // If we are at the last block of input channels, read the bias and
        // finalize the output.
        if (i_ich == IN_CH - IN_CH_PAR) {
          TSum wide_acc = acc_buff_par[acc_index] + bias_data[i_och_par];
          wide_acc = activation(wide_acc);
          TOutput output_value = quantizer(wide_acc);
          output_data[i_och_par] = output_value;

          // If we are at the last output channel of the block, write the output
          // data to the output stream.
          if (i_och_par == OUT_CH_PAR - 1) {
            o_data[i_w_par].write(output_data);
          }
        }
      }
    }
  }
};