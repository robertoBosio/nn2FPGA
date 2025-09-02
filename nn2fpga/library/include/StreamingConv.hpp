#pragma once
#include "hls_stream.h"
#include "ap_int.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>

template <typename TInputStruct, typename TInput, typename TWeightStruct,
          typename TBiasStruct, typename TOutputStruct, typename TOutput,
          typename TAcc, typename Quantizer, size_t OUT_CH, size_t IN_CH,
          size_t OUT_HEIGHT, size_t OUT_WIDTH, size_t GROUP, size_t FH,
          size_t FW, size_t STRIDE_H, size_t STRIDE_W, size_t IN_CH_PAR,
          size_t OUT_CH_PAR, size_t W_PAR>
class StreamingConv {
  static constexpr size_t FW_EXPAND = FW + (W_PAR - 1) * STRIDE_W;

public:
  static_assert(OUT_HEIGHT > 0 && OUT_WIDTH > 0,
                "OUT_HEIGHT and OUT_WIDTH must be greater than 0");
  static_assert(W_PAR > 0, "W_PAR must be greater than 0");
  static_assert(OUT_CH_PAR > 0, "OUT_CH_PAR must be greater than 0");
  static_assert(IN_CH_PAR > 0, "IN_CH_PAR must be greater than 0");
  static_assert(FH > 0 && FW > 0, "FH and FW must be greater than 0");
  static_assert(STRIDE_H > 0 && STRIDE_W > 0, "STRIDE must be greater than 0");
  static_assert(GROUP > 0 && GROUP <= IN_CH,
                "GROUP must be between 0 and IN_CH");
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

  StreamingConv() : StreamingConv(1) {}

  StreamingConv(size_t pipeline_depth)
      : STEP_i_hw(0), STEP_i_ich(0), STEP_i_och(0),
        STEP_pipeline_depth(pipeline_depth),
        STEP_actor_status(pipeline_depth,
                          (OUT_HEIGHT * OUT_WIDTH / W_PAR) * (OUT_CH / OUT_CH_PAR) *
                              (IN_CH / IN_CH_PAR)) {
    for (size_t i = 0; i < W_PAR; ++i) {
      STEP_delayed_output[i] =
          PipelineDelayBuffer<TOutputStruct>(pipeline_depth);
    }
  }

  void
  run(hls::stream<TInputStruct> i_data[FH * FW_EXPAND],
      hls::stream<TWeightStruct> i_weights[FH * FW],
      hls::stream<TBiasStruct> i_biases[1],
      hls::stream<TOutputStruct> o_data[W_PAR]) {

    // Accumulator buffer.
    // The order of the loops impose that for each input window, we process
    // all the output channels, thus we need to store an accumulator for
    // each output channel.
    // The number of accumulators used in parallel (i.e. the partitioning of the
    // memory) are determined by OUT_CH_PAR and W_PAR. This means that at each
    // clock cycle, the convolution will process OUT_CH_PAR output channels and
    // W_PAR input windows.
    TAcc acc_buff[OUT_CH / OUT_CH_PAR][OUT_CH_PAR * W_PAR];

    // Input structure to hold the input data.
    TInputStruct input_data[FH][FW_EXPAND];

    for (size_t i_hw = 0; i_hw < OUT_HEIGHT * OUT_WIDTH / W_PAR; i_hw++) {
      for (size_t i_ich = 0; i_ich < IN_CH; i_ich += IN_CH_PAR) {
        for (size_t i_och = 0; i_och < OUT_CH; i_och += OUT_CH_PAR) {
#pragma HLS pipeline II = 1
          StreamingConv::pipeline_body(i_data, i_weights, i_biases, o_data,
                                       input_data, acc_buff[i_och / OUT_CH_PAR],
                                       i_ich, i_och);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputStruct> i_data[FH * FW_EXPAND],
                   hls::stream<TWeightStruct> i_weights[FH * FW],
                   hls::stream<TBiasStruct> i_biases[1],
                   hls::stream<TOutputStruct> o_data[W_PAR]) {
    bool firing_condition = true;

    // Check non empty input streams. Input data are read only at the
    // beginning of the computation of the output channels.
    if (STEP_i_och == 0) {
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
    if (STEP_i_ich == IN_CH - IN_CH_PAR) {
      if (i_biases[0].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      hls::stream<TOutputStruct> instant_output_stream[W_PAR];
      StreamingConv::pipeline_body(
          i_data, i_weights, i_biases, instant_output_stream, STEP_input_data,
          STEP_acc_buff[STEP_i_och / OUT_CH_PAR], STEP_i_ich, STEP_i_och);

      STEP_i_och += OUT_CH_PAR;
      if (STEP_i_och >= OUT_CH) {
        // If we have processed all output channels, reset the index and
        // increment the input channels index.
        STEP_i_och = 0;
        STEP_i_ich += IN_CH_PAR;
      }
      if (STEP_i_ich >= IN_CH) {
        // Reset input channel index if we have processed all
        // input channels and increment the pixel index.
        STEP_i_ich = 0;
        STEP_i_hw++;
      }
      if (STEP_i_hw >= OUT_HEIGHT * OUT_WIDTH / W_PAR) {
        STEP_i_hw = 0;
      }

      // Insert the firing status for the current step.
      STEP_actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_w_par = 0; i_w_par < W_PAR; ++i_w_par) {
        if (!instant_output_stream[i_w_par].empty()) {
          STEP_delayed_output[i_w_par].push(
              instant_output_stream[i_w_par].read(), true);
        } else {
          // If the output stream is empty, push a placeholder.
          STEP_delayed_output[i_w_par].push(TOutputStruct(), false);
        }
      }
    } else {
      // If no data is available, push empty outputs.
      for (size_t i_w_par = 0; i_w_par < W_PAR; ++i_w_par) {
        STEP_delayed_output[i_w_par].push(TOutputStruct(), false);
      }
    }

    // Advance the state of the actor firings.
    STEP_actor_status.advance();

    // Write the output data to the output stream.
    TOutputStruct out;
    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      if (STEP_delayed_output[i_w_par].pop(out)) {
        o_data[i_w_par].write(out);
      }
    }

    // Return the current actor status.
    return STEP_actor_status;
  }

private:
  // State variables for the step function.
  size_t STEP_i_hw;
  size_t STEP_i_ich;
  size_t STEP_i_och;
  size_t STEP_pipeline_depth;
  TInputStruct STEP_input_data[FH][FW_EXPAND];
  TAcc STEP_acc_buff[OUT_CH / OUT_CH_PAR][OUT_CH_PAR * W_PAR];

  // CSDFG state variables
  ActorStatus STEP_actor_status;
  PipelineDelayBuffer<TOutputStruct> STEP_delayed_output[W_PAR];

  static void pipeline_body(hls::stream<TInputStruct> i_data[FH * FW_EXPAND],
                            hls::stream<TWeightStruct> i_weights[FH * FW],
                            hls::stream<TBiasStruct> i_biases[1],
                            hls::stream<TOutputStruct> o_data[W_PAR],
                            TInputStruct input_data[FH][FW_EXPAND],
                            TAcc acc_buff_par[OUT_CH_PAR * W_PAR],
                            size_t i_ich, size_t i_och) {
#pragma HLS inline

    Quantizer quantizer;
    // Output structure to hold the results.
    TOutputStruct output_data;
    // Weight structure to hold the weights.
    TWeightStruct weight_data[FH][FW];
    // Bias structure to hold the biases.
    TBiasStruct bias_data;

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
          ap_int<32> wide_acc = acc_buff_par[acc_index] + bias_data[i_och_par];
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