#pragma once
#include <cstddef>
#include "utils/CSDFG_utils.hpp"
#include "hls_stream.h"

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, typename Quantizer, size_t IN_HEIGHT,
          size_t IN_WIDTH, size_t IN_CH, size_t IN_W_PAR, size_t OUT_W_PAR,
          size_t IN_CH_PAR, size_t OUT_CH_PAR>
class BandwidthAdjustIncreaseStreams {
public:
  static_assert(IN_W_PAR > 0, "IN_W_PAR must be greater than 0");
  static_assert(OUT_W_PAR > 0, "OUT_W_PAR must be greater than 0");
  static_assert(IN_W_PAR < OUT_W_PAR, "IN_W_PAR must be less than OUT_W_PAR");
  static_assert(OUT_W_PAR % IN_W_PAR == 0,
                "OUT_W_PAR must be a multiple of IN_W_PAR");
  static_assert(IN_WIDTH % IN_W_PAR == 0,
                "IN_WIDTH must be a multiple of IN_W_PAR");
  static_assert(IN_WIDTH % OUT_W_PAR == 0,
                "IN_WIDTH must be a multiple of OUT_W_PAR");
  static_assert(IN_CH % IN_CH_PAR == 0,
                "IN_CH_PAR must be a multiple of CH_PAR");
  static_assert(IN_CH % OUT_CH_PAR == 0,
                "OUT_CH_PAR must be a multiple of CH_PAR");
  static_assert(IN_CH_PAR == OUT_CH_PAR,
                "IN_CH_PAR must be equal to OUT_CH_PAR");

  BandwidthAdjustIncreaseStreams() : BandwidthAdjustIncreaseStreams(1) {}

  BandwidthAdjustIncreaseStreams(size_t pipeline_depth)
      : STEP_i_hw(0), STEP_i_out_stream(0), STEP_i_ch(0),
        STEP_pipeline_depth(pipeline_depth),
        STEP_actor_status(pipeline_depth, IN_HEIGHT * IN_WIDTH * IN_CH /
                                              (OUT_CH_PAR * IN_W_PAR)) {
    for (size_t i = 0; i < OUT_W_PAR; ++i) {
      STEP_delayed_output[i] =
          PipelineDelayBuffer<TOutputStruct>(pipeline_depth);
    }
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
           hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR]) {
    for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH; i_hw += OUT_W_PAR) {
      for (size_t i_out_stream = 0; i_out_stream < OUT_W_PAR;
           i_out_stream += IN_W_PAR) {
      BANDWIDTHADJUSTINCREASESTREAMS_RUN_LOOP:
        for (size_t i_ch = 0; i_ch < IN_CH; i_ch += OUT_CH_PAR) {
#pragma HLS pipeline II = 1
          BandwidthAdjustIncreaseStreams::pipeline_body(
              input_data_stream, output_data_stream, i_out_stream);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
                   hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR]) {
    bool firing_condition = true;
    for (size_t i_in_stream = 0; i_in_stream < IN_W_PAR; i_in_stream++) {
      if (input_data_stream[i_in_stream].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      hls::stream<TOutputStruct> instant_output_stream[OUT_W_PAR];
      BandwidthAdjustIncreaseStreams::pipeline_body(
          input_data_stream, instant_output_stream, STEP_i_out_stream);

      STEP_i_ch += OUT_CH_PAR;
      if (STEP_i_ch >= IN_CH) {
        // If we have processed all input channels, reset the index and
        // increment the height/width index.
        STEP_i_ch = 0;
        STEP_i_out_stream += IN_W_PAR;
      }
      if (STEP_i_out_stream >= OUT_W_PAR) {
        // If we have processed all output streams, reset the index and
        // increment the height/width index.
        STEP_i_out_stream = 0;
        STEP_i_hw += OUT_W_PAR;
      }
      if (STEP_i_hw >= IN_HEIGHT * IN_WIDTH) {
        // Reset height and width index if we have processed all data.
        STEP_i_hw = 0;
      }

      // Insert the firing status for the current step.
      STEP_actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++) {
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
      for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++) {
        STEP_delayed_output[i_w_par].push(TOutputStruct(), false);
      }
    }

    // Advance the state of the actor firings.
    STEP_actor_status.advance();

    // Write the output data to the output stream.
    TOutputStruct out;
    for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++) {
      if (STEP_delayed_output[i_w_par].pop(out)) {
        output_data_stream[i_w_par].write(out);
      }
    }

    // Return the current firing iteration index.
    return STEP_actor_status;
  }

private:
  size_t STEP_i_hw;           // Index for height and width.
  size_t STEP_i_out_stream;   // Index for output stream.
  size_t STEP_i_ch;           // Index for channels.
  size_t STEP_pipeline_depth; // Pipeline depth for the actor.

  // CSDFG state variables
  ActorStatus STEP_actor_status;
  PipelineDelayBuffer<TOutputStruct>
      STEP_delayed_output[OUT_W_PAR]; // Delayed output buffers to maintain
                                      // pipeline depth for each parallel
                                      // output

  static void
  pipeline_body(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
                hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR],
                size_t i_out_stream) {
#pragma HLS inline
    TInputStruct
        s_input_struct; // Input structure to read data from the input stream.
    TOutputStruct s_output_struct; // Output structure to hold the results.
    Quantizer quantizer;           // Quantizer instance for quantization.

    for (size_t i_w_par = 0; i_w_par < IN_W_PAR; i_w_par++) {
      // Read the input data structure from the input stream.
      s_input_struct = input_data_stream[i_w_par].read();

      for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR; i_och_par++) {
        // Extract the data for the current pixel output channel.
        TInput s_input_data = s_input_struct[i_och_par];

        // Quantize the input data.
        TOutput s_output_data = quantizer(s_input_data);

        // Store the quantized data in the output structure.
        s_output_struct[i_och_par] = s_output_data;
      }
      output_data_stream[i_out_stream + i_w_par].write(s_output_struct);
    }
  }
};

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, typename Quantizer, size_t IN_HEIGHT,
          size_t IN_WIDTH, size_t IN_CH, size_t IN_W_PAR, size_t OUT_W_PAR,
          size_t IN_CH_PAR, size_t OUT_CH_PAR>
class BandwidthAdjustDecreaseStreams {
public:
  static_assert(IN_W_PAR > 0, "IN_W_PAR must be greater than 0");
  static_assert(OUT_W_PAR > 0, "OUT_W_PAR must be greater than 0");
  static_assert(IN_W_PAR > OUT_W_PAR, "OUT_W_PAR must be less than IN_W_PAR");
  static_assert(IN_W_PAR % OUT_W_PAR == 0,
                "OUT_W_PAR must be a multiple of IN_W_PAR");
  static_assert(IN_WIDTH % IN_W_PAR == 0,
                "IN_WIDTH must be a multiple of IN_W_PAR");
  static_assert(IN_WIDTH % OUT_W_PAR == 0,
                "IN_WIDTH must be a multiple of OUT_W_PAR");
  static_assert(IN_CH % IN_CH_PAR == 0,
                "IN_CH_PAR must be a multiple of CH_PAR");
  static_assert(IN_CH % OUT_CH_PAR == 0,
                "OUT_CH_PAR must be a multiple of CH_PAR");
  static_assert(IN_CH_PAR == OUT_CH_PAR,
                "IN_CH_PAR must be equal to OUT_CH_PAR");

  BandwidthAdjustDecreaseStreams() : BandwidthAdjustDecreaseStreams(1) {}

  BandwidthAdjustDecreaseStreams(size_t pipeline_depth)
      : STEP_i_hw(0), STEP_i_in_stream(0), STEP_i_ch(0),
        STEP_pipeline_depth(pipeline_depth),
        STEP_actor_status(pipeline_depth, IN_HEIGHT * IN_WIDTH * IN_CH /
                                              (OUT_CH_PAR * OUT_W_PAR)) {
    for (size_t i = 0; i < OUT_W_PAR; ++i) {
      STEP_delayed_output[i] =
          PipelineDelayBuffer<TOutputStruct>(pipeline_depth);
    }
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
           hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR]) {
    for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH; i_hw += IN_W_PAR) {
      for (size_t i_in_stream = 0; i_in_stream < IN_W_PAR;
           i_in_stream += OUT_W_PAR) {
      BANDWIDTHADJUSTDECREASESTREAMS_RUN_LOOP:
        for (size_t i_ch = 0; i_ch < IN_CH; i_ch += OUT_CH_PAR) {
#pragma HLS pipeline II = 1
          BandwidthAdjustDecreaseStreams::pipeline_body(
              input_data_stream, output_data_stream, i_in_stream);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
                   hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR]) {

    // Check if there is data in the input streams considered in this step.
    bool firing_condition = true;
    for (size_t i_in_stream = 0; i_in_stream < OUT_W_PAR; i_in_stream++) {
      if (input_data_stream[STEP_i_in_stream + i_in_stream].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {

      hls::stream<TOutputStruct> instant_output_stream[OUT_W_PAR];
      BandwidthAdjustDecreaseStreams::pipeline_body(
          input_data_stream, instant_output_stream, STEP_i_in_stream);

      STEP_i_ch += OUT_CH_PAR;
      if (STEP_i_ch >= IN_CH) {
        // If we have processed all input channels, reset the index and
        // increment the height/width index.
        STEP_i_ch = 0;
        STEP_i_in_stream += OUT_W_PAR;
      }
      if (STEP_i_in_stream >= IN_W_PAR) {
        // If we have processed all output streams, reset the index and
        // increment the height/width index.
        STEP_i_in_stream = 0;
        STEP_i_hw += IN_W_PAR;
      }
      if (STEP_i_hw >= IN_HEIGHT * IN_WIDTH) {
        STEP_i_hw =
            0; // Reset height and width index if we have processed all data.
      }

      // Insert the firing status for the current step.
      STEP_actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i = 0; i < OUT_W_PAR; ++i) {
        STEP_delayed_output[i].push(instant_output_stream[i].read(), true);
      }
    } else {
      // If no data is available, push empty outputs.
      for (size_t i = 0; i < OUT_W_PAR; ++i) {
        STEP_delayed_output[i].push(TOutputStruct(), false);
      }
    }

    // Advance the state of the actor firings.
    STEP_actor_status.advance();

    // Write the output data to the output stream.
    TOutputStruct out;
    for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++) {
      if (STEP_delayed_output[i_w_par].pop(out)) {
        output_data_stream[i_w_par].write(
            out); // Write the output to the stream.
      }
    }

    // Return the current actor status.
    return STEP_actor_status;
  }

private:
  size_t STEP_i_hw;           // Index for height and width.
  size_t STEP_i_in_stream;    // Index for input stream.
  size_t STEP_i_ch;           // Index for channels.
  size_t STEP_pipeline_depth; // Pipeline depth for the actor.

  // CSDFG state variables
  ActorStatus STEP_actor_status;
  PipelineDelayBuffer<TOutputStruct>
      STEP_delayed_output[OUT_W_PAR]; // Delayed output buffers to maintain
                                      // pipeline depth for each parallel
                                      // output

  static void
  pipeline_body(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
                hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR],
                size_t i_in_stream) {
#pragma HLS inline
    TInputStruct
        s_input_struct; // Input structure to read data from the input stream.
    TOutputStruct s_output_struct; // Output structure to hold the results.
    Quantizer quantizer;           // Quantizer instance for quantization.

    for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++) {
      // Read the input data structure from the input stream.
      s_input_struct = input_data_stream[i_in_stream + i_w_par].read();

      for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR; i_och_par++) {
        // Extract the data for the current pixel output channel.
        TInput s_input_data = s_input_struct[i_och_par];

        // Quantize the input data.
        TOutput s_output_data = quantizer(s_input_data);

        // Store the quantized data in the output structure.
        s_output_struct[i_och_par] = s_output_data;
      }
      output_data_stream[i_w_par].write(s_output_struct);
    }
  }
};

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, typename Quantizer, size_t IN_HEIGHT,
          size_t IN_WIDTH, size_t IN_CH, size_t IN_W_PAR, size_t OUT_W_PAR,
          size_t IN_CH_PAR, size_t OUT_CH_PAR>
class BandwidthAdjustIncreaseChannels {
public:
  static_assert(IN_W_PAR > 0, "IN_W_PAR must be greater than 0");
  static_assert(OUT_W_PAR > 0, "OUT_W_PAR must be greater than 0");
  static_assert(IN_CH_PAR < OUT_CH_PAR,
                "IN_CH_PAR must be less than OUT_CH_PAR");
  static_assert(OUT_CH_PAR % IN_CH_PAR == 0,
                "OUT_CH_PAR must be a multiple of IN_CH_PAR");
  static_assert(IN_WIDTH % IN_W_PAR == 0,
                "IN_WIDTH must be a multiple of IN_W_PAR");
  static_assert(IN_WIDTH % OUT_W_PAR == 0,
                "IN_WIDTH must be a multiple of OUT_W_PAR");
  static_assert(IN_W_PAR == OUT_W_PAR, "IN_W_PAR must be equal to OUT_W_PAR");
  static_assert(IN_CH % IN_CH_PAR == 0,
                "IN_CH must be a multiple of IN_CH_PAR");
  static_assert(IN_CH % OUT_CH_PAR == 0,
                "IN_CH must be a multiple of OUT_CH_PAR");

  BandwidthAdjustIncreaseChannels() : BandwidthAdjustIncreaseChannels(1) {}

  BandwidthAdjustIncreaseChannels(size_t pipeline_depth)
      : STEP_i_hw(0), STEP_i_och_par(0), STEP_i_ch(0),
        STEP_pipeline_depth(pipeline_depth),
        STEP_actor_status(pipeline_depth, IN_HEIGHT * IN_WIDTH * IN_CH /
                                              (IN_CH_PAR * IN_W_PAR)) {
    for (size_t i = 0; i < OUT_W_PAR; ++i) {
      STEP_delayed_output[i] =
          PipelineDelayBuffer<TOutputStruct>(pipeline_depth);
    }
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
           hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR]) {
    TOutputStruct
        output_data[OUT_W_PAR]; // Output structure to hold the results.
    for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH; i_hw += IN_W_PAR) {
      for (size_t i_ch = 0; i_ch < IN_CH; i_ch += OUT_CH_PAR) {
      BANDWIDTHADJUSTINCREASECHANNELS_RUN_LOOP:
        for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR;
             i_och_par += IN_CH_PAR) {
#pragma HLS pipeline II = 1
          BandwidthAdjustIncreaseChannels::pipeline_body(
              input_data_stream, output_data_stream, output_data, i_och_par);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
                   hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR]) {

    bool firing_condition = true;
    for (size_t i_in_stream = 0; i_in_stream < IN_W_PAR; i_in_stream++) {
      if (input_data_stream[i_in_stream].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {
      hls::stream<TOutputStruct> instant_output_stream[OUT_W_PAR];
      BandwidthAdjustIncreaseChannels::pipeline_body(
          input_data_stream, instant_output_stream, STEP_output_data,
          STEP_i_och_par);

      STEP_i_och_par += IN_CH_PAR;
      if (STEP_i_och_par >= OUT_CH_PAR) {
        // If we have processed all input channels, reset the index and
        // increment the height/width index.
        STEP_i_och_par = 0;
        STEP_i_ch += OUT_CH_PAR;
      }
      if (STEP_i_ch >= IN_CH) {
        // If we have processed all output streams, reset the index and
        // increment the height/width index.
        STEP_i_ch = 0;
        STEP_i_hw += IN_W_PAR;
      }
      if (STEP_i_hw >= IN_HEIGHT * IN_WIDTH) {
        STEP_i_hw =
            0; // Reset height and width index if we have processed all data.
      }

      // Insert the firing status for the current step.
      STEP_actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i = 0; i < OUT_W_PAR; ++i) {
        if (!instant_output_stream[i].empty()) {
          STEP_delayed_output[i].push(instant_output_stream[i].read(), true);
        } else {
          // If the output stream is empty, push a placeholder.
          STEP_delayed_output[i].push(TOutputStruct(), false);
        }
      }
    } else {
      // If no data is available, push empty outputs.
      for (size_t i = 0; i < OUT_W_PAR; ++i) {
        STEP_delayed_output[i].push(TOutputStruct(), false);
      }
    }

    // Advance the state of the actor firings.
    STEP_actor_status.advance();

    // Write the output data to the output stream.
    TOutputStruct out;
    for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++) {
      if (STEP_delayed_output[i_w_par].pop(out)) {
        output_data_stream[i_w_par].write(out);
      }
    }

    // Return the current firing iteration index.
    return STEP_actor_status;
  }

private:
  TOutputStruct
      STEP_output_data[OUT_W_PAR]; // Output structure to hold the results.
  size_t STEP_i_hw;                // Index for height and width.
  size_t STEP_i_och_par;           // Index for output channels.
  size_t STEP_i_ch;                // Index for input channels.
  size_t STEP_pipeline_depth;      // Pipeline depth for the actor.

  // CSDFG state variables
  ActorStatus STEP_actor_status;
  PipelineDelayBuffer<TOutputStruct>
      STEP_delayed_output[OUT_W_PAR]; // Delayed output buffers to maintain
                                      // pipeline depth for each parallel
                                      // output

  static void
  pipeline_body(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
                hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR],
                TOutputStruct s_output_struct[OUT_W_PAR], size_t i_och_par) {
#pragma HLS inline
    TInputStruct
        s_input_struct;  // Input structure to read data from the input stream.
    Quantizer quantizer; // Quantizer instance for quantization.

    for (size_t i_w_par = 0; i_w_par < IN_W_PAR; i_w_par++) {
      // Read the input data structure from the input stream.
      s_input_struct = input_data_stream[i_w_par].read();

      for (size_t i_ich_par = 0; i_ich_par < IN_CH_PAR; i_ich_par++) {
        // Extract the data for the current pixel channel.
        TInput s_input_data = s_input_struct[i_ich_par];

        // Quantize the input data.
        TOutput s_output_data = quantizer(s_input_data);

        // Store the quantized data in the output structure.
        s_output_struct[i_w_par][i_och_par + i_ich_par] = s_output_data;
      }

      // If we have processed all output channels, write the output structure to
      // the output stream.
      if (i_och_par == OUT_CH_PAR - IN_CH_PAR) {
        output_data_stream[i_w_par].write(s_output_struct[i_w_par]);
      }
    }
  }
};

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, typename Quantizer, size_t IN_HEIGHT,
          size_t IN_WIDTH, size_t IN_CH, size_t IN_W_PAR, size_t OUT_W_PAR,
          size_t IN_CH_PAR, size_t OUT_CH_PAR>
class BandwidthAdjustDecreaseChannels {
public:
  static_assert(IN_W_PAR > 0, "IN_W_PAR must be greater than 0");
  static_assert(OUT_W_PAR > 0, "OUT_W_PAR must be greater than 0");
  static_assert(IN_CH_PAR > OUT_CH_PAR,
                "OUT_CH_PAR must be less than IN_CH_PAR");
  static_assert(IN_CH_PAR % OUT_CH_PAR == 0,
                "IN_CH_PAR must be a multiple of OUT_CH_PAR");
  static_assert(IN_WIDTH % IN_W_PAR == 0,
                "IN_WIDTH must be a multiple of IN_W_PAR");
  static_assert(IN_WIDTH % OUT_W_PAR == 0,
                "IN_WIDTH must be a multiple of OUT_W_PAR");
  static_assert(IN_W_PAR == OUT_W_PAR, "IN_W_PAR must be equal to OUT_W_PAR");
  static_assert(IN_CH % IN_CH_PAR == 0,
                "IN_CH must be a multiple of IN_CH_PAR");
  static_assert(IN_CH % OUT_CH_PAR == 0,
                "IN_CH must be a multiple of OUT_CH_PAR");

  BandwidthAdjustDecreaseChannels() : BandwidthAdjustDecreaseChannels(1) {}

  BandwidthAdjustDecreaseChannels(size_t pipeline_depth)
      : STEP_i_hw(0), STEP_i_ich_par(0), STEP_i_ch(0),
        STEP_pipeline_depth(pipeline_depth),
        STEP_actor_status(pipeline_depth, IN_HEIGHT * IN_WIDTH * IN_CH /
                                              (OUT_CH_PAR * IN_W_PAR)) {
    for (size_t i = 0; i < OUT_W_PAR; ++i) {
      STEP_delayed_output[i] =
          PipelineDelayBuffer<TOutputStruct>(pipeline_depth);
    }
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
           hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR]) {
    TInputStruct input_data[IN_W_PAR]; // Input structure to hold the data read.
    for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH; i_hw += IN_W_PAR) {
      for (size_t i_ch = 0; i_ch < IN_CH; i_ch += IN_CH_PAR) {
      BANDWIDTHADJUSTDECREASECHANNELS_RUN_LOOP:
        for (size_t i_ich_par = 0; i_ich_par < IN_CH_PAR;
             i_ich_par += OUT_CH_PAR) {
#pragma HLS pipeline II = 1
          BandwidthAdjustDecreaseChannels::pipeline_body(
              input_data_stream, output_data_stream, input_data, i_ich_par);
        }
      }
    }
  }

  ActorStatus step(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
              hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR]) {
    bool firing_condition = true;
    if (STEP_i_ich_par == 0) {
      for (size_t i_in_stream = 0; i_in_stream < IN_W_PAR; i_in_stream++) {
        if (input_data_stream[i_in_stream].empty()) {
            firing_condition = false;
        }
      }
    }

    if (firing_condition) {
        hls::stream<TOutputStruct> instant_output_stream[OUT_W_PAR];
      BandwidthAdjustDecreaseChannels::pipeline_body(
          input_data_stream, instant_output_stream, STEP_input_data,
          STEP_i_ich_par);

      STEP_i_ich_par += OUT_CH_PAR;
      if (STEP_i_ich_par >= IN_CH_PAR) {
        // If we have processed all input channels, reset the index and
        // increment the height/width index.
        STEP_i_ich_par = 0;
        STEP_i_ch += IN_CH_PAR;
      }
      if (STEP_i_ch >= IN_CH) {
        // If we have processed all output streams, reset the index and
        // increment the height/width index.
        STEP_i_ch = 0;
        STEP_i_hw += IN_W_PAR;
      }
      if (STEP_i_hw >= IN_HEIGHT * IN_WIDTH) {
        STEP_i_hw =
            0; // Reset height and width index if we have processed all data.
      }

        // Insert the firing status for the current step.
        STEP_actor_status.fire();

        // Add the output to the delayed output stream.
        for (size_t i = 0; i < OUT_W_PAR; ++i) {
          STEP_delayed_output[i].push(instant_output_stream[i].read(), true);
        }
    } else {
      // If no data is available, push empty outputs.
      for (size_t i = 0; i < OUT_W_PAR; ++i) {
        STEP_delayed_output[i].push(TOutputStruct(), false);
      }
    }

    // Advance the state of the actor firings.
    STEP_actor_status.advance();

    // Write the output data to the output stream.
    TOutputStruct out;
    for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++) {
      if (STEP_delayed_output[i_w_par].pop(out)) {
        output_data_stream[i_w_par].write(out);
      }
    }

    // Return the current actor status.
    return STEP_actor_status;
  }

private:
  TInputStruct
      STEP_input_data[IN_W_PAR]; // Output structure to hold the results.
  size_t STEP_i_hw;              // Index for height and width.
  size_t STEP_i_ich_par;         // Index for input channels.
  size_t STEP_i_ch;              // Index for input channels.
  size_t STEP_pipeline_depth;    // Pipeline depth for the actor.

  // CSDFG state variables
  ActorStatus STEP_actor_status;
  PipelineDelayBuffer<TOutputStruct>
      STEP_delayed_output[OUT_W_PAR]; // Delayed output buffers to maintain
                                      // pipeline depth for each parallel
                                      // output
  static void
  pipeline_body(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
                hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR],
                TInputStruct s_input_struct[IN_W_PAR], size_t i_ich_par) {
#pragma HLS inline
    TOutputStruct
        s_output_struct; // Output structure to read data from the input stream.
    Quantizer quantizer; // Quantizer instance for quantization.

    for (size_t i_w_par = 0; i_w_par < IN_W_PAR; i_w_par++) {
      // Read the input data structure from the input stream.
      if (i_ich_par == 0) {
        s_input_struct[i_w_par] = input_data_stream[i_w_par].read();
      }

      for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR; i_och_par++) {
        // Extract the data for the current pixel channel.
        TInput s_input_data = s_input_struct[i_w_par][i_ich_par + i_och_par];

        // Quantize the input data.
        TOutput s_output_data = quantizer(s_input_data);

        // Store the quantized data in the output structure.
        s_output_struct[i_och_par] = s_output_data;
      }

      output_data_stream[i_w_par].write(s_output_struct);
    }
  }
};