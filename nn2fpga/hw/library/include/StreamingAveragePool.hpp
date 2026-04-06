#pragma once
#include "DequantQuant.hpp"
#include "ap_int.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cassert>
#include <cstddef>

/**
 * @brief StreamingAveragePool implements a quantized average pooling with only
 * streaming in input and output. Works only with NHWC data layout.
 *
 * @tparam TInputWord      Data type for input word (packed input channels).
 * @tparam TInput          Data type for individual input elements.
 * @tparam OUT_CH          Number of output channels.
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

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, typename TAcc, typename TDiv,
          size_t OUT_CH, size_t OUT_HEIGHT, size_t OUT_WIDTH, size_t FH,
          size_t FW, size_t STRIDE_H, size_t STRIDE_W, size_t CH_PAR,
          size_t W_PAR>
class StreamingAveragePool {
  static constexpr size_t FW_EXPAND = FW + (W_PAR - 1) * STRIDE_W;

public:
  static_assert(OUT_HEIGHT > 0 && OUT_WIDTH > 0,
                "OUT_HEIGHT and OUT_WIDTH must be greater than 0");
  static_assert(W_PAR > 0, "W_PAR must be greater than 0");
  static_assert(CH_PAR > 0, "CH_PAR must be greater than 0");
  static_assert(FH > 0 && FW > 0, "FH and FW must be greater than 0");
  static_assert(STRIDE_H > 0 && STRIDE_W > 0, "STRIDE must be greater than 0");
  static_assert(CH_PAR > 0, "CH_PAR must be greater than 0");
  static_assert(W_PAR > 0, "W_PAR must be greater than 0");
  static_assert(OUT_CH % CH_PAR == 0, "OUT_CH must be a multiple of CH_PAR");
  static_assert(OUT_WIDTH % W_PAR == 0,
                "OUT_WIDTH must be a multiple of W_PAR");

  StreamingAveragePool() = default;

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
           hls::stream<TOutputWord> o_data[W_PAR]) {

    for (size_t i_hw = 0; i_hw < OUT_HEIGHT * OUT_WIDTH / W_PAR; i_hw++) {
    AVERAGEPOOL_RUN_LOOP:
      std::cout << "Processing (0, " << i_hw / (OUT_WIDTH / W_PAR) << ", " << i_hw % (OUT_WIDTH / W_PAR) << ")\n";
      for (size_t i_ch = 0; i_ch < OUT_CH; i_ch += CH_PAR) {
#pragma HLS pipeline II = 1
        StreamingAveragePool::pipeline_body(i_data, o_data);
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
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
      StreamingAveragePool::pipeline_body(i_data, instant_output_stream);

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
                            hls::stream<TOutputWord> o_data[W_PAR]) {
#pragma HLS inline

    // Input structure to hold the results.
    TAcc output_buffer[W_PAR][CH_PAR];
    // Input structure to hold the input data.
    TInputWord input_data[FH][FW_EXPAND];
    // Output structure to hold the output data.
    TOutputWord output_data[W_PAR];
    // Quantizer instance.
    Quantizer quantizer;

    // Read the input data for the current expanded window.
    for (size_t fh = 0; fh < FH; fh++) {
      for (size_t fw = 0; fw < FW_EXPAND; fw++) {
        input_data[fh][fw] = i_data[fh * FW_EXPAND + fw].read();
      }
    }

    // Initialize the output data to the minimum value.
    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      for (size_t i_ch_par = 0; i_ch_par < CH_PAR; i_ch_par++) {
        output_buffer[i_w_par][i_ch_par] = 0;
      }
    }

    for (size_t i_w_par = 0; i_w_par < W_PAR; i_w_par++) {
      for (size_t i_ch_par = 0; i_ch_par < CH_PAR; i_ch_par++) {
        for (size_t i_fh = 0; i_fh < FH; i_fh++) {
          for (size_t i_fw = 0; i_fw < FW; i_fw++) {

            // Compute the filter width index inside the expanded input window.
            size_t i_fw_expanded = i_fw + i_w_par * STRIDE_W;

            output_buffer[i_w_par][i_ch_par] +=
                input_data[i_fh][i_fw_expanded][i_ch_par];
          }
        }

        TDiv divisor = FH * FW;
        // Round the accumulated value to the nearest integer.
        // This is not strictly correct, as ties should be rounded to the
        // nearest even number, but it requires the use of a modulo operation,
        // which is quite expensive. Instead, we are rounding ties up.
        // TAcc bias = (output_buffer[i_w_par][i_ch_par] >= 0) ? (TAcc)(divisor >> 1)
        //                                          : (TAcc) - (divisor >> 1);
        // TAcc rounded_value = output_buffer[i_w_par][i_ch_par] + bias;
        // TAcc result = rounded_value / divisor; // Calculate the average.
        TAcc quotient = output_buffer[i_w_par][i_ch_par] / divisor;
        ap_int<TDiv::width + 1> remainder = output_buffer[i_w_par][i_ch_par] % divisor;
        ap_int<TDiv::width + 2> double_remainder = remainder * 2;
        if (double_remainder > divisor || (double_remainder == divisor && (quotient & 1))) {
          quotient += 1;
        }
        if (double_remainder < -divisor || (double_remainder == -divisor && (quotient & 1))) {
          quotient -= 1;
        }
        TAcc result = quotient;

        TOutput out_data = quantizer(result);
        output_data[i_w_par][i_ch_par] = out_data;
        std::cout << "Accumulator: " << output_buffer[i_w_par][i_ch_par]
                  << ", Result: " << result << ", Quantized: " << out_data
                  << std::endl;
            
        // Write the output data only after the computation of all
        // output channels for the current pixels.
        if (i_ch_par == CH_PAR - 1) {
          o_data[i_w_par].write(output_data[i_w_par]);
        }
      }
    }
  }
};