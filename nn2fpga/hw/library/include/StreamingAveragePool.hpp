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
 * @tparam TInputWord      Data type for input word.
 * @tparam TInput          Data type for individual input elements.
 * @tparam OUT_DIM0        Dimension 0 of the output feature map.
 * @tparam OUT_DIM1        Dimension 1 of the output feature map.
 * @tparam OUT_DIM2        Dimension 2 of the output feature map.
 * @tparam FH              Filter height.
 * @tparam FW              Filter width.
 * @tparam STRIDE_DIM0     Stride along dimension 0.
 * @tparam STRIDE_DIM1     Stride along dimension 1.
 * @tparam DIM1_UNROLL     Parallelism factor for dimension 1.
 * @tparam DIM2_UNROLL     Parallelism factor for dimension 2.
 *
 * @note
 * - The class provides two main interfaces:
 *   - run(): Processes the entire convolution in a blocking fashion.
 *   - step(): Processes one pipeline step, suitable for CSDFG (Cyclo-Static
 * Data Flow Graph) scheduling.
 *
 * @section Implementation Details
 * - Input data are packed into words of DIM2_UNROLL data elements. The window is
 * expanded to account for the stride and the width parallelism factor
 * DIM1_UNROLL, such that no data is duplicated. The input stream is an array of
 * FH*(FW+(DIM1_UNROLL-1)*STRIDE_DIM1) streams, each providing DIM2_UNROLL input
 * data elements. The input window is completely reused.
 * - Accumulators and input buffers are partitioned for parallel access.
 *
 */

template <typename TInputWord, typename TInput, typename TOutputWord,
          typename TOutput, typename Quantizer, typename TAcc, typename TDiv,
          size_t OUT_DIM0, size_t OUT_DIM1, size_t OUT_DIM2, size_t FH,
          size_t FW, size_t STRIDE_DIM0, size_t STRIDE_DIM1, size_t DIM1_UNROLL,
          size_t DIM2_UNROLL>
class StreamingAveragePool {
  static constexpr size_t FW_EXPAND = FW + (DIM1_UNROLL - 1) * STRIDE_DIM1;

  static_assert(OUT_DIM0 > 0 && OUT_DIM1 > 0,
                "OUT_DIM0 and OUT_DIM1 must be greater than 0");
  static_assert(DIM1_UNROLL > 0, "DIM1_UNROLL must be greater than 0");
  static_assert(DIM2_UNROLL > 0, "DIM2_UNROLL must be greater than 0");
  static_assert(FH > 0 && FW > 0, "FH and FW must be greater than 0");
  static_assert(STRIDE_DIM0 > 0 && STRIDE_DIM1 > 0,
                "STRIDE must be greater than 0");
  static_assert(DIM2_UNROLL > 0, "DIM2_UNROLL must be greater than 0");
  static_assert(DIM1_UNROLL > 0, "DIM1_UNROLL must be greater than 0");
  static_assert(OUT_DIM2 % DIM2_UNROLL == 0,
                "OUT_DIM2 must be a multiple of DIM2_UNROLL");
  static_assert(OUT_DIM1 % DIM1_UNROLL == 0,
                "OUT_DIM1 must be a multiple of DIM1_UNROLL");

public:
  StreamingAveragePool() = default;

  void step_init(size_t pipeline_depth = 1) {
    auto &st = registry()[this];
    st.init(pipeline_depth);
  }

  template <size_t HLS_TAG>
  void run(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
           hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {

    for (size_t i_dim01 = 0; i_dim01 < OUT_DIM0 * OUT_DIM1 / DIM1_UNROLL;
         i_dim01++) {
    AVERAGEPOOL_RUN_LOOP:
      for (size_t i_dim2 = 0; i_dim2 < OUT_DIM2; i_dim2 += DIM2_UNROLL) {
#pragma HLS pipeline II = 1
        StreamingAveragePool::pipeline_body(i_data, o_data);
      }
    }
  }

  ActorStatus step(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
                   hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
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

      hls::stream<TOutputWord> instant_output_stream[DIM1_UNROLL];
      StreamingAveragePool::pipeline_body(i_data, instant_output_stream);

      st.i_dim2 += DIM2_UNROLL;
      if (st.i_dim2 >= OUT_DIM2) {
        // If we have processed all output channels, reset the index and
        // increment the input channels index.
        st.i_dim2 = 0;
        st.i_dim01++;
      }
      if (st.i_dim01 >= OUT_DIM0 * OUT_DIM1 / DIM1_UNROLL) {
        st.i_dim01 = 0;
      }

      // Insert the firing status for the current step.
      st.actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; ++i_dim1_par) {
        if (!instant_output_stream[i_dim1_par].empty()) {
          st.delayed_output[i_dim1_par].push(
              instant_output_stream[i_dim1_par].read(), true);
        } else {
          // If the output stream is empty, push a placeholder.
          st.delayed_output[i_dim1_par].push(TOutputWord(), false);
        }
      }
    } else {
      // If no data is available, push empty outputs.
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
  struct StepState {
    // Loop iteration indexes.
    size_t i_dim01 = 0, i_dim2 = 0;

    PipelineDelayBuffer<TOutputWord> delayed_output[DIM1_UNROLL];
    ActorStatus actor_status{1, 1};
    bool initialized = false;

    void init(size_t depth) {
      if (initialized)
        return;
      for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
        delayed_output[i_dim1_par] = PipelineDelayBuffer<TOutputWord>(depth);
      }
      actor_status = ActorStatus(depth, OUT_DIM0 * (OUT_DIM1 / DIM1_UNROLL) *
                                            (OUT_DIM2 / DIM2_UNROLL));
      initialized = true;
    }
  };

  using Registry = std::unordered_map<const void *, StepState>;
  static Registry &registry() {
    static Registry r;
    return r;
  }

  static void pipeline_body(hls::stream<TInputWord> i_data[FH * FW_EXPAND],
                            hls::stream<TOutputWord> o_data[DIM1_UNROLL]) {
#pragma HLS inline

    // Input structure to hold the results.
    TAcc output_buffer[DIM1_UNROLL][DIM2_UNROLL];
    // Input structure to hold the input data.
    TInputWord input_data[FH][FW_EXPAND];
    // Output structure to hold the output data.
    TOutputWord output_data[DIM1_UNROLL];
    // Quantizer instance.
    Quantizer quantizer;

    // Read the input data for the current expanded window.
    for (size_t fh = 0; fh < FH; fh++) {
      for (size_t fw = 0; fw < FW_EXPAND; fw++) {
        input_data[fh][fw] = i_data[fh * FW_EXPAND + fw].read();
      }
    }

    // Initialize the output data to the minimum value.
    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL; i_dim2_par++) {
        output_buffer[i_dim1_par][i_dim2_par] = 0;
      }
    }

    for (size_t i_dim1_par = 0; i_dim1_par < DIM1_UNROLL; i_dim1_par++) {
      for (size_t i_dim2_par = 0; i_dim2_par < DIM2_UNROLL; i_dim2_par++) {
        for (size_t i_fh = 0; i_fh < FH; i_fh++) {
          for (size_t i_fw = 0; i_fw < FW; i_fw++) {

            // Compute the filter width index inside the expanded input window.
            size_t i_fw_expanded = i_fw + i_dim1_par * STRIDE_DIM1;

            output_buffer[i_dim1_par][i_dim2_par] +=
                input_data[i_fh][i_fw_expanded][i_dim2_par];
          }
        }

        TDiv divisor = FH * FW;
        // Round the accumulated value to the nearest integer.
        // This is not strictly correct, as ties should be rounded to the
        // nearest even number, but it requires the use of a modulo operation,
        // which is quite expensive. Instead, we are rounding ties up.
        // TAcc bias = (output_buffer[i_dim1_par][i_dim2_par] >= 0) ?
        // (TAcc)(divisor >> 1)
        //                                          : (TAcc) - (divisor >> 1);
        // TAcc rounded_value = output_buffer[i_dim1_par][i_dim2_par] + bias;
        // TAcc result = rounded_value / divisor; // Calculate the average.
        TAcc quotient = output_buffer[i_dim1_par][i_dim2_par] / divisor;
        ap_int<TDiv::width + 1> remainder =
            output_buffer[i_dim1_par][i_dim2_par] % divisor;
        ap_int<TDiv::width + 2> double_remainder = remainder * 2;
        if (double_remainder > divisor ||
            (double_remainder == divisor && (quotient & 1))) {
          quotient += 1;
        }
        if (double_remainder < -divisor ||
            (double_remainder == -divisor && (quotient & 1))) {
          quotient -= 1;
        }
        TAcc result = quotient;

        TOutput out_data = quantizer(result);
        output_data[i_dim1_par][i_dim2_par] = out_data;
        std::cout << "Accumulator: " << output_buffer[i_dim1_par][i_dim2_par]
                  << ", Result: " << result << ", Quantized: " << out_data
                  << std::endl;

        // Write the output data only after the computation of all
        // output channels for the current pixels.
        if (i_dim2_par == DIM2_UNROLL - 1) {
          o_data[i_dim1_par].write(output_data[i_dim1_par]);
        }
      }
    }
  }
};