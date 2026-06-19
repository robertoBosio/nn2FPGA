#include "StreamingConstMul.hpp"
#include "StreamingSoftmax.hpp"
#include "BandwidthAdjust.hpp"
#include "StreamingTensorDuplicator.hpp"
#include "YoloAttention/QKMatMul.hpp"
#include "YoloAttention/SplitReshapeQKV.hpp"
#include "YoloAttention/ReshapeV.hpp"
#include "YoloAttention/VPMatMul.hpp"
#include "YoloAttention/Transpose.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include "DequantQuant.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

void wrap_run(hls::stream<test_config::TInputWord> input_data[1],
              hls::stream<test_config::TSplitWord> o_v_data[1],
              hls::stream<test_config::TYOutputWord> o_matmul_data[1]) {
#pragma HLS dataflow

  hls::stream<test_config::TSplitWord> stream_q[2];
  hls::stream<test_config::TSplitWord> stream_k[2];
  hls::stream<test_config::TSplitWord> stream_v[2];
  hls::stream<test_config::TSplitWord> stream_v_copy[2];
  hls::stream<test_config::TQPackedWord> stream_q_packed[2];
  hls::stream<test_config::TKPackedWord> stream_k_packed[2];
  hls::stream<test_config::TQKWord> stream_qk[2];
  hls::stream<test_config::TQKScaledWord> stream_qkscaled[2];
  hls::stream<test_config::TSoftmaxWord> stream_softmax[2];
  hls::stream<test_config::TVPackedWord> stream_v_transposed_packed[2];
  hls::stream<test_config::TSplitWord> stream_v_parallel_head0[test_config::VP_REDUCE_PAR];
  hls::stream<test_config::TSplitWord> stream_v_parallel_head1[test_config::VP_REDUCE_PAR];
  hls::stream<test_config::TPPackedWord> stream_softmax_packed[2];
  hls::stream<test_config::TYOutputWord> stream_attention_out[2];
  hls::stream<test_config::TSplitWord> stream_v_out[2];

  SplitReshapeQKV<
      test_config::TInputWord,
      test_config::TInput,
      test_config::TSplitWord,
      test_config::TSplit,
      test_config::SplitQuantizer,
      test_config::IN_HEIGHT,
      test_config::IN_WIDTH,
      test_config::IN_CH,
      test_config::REDUCE_PAR>
      split_reshape;
  split_reshape.run<0>(input_data, stream_q, stream_k, stream_v);

  StreamingTensorDuplicator<
      test_config::TSplitWord,
      test_config::DIM_V,
      test_config::DIM_SEQ_VP,
      1,
      test_config::W_PAR,
      test_config::REDUCE_PAR>
      tensor_duplicator_head0;
  tensor_duplicator_head0.run<1>(&stream_v[0], &stream_v_out[0], &stream_v_copy[0]);
  
  StreamingTensorDuplicator<
      test_config::TSplitWord,
      test_config::DIM_V,
      test_config::DIM_SEQ_VP,
      1,
      test_config::W_PAR,
      test_config::REDUCE_PAR>
      tensor_duplicator_head1;
  tensor_duplicator_head1.run<2>(&stream_v[1], &stream_v_out[1], &stream_v_copy[1]);

  BandwidthAdjustIncreaseWord<
      test_config::TSplitWord,
      test_config::TSplit,
      test_config::TQPackedWord,
      test_config::TSplit,
      DequantQuantEqual<test_config::TSplit>,
      1,
      test_config::DIM_Q,
      test_config::DIM_SEQ_QK,
      1,
      1,
      1,
      test_config::QK_REDUCE_PAR>
      pack_q_head0;
  pack_q_head0.run<3>(&stream_q[0], &stream_q_packed[0]);

  BandwidthAdjustIncreaseWord<
      test_config::TSplitWord,
      test_config::TSplit,
      test_config::TQPackedWord,
      test_config::TSplit,
      DequantQuantEqual<test_config::TSplit>,
      1,
      test_config::DIM_Q,
      test_config::DIM_SEQ_QK,
      1,
      1,
      1,
      test_config::QK_REDUCE_PAR>
      pack_q_head1;
  pack_q_head1.run<4>(&stream_q[1], &stream_q_packed[1]);

  BandwidthAdjustIncreaseWord<
      test_config::TSplitWord,
      test_config::TSplit,
      test_config::TKPackedWord,
      test_config::TSplit,
      DequantQuantEqual<test_config::TSplit>,
      1,
      test_config::DIM_K,
      test_config::DIM_SEQ_QK,
      1,
      1,
      1,
      test_config::QK_REDUCE_PAR>
      pack_k_head0;
  pack_k_head0.run<5>(&stream_k[0], &stream_k_packed[0]);

  BandwidthAdjustIncreaseWord<
      test_config::TSplitWord,
      test_config::TSplit,
      test_config::TKPackedWord,
      test_config::TSplit,
      DequantQuantEqual<test_config::TSplit>,
      1,
      test_config::DIM_K,
      test_config::DIM_SEQ_QK,
      1,
      1,
      1,
      test_config::QK_REDUCE_PAR>
      pack_k_head1;
  pack_k_head1.run<6>(&stream_k[1], &stream_k_packed[1]);

  QKMatMul<
      test_config::TQPackedWord,
      test_config::TSplit,
      test_config::TKPackedWord,
      test_config::TSplit,
      test_config::TQKWord,
      test_config::TQK,
      test_config::TAccQK,
      test_config::QKQuantizer,
      1,
      test_config::DIM_Q,
      test_config::DIM_K,
      test_config::DIM_SEQ_QK,
      test_config::QK_REDUCE_PAR>
      matmulqk_head0;
  matmulqk_head0.run<7>(&stream_q_packed[0], &stream_k_packed[0], &stream_qk[0]);
  
  QKMatMul<
      test_config::TQPackedWord,
      test_config::TSplit,
      test_config::TKPackedWord,
      test_config::TSplit,
      test_config::TQKWord,
      test_config::TQK,
      test_config::TAccQK,
      test_config::QKQuantizer,
      1,
      test_config::DIM_Q,
      test_config::DIM_K,
      test_config::DIM_SEQ_QK,
      test_config::QK_REDUCE_PAR>
      matmulqk_head1;
  matmulqk_head1.run<8>(&stream_q_packed[1], &stream_k_packed[1], &stream_qk[1]);

  StreamingConstMul<
      test_config::TQKWord,
      test_config::TQK,
      test_config::TConst,
      test_config::TQKScaledWord,
      test_config::TQKScaled,
      test_config::TMul,
      test_config::MulActivation,
      test_config::MulQuantizer,
      test_config::MUL_HEIGHT,
      test_config::MUL_WIDTH,
      1,
      test_config::MUL_W_PAR,
      test_config::MUL_CH_PAR>
      mul_head0;
  mul_head0.run<9>(&stream_qk[0], test_config::constant_scaler, &stream_qkscaled[0]);

  StreamingConstMul<
      test_config::TQKWord,
      test_config::TQK,
      test_config::TConst,
      test_config::TQKScaledWord,
      test_config::TQKScaled,
      test_config::TMul,
      test_config::MulActivation,
      test_config::MulQuantizer,
      test_config::MUL_HEIGHT,
      test_config::MUL_WIDTH,
      1,
      test_config::MUL_W_PAR,
      test_config::MUL_CH_PAR>
      mul_head1;
  mul_head1.run<10>(&stream_qk[1], test_config::constant_scaler, &stream_qkscaled[1]);

  StreamingSoftmax<
      test_config::TQKScaledWord,
      test_config::TQKScaled,
      test_config::TSoftmaxWord,
      test_config::TSoftmax,
      test_config::TLUT,
      test_config::TAccSoftmax,
      test_config::TDivSoftmax,
      test_config::SoftmaxQuantizer,
      test_config::LUT_SIZE,
      test_config::MUL_HEIGHT,
      test_config::MUL_WIDTH,
      test_config::MUL_W_PAR,
      test_config::MUL_CH_PAR>
      streaming_softmax_head0;
  streaming_softmax_head0.run<11>(&stream_qkscaled[0], test_config::LUTmem, &stream_softmax[0]);

  StreamingSoftmax<
      test_config::TQKScaledWord,
      test_config::TQKScaled,
      test_config::TSoftmaxWord,
      test_config::TSoftmax,
      test_config::TLUT,
      test_config::TAccSoftmax,
      test_config::TDivSoftmax,
      test_config::SoftmaxQuantizer,
      test_config::LUT_SIZE,
      test_config::MUL_HEIGHT,
      test_config::MUL_WIDTH,
      test_config::MUL_W_PAR,
      test_config::MUL_CH_PAR>
      streaming_softmax_head1;
  streaming_softmax_head1.run<12>(&stream_qkscaled[1], test_config::LUTmem, &stream_softmax[1]);

  BandwidthAdjustIncreaseStreams<
      test_config::TSplitWord,
      test_config::TSplit,
      test_config::TSplitWord,
      test_config::TSplit,
      DequantQuantEqual<test_config::TSplit>,
      1,
      test_config::DIM_SEQ_VP,
      test_config::DIM_V,
      1,
      test_config::VP_REDUCE_PAR,
      1,
      1>
      stream_expand_v_head0;
  stream_expand_v_head0.run<13>(&stream_v_copy[0], stream_v_parallel_head0);

  TransposeRowColReplayPacked<
      test_config::TSplitWord,
      test_config::TVPackedWord,
      test_config::DIM_V,
      test_config::DIM_SEQ_VP,
      test_config::DIM_P,
      test_config::VP_REDUCE_PAR>
      transpose_v_head0;
  transpose_v_head0.run<14>(stream_v_parallel_head0, &stream_v_transposed_packed[0]);
  
  BandwidthAdjustIncreaseStreams<
      test_config::TSplitWord,
      test_config::TSplit,
      test_config::TSplitWord,
      test_config::TSplit,
      DequantQuantEqual<test_config::TSplit>,
      1,
      test_config::DIM_SEQ_VP,
      test_config::DIM_V,
      1,
      test_config::VP_REDUCE_PAR,
      1,
      1>
      stream_expand_v_head1;
  stream_expand_v_head1.run<15>(&stream_v_copy[1], stream_v_parallel_head1);

  TransposeRowColReplayPacked<
      test_config::TSplitWord,
      test_config::TVPackedWord,
      test_config::DIM_V,
      test_config::DIM_SEQ_VP,
      test_config::DIM_P,
      test_config::VP_REDUCE_PAR>
      transpose_v_head1;
  transpose_v_head1.run<16>(stream_v_parallel_head1, &stream_v_transposed_packed[1]);

  BandwidthAdjustIncreaseWord<
      test_config::TSoftmaxWord,
      test_config::TSoftmax,
      test_config::TPPackedWord,
      test_config::TSoftmax,
      DequantQuantEqual<test_config::TSoftmax>,
      1,
      test_config::DIM_P,
      test_config::DIM_SEQ_VP,
      1,
      1,
      1,
      test_config::VP_REDUCE_PAR>
      pack_softmax_head0;
  pack_softmax_head0.run<17>(&stream_softmax[0], &stream_softmax_packed[0]);

  BandwidthAdjustIncreaseWord<
      test_config::TSoftmaxWord,
      test_config::TSoftmax,
      test_config::TPPackedWord,
      test_config::TSoftmax,
      DequantQuantEqual<test_config::TSoftmax>,
      1,
      test_config::DIM_P,
      test_config::DIM_SEQ_VP,
      1,
      1,
      1,
      test_config::VP_REDUCE_PAR>
      pack_softmax_head1;
  pack_softmax_head1.run<18>(&stream_softmax[1], &stream_softmax_packed[1]);

  VPMatMul<
      test_config::TVPackedWord,
      test_config::TSplit,
      test_config::TPPackedWord,
      test_config::TSoftmax,
      test_config::TYOutputWord,
      test_config::TYOutput,
      test_config::TAccVP,
      test_config::VPQuantizer,
      1,
      test_config::DIM_V,
      test_config::DIM_P,
      test_config::DIM_SEQ_VP,
      test_config::VP_REDUCE_PAR>
      matmulvp_head0;
  matmulvp_head0.run<19>(&stream_v_transposed_packed[0], &stream_softmax_packed[0], &stream_attention_out[0]);

  VPMatMul<
      test_config::TVPackedWord,
      test_config::TSplit,
      test_config::TPPackedWord,
      test_config::TSoftmax,
      test_config::TYOutputWord,
      test_config::TYOutput,
      test_config::TAccVP,
      test_config::VPQuantizer,
      1,
      test_config::DIM_V,
      test_config::DIM_P,
      test_config::DIM_SEQ_VP,
      test_config::VP_REDUCE_PAR>
      matmulvp_head1;
  matmulvp_head1.run<20>(&stream_v_transposed_packed[1], &stream_softmax_packed[1], &stream_attention_out[1]);

  ReshapeV<test_config::TSplitWord, test_config::TSplit,
           test_config::TSplitWord, test_config::TSplit,
           DequantQuantEqual<test_config::TSplit>, 2, 64, 400, 20, 20, 128,
           test_config::REDUCE_PAR>
      reshape_v;
  reshape_v.run<21>(stream_v_out, o_v_data);

  ReshapeV<test_config::TYOutputWord, test_config::TYOutput,
           test_config::TYOutputWord, test_config::TYOutput,
           DequantQuantEqual<test_config::TYOutput>, 2, 64, 400, 20, 20, 128,
           test_config::REDUCE_PAR>
      reshape_y;
  reshape_y.run<22>(stream_attention_out, o_matmul_data);
}

bool test_run() {
  hls::stream<test_config::TInputWord> input_data[1];
  hls::stream<test_config::TSplitWord> o_v_data[1];
  hls::stream<test_config::TYOutputWord> o_matmul_data[1];

  // Streaming input tensor.
  for (size_t i_h = 0; i_h < test_config::IN_HEIGHT; i_h++) {
    for (size_t i_w = 0; i_w < test_config::IN_WIDTH; i_w++) {
      for (size_t i_ch = 0; i_ch < test_config::IN_CH; i_ch+= test_config::CH_PAR) {
        test_config::TInputWord input_word;
        for (size_t ch_par = 0; ch_par < test_config::CH_PAR; ch_par++) {
          input_word[ch_par] =
              test_config::input_tensor[0][i_ch + ch_par][i_h][i_w];
        }
        input_data[0].write(input_word);
      }
    }
  }

  // Run the operator
  wrap_run(input_data, o_v_data, o_matmul_data);

  // Check output
  bool flag = true;

  for (size_t i_h = 0; i_h < test_config::OUT_HEIGHT; i_h++) {
    for (size_t i_w = 0; i_w < test_config::OUT_WIDTH; i_w++) {
      for (size_t i_ch = 0; i_ch < test_config::OUT_CH;
           i_ch += test_config::CH_PAR) {
        test_config::TYOutputWord data = o_matmul_data[0].read();
        bool cmp;
        for (size_t ch_par = 0; ch_par < test_config::CH_PAR; ch_par++) {
          cmp = (data[ch_par] ==
                 test_config::output_tensor_y[0][i_ch + ch_par][i_h][i_w]);
          if (!cmp) {
            std::cout << "Mismatch at index (i_h=" << i_h << ", i_w=" << i_w << ", ch=" << i_ch + ch_par
                      << "): " << data[ch_par]
                      << " != " << test_config::output_tensor_y[0][i_ch + ch_par][i_h][i_w]
                      << std::endl;
          }
          flag &= cmp;
        }
      }
    }
  }

  // Check v tensor output from split-reshape.
  std::cout << "Checking V tensor output from split-reshape..." << std::endl;
  for (size_t i_h = 0; i_h < test_config::OUT_HEIGHT; i_h++) {
    for (size_t i_w = 0; i_w < test_config::OUT_WIDTH; i_w++) {
      for (size_t i_ch = 0; i_ch < test_config::OUT_CH;
           i_ch += test_config::CH_PAR) {
        test_config::TSplitWord data = o_v_data[0].read();
        bool cmp;
        for (size_t ch_par = 0; ch_par < test_config::CH_PAR; ch_par++) {
          cmp = (data[ch_par] ==
                 test_config::output_tensor_v[0][i_ch + ch_par][i_h][i_w]);
          if (!cmp) {
            std::cout << "Mismatch at index (i_h=" << i_h << ", i_w=" << i_w << ", ch=" << i_ch + ch_par
                      << "): " << data[ch_par]
                      << " != " << test_config::output_tensor_v[0][i_ch + ch_par][i_h][i_w]
                      << std::endl;
          }
          flag &= cmp;
        }
      }
    }
  }

  return flag;
}

int main(int argc, char **argv) {

  bool all_passed = true;

  all_passed &= test_run();

  // Testing the pipeline with csim only, as it is only relevant for fifo depth
  // estimations
  // if (argc > 1 && std::string(argv[1]) == "csim") {
  //   all_passed &= test_step();
  // }

  if (!all_passed) {
    std::cout << "Failed." << std::endl;
  } else {
    std::cout << "Passed." << std::endl;
  }

  return all_passed ? 0 : 1;
}
