#pragma once
#include "hls_stream.h"
#include "ap_int.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>
#include <cassert>
#include <unordered_map>


template <typename TInputWordA, typename TInputA,
          typename TInputWordB, typename TInputB,
          typename TOutputWord, typename TOutput,
          typename TAcc, typename Quantizer,
          size_t IN_HEIGHT, size_t IN_WIDTH, size_t OUT_WIDTH,
          size_t IN_CH, size_t W_PAR, size_t CH_PAR>
class StreamingMatMul {

public:
    StreamingMatMul() = default;

   
    struct StepState {
        size_t i_r = 0;
        size_t i_j = 0;
        size_t i_k = 0;
        size_t i_ch = 0;

        TInputA  local_A[IN_CH][IN_WIDTH];
        TInputB  local_B[OUT_WIDTH][IN_CH][IN_WIDTH];
        TAcc     acc[IN_CH];

        ActorStatus actor_status{1, 1};
        bool initialized = false;

        void init() {
            if (initialized) return;
            // one firing per (r, j, k_step)
            actor_status = ActorStatus(
                1,
                IN_HEIGHT * OUT_WIDTH * (IN_WIDTH / W_PAR)*(IN_CH / CH_PAR));
            initialized = true;
        }
    };

    using Registry = std::unordered_map<const void *, StepState>;
    static Registry &registry() {
        static Registry r;
        return r;
    }

    void step_init() {
        registry()[this].init();
    }

    
    // run(): blocking, processes the full tensor
    
    template <size_t HLS_TAG>
    void run(hls::stream<TInputWordA> in_data_A[CH_PAR],
             hls::stream<TInputWordB> in_data_B[CH_PAR],
             hls::stream<TOutputWord>& mat_out) {
        #pragma HLS INLINE off

        TInputA local_A[IN_CH][IN_WIDTH];
        TInputB local_B[OUT_WIDTH][IN_CH][IN_WIDTH];
        TAcc    acc[IN_CH];

        #pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS ARRAY_PARTITION variable=local_A complete dim=1
        #pragma HLS ARRAY_PARTITION variable=local_A block factor=W_PAR dim=2
        #pragma HLS ARRAY_PARTITION variable=local_B complete dim=2
        #pragma HLS ARRAY_PARTITION variable=local_B complete dim=3

        MATMUL_RUN_LOOP:
        for (int r = 0; r < (int)IN_HEIGHT; r++) {
            for (int j = 0; j < (int)OUT_WIDTH; j++) {
                for (int k = 0; k < (int)IN_WIDTH; k += (int)W_PAR) {
                    for(int ch=0 ; ch<(int)IN_CH; ch+= (int)CH_PAR) {
                        #pragma HLS PIPELINE II=1
                    
                        StreamingMatMul::pipeline_body(
                            in_data_A, in_data_B,
                            local_A, local_B, acc,
                            mat_out, r, j, k, ch);
                        }
                
                }
            }
        }
    }

    ActorStatus step(hls::stream<TInputWordA> in_data_A[CH_PAR],
                     hls::stream<TInputWordB> in_data_B[CH_PAR],
                     hls::stream<TOutputWord>& mat_out) {
        auto it = registry().find(this);
        assert(it != registry().end() && "Instance not initialized");
        auto &st = it->second;

        int r = (int)st.i_r;
        int j = (int)st.i_j;
        int k = (int)st.i_k;
        int ch = (int)st.i_ch;

        // A is read when j==0, B is read when r==0
        bool firing_condition = true;
        if (j == 0 ) {
            for (size_t i_ch = 0; i_ch < CH_PAR; i_ch++) {
                if (in_data_A[i_ch].empty()) { firing_condition = false; break; }
            }
        }
        if (r == 0) {
            for (size_t i_ch = 0; i_ch < CH_PAR; i_ch++) {
                if (in_data_B[i_ch].empty()) { firing_condition = false; break; }
            }
        }

        if (firing_condition) {
            // Use a local stream to capture instant output
            hls::stream<TOutputWord> instant_out;
            StreamingMatMul::pipeline_body(
                in_data_A, in_data_B,
                st.local_A, st.local_B, st.acc,
                instant_out, r, j, k, ch);

            // Advance loop iterators: k → j → r → ch 
            st.i_ch += CH_PAR;
            if (st.i_ch >= IN_CH) {
                st.i_ch = 0;
                st.i_k += W_PAR;
            }
            
            if (st.i_k >= IN_WIDTH) {
                st.i_k = 0;
                st.i_j++;
            }
            if (st.i_j >= OUT_WIDTH) {
                st.i_j = 0;
                st.i_r++;
            }
            if (st.i_r >= IN_HEIGHT) {
                st.i_r = 0;
            }

            st.actor_status.fire();

            // Write directly to output — no delay buffer needed
            TOutputWord out_val;
            while (!instant_out.empty())
                mat_out.write(instant_out.read());
        }

        st.actor_status.advance();
        return st.actor_status;
    }

private:
    static void pipeline_body(
            hls::stream<TInputWordA>  in_data_A[CH_PAR],
            hls::stream<TInputWordB>  in_data_B[CH_PAR],
            TInputA  local_A[IN_CH][IN_WIDTH],
            TInputB  local_B[OUT_WIDTH][IN_CH][IN_WIDTH],
            TAcc     acc[IN_CH],
            hls::stream<TOutputWord>& mat_out,
            int r, int j, int k, int ch) {
        #pragma HLS inline

        Quantizer   q;
        TOutputWord pktOut_reg;
        TInputWordA pktA_reg;
        TInputWordB pktB_reg;

        
            //#pragma HLS UNROLL

            if (k == 0) {
                for (int i_ch = 0; i_ch < (int)CH_PAR; i_ch++)
                    acc[ch + i_ch] = 0;
            }

            // Load B once per r==0 for each ch group
            if (r == 0) {
                for (int i_ch = 0; i_ch < (int)CH_PAR; i_ch++) {
                    pktB_reg = in_data_B[i_ch].read();
                    for (int i_w = 0; i_w < (int)W_PAR; i_w++) {
                        #pragma HLS UNROLL
                        local_B[j][ch + i_ch][k + i_w] = (TInputB)pktB_reg[i_w];
                    }
                }
            }

            // Load A once per j==0 for each ch group
            if (j == 0) {
                for (int i_ch = 0; i_ch < (int)CH_PAR; i_ch++) {
                    pktA_reg = in_data_A[i_ch].read();
                    for (int i_w = 0; i_w < (int)W_PAR; i_w++) {
                        #pragma HLS UNROLL
                        local_A[ch+ i_ch][k + i_w] = (TInputA)pktA_reg[i_w];
                    }
                }
            }

            // Accumulate
            for (int i_acc = 0; i_acc < (int)W_PAR; i_acc++) {
                for (int i_ch = 0; i_ch < (int)CH_PAR; i_ch++) {
                    acc[ch + i_ch] += (TAcc)local_A[ch + i_ch][k + i_acc] *
                                 (TAcc)local_B[j][ch + i_ch][k + i_acc];
                }
            }

            // Write output packet for this ch group when dot-product complete
            if (k == (int)IN_WIDTH - (int)W_PAR) {
                for (int i_ch = 0; i_ch < (int)CH_PAR; i_ch++) {
                    constexpr int OBW = TOutput::width;
                    pktOut_reg.data.range((i_ch * OBW) + 7, i_ch * OBW) = q(acc[ch + i_ch]);
                }
                pktOut_reg.keep = -1;
                pktOut_reg.last = (ch + (int)CH_PAR == (int)IN_CH &&
                                   r == (int)IN_HEIGHT - 1 &&
                                   j == (int)OUT_WIDTH  - 1);
                mat_out.write(pktOut_reg);
            }
        }
    
};
