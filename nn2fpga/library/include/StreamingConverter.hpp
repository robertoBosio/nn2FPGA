#pragma once
#include "hls_stream.h"
#include "ap_int.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>
#include <cassert>
#include <unordered_map>

template <typename TInputWord, typename TInput,
          typename TOutputWord, typename TOutput,
          size_t IN_HEIGHT, size_t IN_WIDTH,
          size_t IN_CH, size_t W_PAR, size_t CH_PAR>

class StreamingConverter {
public:
    StreamingConverter() = default;
    struct StepState {
        size_t i_k = 0;
        size_t i_w = 0;
        size_t i_ch = 0;
        size_t i_pass = 0;
        
        
        ActorStatus actor_status{1, 1};
        bool initialized = false;

        void init() {
            if (initialized) return;
            // one firing per (r, j, k_step)
            actor_status = ActorStatus(
                1,
                ((IN_CH/CH_PAR) * (IN_WIDTH/W_PAR) * IN_HEIGHT)); 
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

    template <size_t HLS_TAG>
    void run(hls::stream<TInputWord> in_data[W_PAR],
             hls::stream<TOutputWord> out_data[CH_PAR]) {
        #pragma HLS INLINE off
    
        for (int k = 0; k < (int)IN_HEIGHT; k++) {
            for (int w = 0; w < (int)(IN_WIDTH / W_PAR); w++) {
                for (int ch = 0; ch< (int)(IN_CH / CH_PAR); ch++) {
                        #pragma HLS PIPELINE II=1
                        pipeline_body(in_data, out_data,k, w, ch);
                }
            }
            
        }
        
    }
    ActorStatus step(hls::stream<TInputWord> in_data[W_PAR],
                     hls::stream<TOutputWord> out_data[CH_PAR]) {
        auto it = registry().find(this);
        assert(it != registry().end() && "Instance not initialized");
        auto &st = it->second;
        int k    = (int)st.i_k;
        int w    = (int)st.i_w;
        int ch   = (int)st.i_ch;
        int pass = (int)st.i_pass;

    
        // ── Firing condition ──────────────────────────────────────────
    
        bool firing_condition = true;
       
           
        for (int w_i = 0; w_i < (int)W_PAR; w_i++) {
            if (in_data[w_i].empty()) {
                firing_condition = false;
                break;
            }
        }
        
        
        if (firing_condition) {
            
            //hls::stream<TOutputWord> instant_out_data[CH_PAR];
            StreamingConverter::pipeline_body(
                in_data, out_data, k, w, ch,pass);
                

            // Advance loop iterators: k → j → r → ch 
            st.i_ch ++;
            if (st.i_ch >= (int)(IN_CH/CH_PAR)) {
                st.i_ch = 0;
                st.i_w++;
            }
            
            if (st.i_w >= (int)(IN_WIDTH/W_PAR)){
                st.i_w = 0;
                st.i_k++;
            }
            if (st.i_k >=  (int)IN_HEIGHT) {
                st.i_k = 0;
                st.i_pass++;
            }
    

            st.actor_status.fire();

            // Write directly to output — no delay buffer needed
            //?
            
        }
                

        st.actor_status.advance();
        return st.actor_status;
    }

    private:
    static void pipeline_body(
        hls::stream<TInputWord> in_data[W_PAR],
        hls::stream<TOutputWord> out_data[CH_PAR],
        int k, int w, int ch, int pass) {
        #pragma HLS INLINE
        
        TOutputWord word_ch[CH_PAR];
        #pragma HLS ARRAY_PARTITION variable=word_ch complete

        for (int w_i = 0; w_i < (int)W_PAR; w_i++) {
            #pragma HLS UNROLL
            TInputWord pkt = in_data[w_i].read();
            for (int ch_i = 0; ch_i < (int)CH_PAR; ch_i++) {
                #pragma HLS UNROLL
                word_ch[ch_i][w_i] = pkt[ch_i];
                    
            }
        
        }

        for (int ch_i = 0; ch_i < (int)CH_PAR; ch_i++) {
            #pragma HLS UNROLL
            out_data[ch_i].write(word_ch[ch_i]);
        }
    }
};