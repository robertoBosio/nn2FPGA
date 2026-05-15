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
          size_t IN_CH,typename INDEX_T>
          
class StreamingTranspose {

public:
    StreamingTranspose() = default;

     struct StepState {
        size_t i_k = 0;
        size_t i_w = 0;
        size_t i_ch = 0;
        size_t i_pass = 0;

        
        TInput  buf[IN_CH * IN_WIDTH * IN_HEIGHT];
        
        ActorStatus actor_status{1, 1};
        bool initialized = false;

        void init() {
            if (initialized) return;
            // one firing per (r, j, k_step)
            actor_status = ActorStatus(
                1,
                2 * (IN_CH * IN_WIDTH * IN_HEIGHT)); // 2 comes from the fact that we have 2 passes (write and read) with same number of iterations
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
    void run(hls::stream<TInputWord> in_data[1],
             hls::stream<TOutputWord> out_data[1]) {
        #pragma HLS INLINE off
        TInputB buf[IN_CH * IN_WIDTH * IN_HEIGHT];

        for (int pass = 0; pass < 2; pass++) {
            for (int k = 0; k < (int)IN_HEIGHT; k++) {
                for (int w = 0; w < (int)IN_WIDTH; w++) {
                    for (int ch = 0; ch < (int)IN_CH; ch++) {
                        #pragma HLS PIPELINE II=1
                        pipeline_body(in_data, out_data, buf, k, w, ch, pass);
                    }
                }
            }
        }
    }
     ActorStatus step(hls::stream<TInputWord> in_data[1],
                     hls::stream<TOutputWord> out_data[1]) {
        auto it = registry().find(this);
        assert(it != registry().end() && "Instance not initialized");
        auto &st = it->second;
        int pass = (int)st.i_pass;
        int k    = (int)st.i_k;
        int w    = (int)st.i_w;
        int ch   = (int)st.i_ch;

    
        // ── Firing condition ──────────────────────────────────────────
    // WRITE pass (pass==0): needs input from in_data_B
    // READ  pass (pass==1): reads only from buf → no stream needed
        bool firing_condition = true;
       if (pass == 0) {
           
            if (in_data[0].empty()) { firing_condition = false; break; }
            
        }
        // pass==1: firing_condition stays true — buf is always ready
       
        

        if (firing_condition) {
            
            hls::stream<TOutputWord> out_word;
            StreamingTranspose::pipeline_body(
                in_data, out_word, st.buf, k, w, ch, pass);
                

            // Advance loop iterators: k → j → r → ch 
            st.i_ch ++;
            if (st.i_ch >= (int)IN_CH) {
                st.i_ch = 0;
                st.i_w++;
            }
            
            if (st.i_w >= (int)IN_WIDTH){
                st.i_w = 0;
                st.i_k++;
            }
            if (st.i_k >=  (int)IN_HEIGHT) {
                st.i_k = 0;
                st.i_pass++;
            }
            if (st.i_pass >= 2) {
                st.i_pass = 0;
            }

            st.actor_status.fire();

            // Write directly to output — no delay buffer needed
             if (pass == 1) {
                out_data[0].write(out_word);
            }
             /*
            while (!instant_out_arr[0].empty())
                out_data_B[0].write(instant_out_arr[0].read());
            */
        }

        st.actor_status.advance();
        return st.actor_status;
    }

private:
    static void pipeline_body(
                    hls::stream<TInputWord>  in_data[1],
                    hls::stream<TOutputWord>  out_data[1],
                    TInput  buf[IN_CH * IN_WIDTH * IN_HEIGHT],
                    int k, int w, int ch, int pass) {
        #pragma HLS inline
            

        //consider matrix[ch][w][k] as the original layout, and matrix[ch][k][w] as the transposed layout
        if (pass == 0) {
            INDEX_T idxRead = ch * IN_WIDTH * IN_HEIGHT + w * IN_HEIGHT + k;
            TInputWordB pkt = in_data[0].read();
            buf[idxRead] = pkt[0];
        } else {
            INDEX_T idxWrite = ch * IN_WIDTH * IN_HEIGHT + k * IN_WIDTH + w;
            TOutputWord out_word;
            out_word[0] = buf[idxWrite];
            out_data[0].write(out_word);
        }
        
    }
};