#include "StreamingTranspose.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>


using TInputWord = std::array<test_config::TInput, 1>;
using TOutputWord = std::array<test_config::TOutput, 1>;
using INDEX_T    = test_config::INDEX_T;

void wrap_run(hls::stream<TInputWord> in_data[1],
              hls::stream<TOutputWord> out_data[1]) {
  StreamingTranspose<test_config::TInputWord, test_config::TInput,
                     test_config::TOutputWord, test_config::TOutput
                     test_config::IN_HEIGHT,test_config::IN_WIDTH,test_config::IN_CH,
                     test_config::INDEX_T> transpose;
  transpose.run<0>(in_data, out_data);
}

bool test_run() {
  hls::stream<TInputWord> in_data[1];
  hls::stream<TOutputWord> out_data[1];

    
 // Write input: pack scalar at lane 0, rest 0
    for (int ch = 0; ch < (int)IN_CH; ch++) {
        for (int w = 0; w < (int)IN_WIDTH; w++) {
            for (int k = 0; k < (int)IN_HEIGHT; k++) {
                TInputWord pkt{};
                pkt[0] = input_tensor1[0][ch][w][k];
                in_data[0].write(pkt);
            }
        }
    }

 wrap_run(in_data, out_data);

 //read output
  bool flag = true;

    // Read back in transposed order [ch][k][w]
    for (int ch = 0; ch < (int)IN_CH; ch++) {
        for (int k = 0; k < (int)IN_HEIGHT; k++) {
            for (int w = 0; w < (int)IN_WIDTH; w++) {
                TOutputWord pkt{};
                out_data[0].read(pkt);
                TOutput got      = pkt[0];
                TOutput expected = input_tensor1[0][ch][w][k]; // original [ch][w][k] → now [ch][k][w]
                bool cmp = (got == expected);
                if (!cmp) {
                    std::cout << "Mismatch at (ch=" << ch
                              << ", k=" << k
                              << ", w=" << w
                              << "). got: " << (int)got
                              << ", expected: " << (int)expected << std::endl;
                }
                flag &= cmp;
            }
        }
    }

    if (!out_data[0].empty()) {
        flag = false;
        std::cout << "Output stream not empty after reading." << std::endl;
    }
    return flag;
    
}


bool test_step() {

  using namespace test_config;

    static constexpr size_t expectedII = 2 * (IN_CH * IN_WIDTH * IN_HEIGHT);

  hls::stream<TInputWordB> in_data[1];
  hls::stream<TOutputWord> out_data[1];
  
  StreamingTranspose< TInputWord, TInput,
                      TOutputWord, TOutput,
                      IN_HEIGHT,
                      IN_WIDTH,
                      IN_CH,
                      W_PAR, CH_PAR,
                      INDEX_T> transpose;
                     
                      
  transpose.step_init();   

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;

 while (true) {
        // feed dummy packet
        in_data[0].write(TInputWord());

        ActorStatus actor_status = transpose.step(in_data, out_data);

        std::vector<ActorStatus> actor_statuses = {actor_status};
        std::vector<size_t>      channel_quantities = {0};
        current_state = CSDFGState(actor_statuses, channel_quantities);

        auto it = visited_states.find(current_state);
        if (it != visited_states.end()) {
            II = clock_cycles - it->second;
            break;
        }
        visited_states.emplace(current_state, clock_cycles);
        clock_cycles++;
        assert(clock_cycles < 10 * expectedII);
    }

  // Flush outputs
    TOutputWord out_val;
    while (out_data[0].read_nb(out_val)) {}

    bool flag = (II == expectedII);
    std::cout << "Expected II: " << expectedII
              << ", Measured II: " << II << std::endl;
    return flag;
}

int main(int argc, char **argv) {
    bool all_passed = true;

    all_passed &= test_run();

    if (argc > 1 && std::string(argv[1]) == "csim") {
        all_passed &= test_step();
    }

    if (!all_passed) {
        std::cout << "Failed." << std::endl;
    } else {
        std::cout << "Passed." << std::endl;
    }

    return all_passed ? 0 : 1;
}