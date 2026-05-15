#include "StreamingConverter.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include "utils/CSDFG_utils.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>

using namespace test_config;

using TInputWord = std::array<test_config::TInput, test_config::W_PAR>;
using TOutputWord = std::array<test_config::TOutput, test_config::CH_PAR>;

// For this unit test, we build a StreamingConverter instance with the same template as in config
using Conv = StreamingConverter<
    TInputWord,   TInput,
    TOutputWord,  TOutput,
    IN_HEIGHT, IN_WIDTH,
    IN_CH, W_PAR, CH_PAR>;

bool test_run() {
    // streams
    hls::stream<TInputWord>  in_data[W_PAR];
    hls::stream<TOutputWord> out_data[CH_PAR];

    // Fill input with a distinct pattern so we can check mapping:
    // value = ch*100 + r*10 + c
    for (int r = 0; r < (int)IN_HEIGHT; ++r) {
        for (int w = 0; w < (int)(IN_WIDTH / W_PAR); ++w) {
            for (int ch = 0; ch < (int)(IN_CH / CH_PAR); ++ch) {
                for (int w_i = 0; w_i < (int)W_PAR; ++w_i) {
                    TInputWord word{};
                    int c = w * W_PAR + w_i;
                    for (int ch_i = 0; ch_i < (int)CH_PAR; ++ch_i) {
                        int ch = ch * CH_PAR + ch_i;
                        int val = ch * 100 + r * 10 + c;
                        word[ch_i] = (TInput)val;
                    }
                    in_data[w_i].write(word);
                }
            }
        }
    }

    // Run DUT
    Conv conv;
    conv.template run<0>(in_data, out_data);

    // Read outputs and check mapping:
    // We expect out_data[ch_i] to carry all values for that channel lane across width streams
    bool ok = true;

    for (int r = 0; r < (int)IN_HEIGHT; ++r) {
        for (int w = 0; w < (int)(IN_WIDTH / W_PAR); ++w) {
            for (int ch = 0; ch < (int)(IN_CH / CH_PAR); ++ch) {
                for (int ch_i = 0; ch_i < (int)CH_PAR; ++ch_i) {
                    int ch = ch * CH_PAR + ch_i;
                    // One output word from stream ch_i
                    TOutputWord out_word = out_data[ch_i].read();
                    for (int w_i = 0; w_i < (int)W_PAR; ++w_i) {
                        int c = w * W_PAR + w_i;
                        int expected = ch * 100 + r * 10 + c;
                        int got = (int)out_word[w_i];
                        if (got != expected) {
                            std::cout << "Mismatch at (r=" << r
                                      << ", w=" << w
                                      << ", ch=" << ch
                                      << ", w_i=" << w_i
                                      << "). got " << got
                                      << ", expected " << expected << "\n";
                            ok = false;
                        }
                    }
                }
            }
        }
    }

    // Ensure no extra outputs
    for (int s = 0; s < (int)CH_PAR; ++s) {
        if (!out_data[s].empty()) {
            std::cout << "Extra data in out_data[" << s << "] after reads.\n";
            ok = false;
        }
    }

    if (ok) {
        std::cout << "test_run: PASSED\n";
    } else {
        std::cout << "test_run: FAILED\n";
    }
    return ok;
}

// Optional: step-based test to verify schedule (similar to your CSDFG tests)
bool test_step() {
    hls::stream<TInputWord>  in_data[W_PAR];
    hls::stream<TOutputWord> out_data[CH_PAR];

    Conv conv;
    conv.step_init();

    // Expected total firings in one full sweep of (k, wg, chg)
    static constexpr size_t expectedFirings =
        (IN_CH / CH_PAR) * (IN_WIDTH / W_PAR) * IN_HEIGHT;

    // For step-based testing, we feed dummy data on every call and count that step() fires expectedFirings times before returning to (0,0,0).
    size_t fire_count = 0;
    size_t max_steps  = 3 * expectedFirings;

    for (size_t cycle = 0; cycle < max_steps; ++cycle) {
        // feed one word per input stream so firing_condition can be true
        for (int w_i = 0; w_i < (int)W_PAR; ++w_i) {
            TInputWord dummy{};
            in_data[w_i].write(dummy);
        }

        ActorStatus st = conv.step(in_data, out_data);
        if (st.has_fired()) {
            fire_count++;
        }

        // optionally break once we think we’ve completed a full sweep
        if (fire_count == expectedFirings) break;
    }

    bool ok = (fire_count == expectedFirings);
    std::cout << "test_step: expected firings = " << expectedFirings
              << ", observed = " << fire_count << "\n";
    return ok;
}

int main(int argc, char** argv) {
    bool all_ok = true;

    all_ok &= test_run();

    if (argc > 1 && std::string(argv[1]) == "csim") {
        all_ok &= test_step();
    }

    if (!all_ok) {
        std::cout << "Overall: FAILED\n";
        return 1;
    } else {
        std::cout << "Overall: PASSED\n";
        return 0;
    }
}