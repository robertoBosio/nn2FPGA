#pragma once
#include "ap_int.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>
 /*In self-timed execution, it behaves like a
 * fixed throughput producer, producing data at a fixed rate defined by the
 * neural network's iteration interval, which means it does not flood the
 * network with data but produces it at the same rate as the neural network is
 * expected to consume it.*/

 template <typename TOutputWord> class FixedThroughputDMA {
 public:
   FixedThroughputDMA() : FixedThroughputDMA(1, 1) {}

   FixedThroughputDMA(size_t words_per_tensor, size_t nnII)
       : STEP_fixed_counter(0), STEP_nnII(nnII),
         STEP_words_per_tensor(words_per_tensor), STEP_actor_status(1, nnII) {}

   ActorStatus step(hls::stream<TOutputWord> &output_data_stream) {
     // Write data with a specific throughput, defined by the ratio between
     // the number of data to produce and the neural network iteration
     // interval (nnII).

     if (STEP_fixed_counter >= STEP_nnII) {
       // If we have reached the II cycles, reset the counter.
       STEP_fixed_counter -= STEP_nnII;

       // Write a token.
       TOutputWord output_data;
       output_data.data = 0;
       output_data_stream.write(output_data);
     }
     STEP_actor_status.fire();
     STEP_actor_status.advance();
     STEP_fixed_counter += STEP_words_per_tensor;
     return STEP_actor_status;
   }

 private:
   // Fixed throughput state variables
   size_t STEP_fixed_counter;
   size_t STEP_nnII;
   size_t STEP_words_per_tensor;
   ActorStatus STEP_actor_status;
 };

 template <typename TInputWord> class InfiniteThroughputDMA {
 public:
   InfiniteThroughputDMA() : InfiniteThroughputDMA(1) {}

   InfiniteThroughputDMA(size_t words_per_tensor)
       : STEP_words_per_tensor(words_per_tensor),
         STEP_actor_status(1, words_per_tensor) {}

   ActorStatus step(hls::stream<TInputWord> &input_data_stream) {
     // Read data with infinite throughput, meaning that it reads data as soon
     // as it is available in the input stream.
     bool firing_condition = true;
     if (input_data_stream.empty()) {
       firing_condition = false;
     }

     if (firing_condition) {
       // Read a token.
       TInputWord input_data = input_data_stream.read();
       (void)input_data; // Suppress unused variable warning
       STEP_actor_status.fire();
     }

     STEP_actor_status.advance();
     return STEP_actor_status;
   }

 private:
   size_t STEP_words_per_tensor;
   ActorStatus STEP_actor_status;
 };