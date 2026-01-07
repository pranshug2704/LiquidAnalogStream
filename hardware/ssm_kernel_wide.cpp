/**
 * SSM Kernel - Full Dimensions with Wide Accumulators
 *
 * Design principles:
 * 1. int8 for weights/inputs (saves bandwidth)
 * 2. int32 for internal math (uses full DSP48 capacity)
 * 3. Re-quantize to int8 only when writing state back
 * 4. ARRAY_PARTITION for parallel dimension computation
 * 5. Distributed RAM (LUTs) for state - zero latency
 */

#ifndef SSM_KERNEL_WIDE_H
#define SSM_KERNEL_WIDE_H

#include <cmath>
#include <cstdint>

// Full dimensions matching Python model
constexpr int D_INNER = 128; // d_model * expand = 64 * 2
constexpr int D_STATE = 16;  // SSM state dimension
constexpr int SEQ_LEN = 128;

// Storage types (narrow for memory)
typedef int8_t narrow_t; // Storage/IO: int8
typedef int32_t wide_t;  // Computation: int32
typedef int64_t acc_t;   // Accumulator for big sums

// Q7 scaling (int8 represents [-1, 1])
constexpr int SCALE_SHIFT = 7;
constexpr narrow_t NARROW_MAX = 127;
constexpr narrow_t NARROW_MIN = -127;

// Delta (Δ) Lookup Table - 256 entries for exp() approximation
// Pre-computed: exp_lut[i] = exp(scale * (i - 128) / 128)
// This avoids expensive exp() on FPGA
constexpr int DT_LUT_SIZE = 256;
extern narrow_t dt_exp_lut[DT_LUT_SIZE]; // Initialized at runtime

// Initialize the exp LUT (call once at startup)
inline void init_dt_lut(float dt_min, float dt_max) {
  for (int i = 0; i < DT_LUT_SIZE; i++) {
    float x = (float)(i - 128) / 128.0f; // [-1, 1]
    float dt = dt_min + (dt_max - dt_min) * (x + 1.0f) / 2.0f;
    float exp_val = std::exp(-dt); // exp(-dt) for discretization
    dt_exp_lut[i] = (narrow_t)(exp_val * NARROW_MAX);
  }
}

// Saturating cast from wide to narrow
inline narrow_t saturate(wide_t x) {
  if (x > NARROW_MAX)
    return NARROW_MAX;
  if (x < NARROW_MIN)
    return NARROW_MIN;
  return (narrow_t)x;
}

/**
 * SSM Single-Step with Wide Accumulators
 *
 * State is stored as int8 but all math uses int32.
 * Re-quantization happens only when writing back.
 */
void ssm_step_wide(narrow_t x_byte,                    // Input byte (Q7)
                   narrow_t h[D_INNER][D_STATE],       // State (stored as int8)
                   const narrow_t A[D_INNER][D_STATE], // A matrix (int8)
                   const narrow_t B[D_STATE],          // B vector (int8)
                   const narrow_t C[D_STATE],          // C vector (int8)
                   const narrow_t D_param[D_INNER],    // D skip (int8)
                   uint8_t dt_idx,                     // Index into exp LUT
                   narrow_t y_out[D_INNER]             // Output (int8)
) {
// HLS: Partition state for parallel access
#pragma HLS ARRAY_PARTITION variable = h complete dim = 2
#pragma HLS ARRAY_PARTITION variable = A complete dim = 2
#pragma HLS ARRAY_PARTITION variable = y_out complete

  // Lookup discretized A from LUT
  narrow_t dA_scale = dt_exp_lut[dt_idx];

// Process all inner dimensions in parallel
#pragma HLS PIPELINE II = 1
inner_loop:
  for (int d = 0; d < D_INNER; d++) {
#pragma HLS UNROLL factor = 32 // 4 parallel blocks of 32

    acc_t y_acc = 0; // 64-bit accumulator for large sums

  // State update - fully unrolled for D_STATE
  state_loop:
    for (int n = 0; n < D_STATE; n++) {
#pragma HLS UNROLL // Complete unroll (D_STATE=16)

      // Wide math: int32 precision
      wide_t h_wide = (wide_t)h[d][n];
      wide_t A_wide = (wide_t)A[d][n];
      wide_t B_wide = (wide_t)B[n];
      wide_t x_wide = (wide_t)x_byte;

      // dA = exp(dt * A) from LUT, scaled
      wide_t dA = ((wide_t)dA_scale * A_wide) >> SCALE_SHIFT;
      // Clamp dA to reasonable range
      dA = (dA > NARROW_MAX) ? NARROW_MAX : (dA < 0 ? 0 : dA);

      // dB = dt_scale * B
      wide_t dB = ((wide_t)dA_scale * B_wide) >> SCALE_SHIFT;

      // State update: h_new = dA * h + dB * x
      wide_t h_new = (dA * h_wide + dB * x_wide) >> SCALE_SHIFT;

      // Re-quantize to int8 for storage
      h[d][n] = saturate(h_new);

      // Accumulate output: y += C * h
      y_acc += (wide_t)C[n] * h[d][n];
    }

    // Skip connection: y = acc + D * x
    wide_t skip = ((wide_t)D_param[d] * x_byte) >> SCALE_SHIFT;
    wide_t y_total = (y_acc >> SCALE_SHIFT) + skip;

    y_out[d] = saturate(y_total);
  }
}

// State memory - fits in distributed RAM (LUTs) at 2KB per layer
// For 12 layers: 24KB total, easily fits in FPGA fabric
#pragma HLS BIND_STORAGE variable = state type = ram_1p impl = lutram

#endif // SSM_KERNEL_WIDE_H

// LUT storage (defined in implementation file)
narrow_t dt_exp_lut[DT_LUT_SIZE];
