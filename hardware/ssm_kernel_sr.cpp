/**
 * SSM Kernel with Stochastic Rounding
 *
 * Features:
 * - LFSR-based pseudo-random for SR (no rand() overhead)
 * - Eliminates negative bias from truncation
 * - Full 128×16 dimensions with int32 accumulators
 */

#ifndef SSM_KERNEL_SR_H
#define SSM_KERNEL_SR_H

#include <cmath>
#include <cstdint>

constexpr int D_INNER = 128;
constexpr int D_STATE = 16;

typedef int8_t narrow_t;
typedef int32_t wide_t;
constexpr int SCALE_SHIFT = 7;
constexpr narrow_t NARROW_MAX = 127;
constexpr narrow_t NARROW_MIN = -127;

// ============================================
// LFSR Stochastic Rounding
// ============================================

// 16-bit LFSR state (seeded with non-zero value)
static uint16_t lfsr_state = 0xACE1;

// Fast LFSR - just a few XOR gates on FPGA
inline uint16_t lfsr_next() {
  uint16_t bit = ((lfsr_state >> 0) ^ (lfsr_state >> 2) ^ (lfsr_state >> 3) ^
                  (lfsr_state >> 5)) &
                 1;
  lfsr_state = (lfsr_state >> 1) | (bit << 15);
  return lfsr_state;
}

// Stochastic rounding: converts wide_t to narrow_t with random rounding
inline narrow_t stochastic_round(wide_t val, int shift) {
  // Extract fractional bits that would be lost
  wide_t mask = (1 << shift) - 1;
  wide_t fraction = val & mask;

  // Random threshold
  uint16_t threshold = lfsr_next() & mask;

  // Probabilistic round-up based on fraction vs threshold
  wide_t rounded = val + (fraction > threshold ? (1 << shift) : 0);
  rounded >>= shift;

  // Saturate to int8 range
  if (rounded > NARROW_MAX)
    return NARROW_MAX;
  if (rounded < NARROW_MIN)
    return NARROW_MIN;
  return (narrow_t)rounded;
}

// Standard saturating round (for comparison)
inline narrow_t standard_round(wide_t val, int shift) {
  val >>= shift;
  if (val > NARROW_MAX)
    return NARROW_MAX;
  if (val < NARROW_MIN)
    return NARROW_MIN;
  return (narrow_t)val;
}

// ============================================
// Delta LUT
// ============================================
constexpr int DT_LUT_SIZE = 256;
static narrow_t dt_exp_lut[DT_LUT_SIZE];

inline void init_dt_lut(float dt_min, float dt_max) {
  for (int i = 0; i < DT_LUT_SIZE; i++) {
    float x = (float)(i - 128) / 128.0f;
    float dt = dt_min + (dt_max - dt_min) * (x + 1.0f) / 2.0f;
    dt_exp_lut[i] = (narrow_t)(std::exp(-dt) * NARROW_MAX);
  }
}

// ============================================
// SSM Step with Stochastic Rounding
// ============================================
void ssm_step_sr(narrow_t x_byte, narrow_t h[D_INNER][D_STATE],
                 const narrow_t A[D_INNER][D_STATE], const narrow_t B[D_STATE],
                 const narrow_t C[D_STATE], const narrow_t D_param[D_INNER],
                 uint8_t dt_idx, narrow_t y_out[D_INNER],
                 bool use_stochastic = true) {
#pragma HLS ARRAY_PARTITION variable = h complete dim = 2
#pragma HLS PIPELINE II = 1

  narrow_t dA_scale = dt_exp_lut[dt_idx];

  for (int d = 0; d < D_INNER; d++) {
#pragma HLS UNROLL factor = 32

    wide_t y_acc = 0;

    for (int n = 0; n < D_STATE; n++) {
#pragma HLS UNROLL

      wide_t h_wide = (wide_t)h[d][n];
      wide_t dA = ((wide_t)dA_scale * (wide_t)A[d][n]) >> SCALE_SHIFT;
      dA = (dA > NARROW_MAX) ? NARROW_MAX : (dA < 0 ? 0 : dA);
      wide_t dB = ((wide_t)dA_scale * (wide_t)B[n]) >> SCALE_SHIFT;

      wide_t h_new = dA * h_wide + dB * (wide_t)x_byte;

      // Apply stochastic or standard rounding
      h[d][n] = use_stochastic ? stochastic_round(h_new, SCALE_SHIFT)
                               : standard_round(h_new, SCALE_SHIFT);

      y_acc += (wide_t)C[n] * h[d][n];
    }

    wide_t skip = ((wide_t)D_param[d] * (wide_t)x_byte);
    wide_t y_total = y_acc + skip;

    y_out[d] = use_stochastic ? stochastic_round(y_total, SCALE_SHIFT)
                              : standard_round(y_total, SCALE_SHIFT);
  }
}

#endif // SSM_KERNEL_SR_H
