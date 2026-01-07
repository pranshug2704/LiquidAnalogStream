/**
 * SSM Kernel - ARM NEON SIMD Optimized
 *
 * Processes 16 int8 values per instruction for sub-microsecond latency.
 * Target: Apple Silicon (M1/M2/M3) and ARM64 processors.
 *
 * Expected speedup: ~16x over scalar code
 * Target latency: <1µs per byte
 */

#ifndef SSM_KERNEL_NEON_H
#define SSM_KERNEL_NEON_H

#include <arm_neon.h>
#include <cmath>
#include <cstdint>

// Dimensions (D_STATE=16 fits perfectly in NEON 128-bit register)
constexpr int D_INNER = 128;
constexpr int D_STATE = 16; // Perfect for int8x16_t

typedef int8_t narrow_t;
typedef int16_t mid_t;
typedef int32_t wide_t;

// ============================================
// LFSR (vectorized for 16 parallel streams)
// ============================================
static uint16_t lfsr_states[16] = {
    0xACE1, 0xBEEF, 0xCAFE, 0xDEAD, 0xFACE, 0x1234, 0x5678, 0x9ABC,
    0xDEF0, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777};

inline void lfsr_next_16(uint16_t *states) {
  for (int i = 0; i < 16; i++) {
    uint16_t bit = ((states[i] >> 0) ^ (states[i] >> 2) ^ (states[i] >> 3) ^
                    (states[i] >> 5)) &
                   1;
    states[i] = (states[i] >> 1) | (bit << 15);
  }
}

// ============================================
// Delta LUT (vectorized load)
// ============================================
static int8_t dt_lut[256];

inline void init_dt_lut() {
  for (int i = 0; i < 256; i++) {
    float x = (float)(i - 128) / 128.0f;
    float dt = 0.001f + (0.1f - 0.001f) * (x + 1.0f) / 2.0f;
    dt_lut[i] = (int8_t)(std::exp(-dt) * 127);
  }
}

// ============================================
// NEON Saturating Arithmetic Helpers
// ============================================

// Saturating narrow from int16 to int8
inline int8x16_t saturate_narrow_s16(int16x8_t lo, int16x8_t hi) {
  return vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi));
}

// Multiply int8 vectors, accumulate to int16
inline int16x8_t vmull_extend_lo(int8x16_t a, int8x16_t b) {
  return vmull_s8(vget_low_s8(a), vget_low_s8(b));
}

inline int16x8_t vmull_extend_hi(int8x16_t a, int8x16_t b) {
  return vmull_s8(vget_high_s8(a), vget_high_s8(b));
}

// ============================================
// SSM State (NEON-aligned)
// ============================================
struct alignas(16) NEONState {
  int8_t h[D_INNER][D_STATE]; // 128×16 = 2KB, aligned

  void reset() {
    for (int d = 0; d < D_INNER; d++) {
      vst1q_s8(&h[d][0], vdupq_n_s8(0));
    }
  }
};

// ============================================
// NEON-Optimized SSM Step
// ============================================
inline void ssm_step_neon(int8_t x_byte, NEONState &state,
                          const int8_t A[D_INNER][D_STATE],
                          const int8_t B[D_STATE], const int8_t C[D_STATE],
                          const int8_t D_param[D_INNER], int8_t dt_scale,
                          int8_t y_out[D_INNER]) {
  // Broadcast input and scale
  int8x16_t x_vec = vdupq_n_s8(x_byte);
  int8x16_t dt_vec = vdupq_n_s8(dt_scale);

  // Load B and C (both are D_STATE=16, one NEON register each)
  int8x16_t B_vec = vld1q_s8(B);
  int8x16_t C_vec = vld1q_s8(C);

  // Compute dB = dt_scale * B (scaled)
  int16x8_t dB_lo = vmull_s8(vget_low_s8(dt_vec), vget_low_s8(B_vec));
  int16x8_t dB_hi = vmull_s8(vget_high_s8(dt_vec), vget_high_s8(B_vec));
  // Shift down by 7 (divide by 128)
  dB_lo = vshrq_n_s16(dB_lo, 7);
  dB_hi = vshrq_n_s16(dB_hi, 7);
  int8x16_t dB_vec = saturate_narrow_s16(dB_lo, dB_hi);

  // Process all 128 inner dimensions
  for (int d = 0; d < D_INNER; d++) {
    // Load current state h[d][0:16]
    int8x16_t h_vec = vld1q_s8(&state.h[d][0]);

    // Load A[d][0:16]
    int8x16_t A_vec = vld1q_s8(&A[d][0]);

    // Compute dA = dt_scale * |A| (use absolute for decay)
    int8x16_t A_abs = vabsq_s8(A_vec);
    int16x8_t dA_lo = vmull_s8(vget_low_s8(dt_vec), vget_low_s8(A_abs));
    int16x8_t dA_hi = vmull_s8(vget_high_s8(dt_vec), vget_high_s8(A_abs));
    dA_lo = vshrq_n_s16(dA_lo, 7);
    dA_hi = vshrq_n_s16(dA_hi, 7);
    int8x16_t dA_vec = saturate_narrow_s16(dA_lo, dA_hi);

    // State update: h_new = dA * h + dB * x
    // Expand to int16 for multiplication
    int16x8_t h_lo = vmull_s8(vget_low_s8(dA_vec), vget_low_s8(h_vec));
    int16x8_t h_hi = vmull_s8(vget_high_s8(dA_vec), vget_high_s8(h_vec));

    int16x8_t input_lo = vmull_s8(vget_low_s8(dB_vec), vget_low_s8(x_vec));
    int16x8_t input_hi = vmull_s8(vget_high_s8(dB_vec), vget_high_s8(x_vec));

    // Add and shift
    h_lo = vaddq_s16(h_lo, input_lo);
    h_hi = vaddq_s16(h_hi, input_hi);
    h_lo = vshrq_n_s16(h_lo, 7);
    h_hi = vshrq_n_s16(h_hi, 7);

    // Saturate back to int8
    h_vec = saturate_narrow_s16(h_lo, h_hi);

    // Store updated state
    vst1q_s8(&state.h[d][0], h_vec);

    // Compute output: y = sum(C * h) + D * x
    int16x8_t y_lo = vmull_s8(vget_low_s8(C_vec), vget_low_s8(h_vec));
    int16x8_t y_hi = vmull_s8(vget_high_s8(C_vec), vget_high_s8(h_vec));

    // Horizontal sum
    int16x8_t y_sum = vaddq_s16(y_lo, y_hi);
    int32x4_t y_wide = vpaddlq_s16(y_sum);
    int64x2_t y_wider = vpaddlq_s32(y_wide);
    int32_t y_scalar = vgetq_lane_s64(y_wider, 0) + vgetq_lane_s64(y_wider, 1);

    // Add skip connection
    y_scalar = (y_scalar >> 7) + ((int32_t)D_param[d] * x_byte >> 7);

    // Saturate
    if (y_scalar > 127)
      y_scalar = 127;
    if (y_scalar < -127)
      y_scalar = -127;
    y_out[d] = (int8_t)y_scalar;
  }
}

#endif // SSM_KERNEL_NEON_H
