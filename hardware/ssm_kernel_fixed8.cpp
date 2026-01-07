/**
 * SSM Kernel - Fixed-Point int8 Implementation
 *
 * Hardware-ready: Uses int8 with Q7 format (1 sign bit, 7 fractional bits).
 * Maps directly to analog voltage ranges or FPGA LUTs.
 */

#ifndef SSM_KERNEL_FIXED8_H
#define SSM_KERNEL_FIXED8_H

#include <cmath>
#include <cstdint>

// Configuration
constexpr int D_INNER = 32; // Reduced for int8 overflow safety
constexpr int D_STATE = 8;
constexpr int SEQ_LEN = 64;

// Fixed-point Q7 format: value = raw / 128
typedef int8_t fixed8_t;
typedef int16_t fixed16_t; // For intermediate products
constexpr int8_t SCALE = 127;

// Conversion utilities
inline fixed8_t float_to_q7(float x) {
  float clamped = fmaxf(-1.0f, fminf(1.0f, x)); // Clamp to [-1, 1]
  return static_cast<fixed8_t>(clamped * SCALE);
}

inline float q7_to_float(fixed8_t x) { return static_cast<float>(x) / SCALE; }

// Q7 multiply: (a/128) * (b/128) * 128 = (a*b) >> 7
inline fixed8_t q7_mul(fixed8_t a, fixed8_t b) {
  fixed16_t product = static_cast<fixed16_t>(a) * b;
  return static_cast<fixed8_t>((product + 64) >> 7); // Round
}

// Q7 multiply-accumulate (MAC) - the core analog operation
inline fixed16_t q7_mac(fixed16_t acc, fixed8_t a, fixed8_t b) {
  return acc + (static_cast<fixed16_t>(a) * b);
}

// Fast exp approximation for small x: exp(x) ≈ 1 + x
inline fixed8_t q7_exp_approx(fixed8_t x) {
  // For small x, exp(x) ≈ 1 + x
  // In Q7: 1.0 = 127, so exp(x) ≈ 127 + x
  fixed16_t result = 127 + x;
  return static_cast<fixed8_t>(fmax(-127, fmin(127, result)));
}

/**
 * SSM Selective Scan - int8 Fixed-Point
 *
 * All operations use int8 (Q7 format) for analog HW compatibility.
 * Intermediate sums use int16 to avoid overflow.
 */
void ssm_scan_fixed8(const fixed8_t x[SEQ_LEN][D_INNER],
                     const fixed8_t dt[SEQ_LEN][D_INNER],
                     const fixed8_t B[SEQ_LEN][D_STATE],
                     const fixed8_t C[SEQ_LEN][D_STATE],
                     const fixed8_t A[D_INNER][D_STATE],
                     const fixed8_t D_param[D_INNER],
                     fixed8_t y[SEQ_LEN][D_INNER]) {
  // Hidden state (int16 to accumulate without overflow)
  fixed16_t h[D_INNER][D_STATE] = {0};

  // Scan loop
  for (int t = 0; t < SEQ_LEN; t++) {
    for (int d = 0; d < D_INNER; d++) {
      fixed16_t y_acc = 0;

      for (int n = 0; n < D_STATE; n++) {
        // dA = exp(dt * A) ≈ 1 + dt * A (Taylor)
        fixed8_t dt_A = q7_mul(dt[t][d], A[d][n]);
        fixed8_t dA = q7_exp_approx(dt_A);

        // dB = dt * B
        fixed8_t dB = q7_mul(dt[t][d], B[t][n]);

        // State update: h = dA * h + dB * x
        // Using int16 for h to avoid overflow
        fixed16_t h_new = (static_cast<fixed16_t>(dA) * (h[d][n] >> 7) +
                           static_cast<fixed16_t>(dB) * x[t][d]) >>
                          7;
        h[d][n] = h_new;

        // Output: y += C * h
        y_acc = q7_mac(y_acc, C[t][n], static_cast<fixed8_t>(h[d][n] >> 7));
      }

      // Skip connection: y = y_acc + D * x
      fixed16_t skip = static_cast<fixed16_t>(D_param[d]) * x[t][d];
      fixed16_t y_total = (y_acc >> 7) + (skip >> 7);

      // Clamp to int8 range
      y[t][d] = static_cast<fixed8_t>(fmax(-127, fmin(127, y_total)));
    }
  }
}

#endif // SSM_KERNEL_FIXED8_H
