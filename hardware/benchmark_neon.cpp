/**
 * NEON SIMD Benchmark
 *
 * Compares scalar vs NEON performance for the SSM kernel.
 */

#include "ssm_kernel_neon.cpp"
#include <chrono>
#include <cmath>
#include <iostream>

// Scalar baseline for comparison
void ssm_step_scalar(int8_t x_byte, int8_t h[D_INNER][D_STATE],
                     const int8_t A[D_INNER][D_STATE], const int8_t B[D_STATE],
                     const int8_t C[D_STATE], const int8_t D_param[D_INNER],
                     int8_t dt_scale, int8_t y_out[D_INNER]) {
  for (int d = 0; d < D_INNER; d++) {
    int32_t y_acc = 0;
    for (int n = 0; n < D_STATE; n++) {
      int32_t A_abs = (A[d][n] < 0) ? -A[d][n] : A[d][n];
      int32_t dA = ((int32_t)dt_scale * A_abs) >> 7;
      int32_t dB = ((int32_t)dt_scale * B[n]) >> 7;

      int32_t h_new = dA * h[d][n] + dB * x_byte;
      h[d][n] = (int8_t)std::max(-127, std::min(127, h_new >> 7));

      y_acc += (int32_t)C[n] * h[d][n];
    }
    int32_t skip = ((int32_t)D_param[d] * x_byte) >> 7;
    int32_t y = (y_acc >> 7) + skip;
    y_out[d] = (int8_t)std::max(-127, std::min(127, y));
  }
}

int main() {
  std::cout << "NEON SIMD Benchmark" << std::endl;
  std::cout << "===================" << std::endl;
  std::cout << "D_INNER: " << D_INNER << ", D_STATE: " << D_STATE << std::endl;

  init_dt_lut();

  // Initialize weights
  int8_t A[D_INNER][D_STATE];
  int8_t B[D_STATE];
  int8_t C[D_STATE];
  int8_t D_param[D_INNER];

  for (int d = 0; d < D_INNER; d++) {
    for (int n = 0; n < D_STATE; n++) {
      A[d][n] = -3;
    }
    D_param[d] = 64;
  }
  for (int n = 0; n < D_STATE; n++) {
    B[n] = 25;
    C[n] = 25;
  }

  // States
  alignas(16) int8_t h_scalar[D_INNER][D_STATE] = {0};
  NEONState state_neon;
  state_neon.reset();

  int8_t y_scalar[D_INNER];
  int8_t y_neon[D_INNER];

  const int N_BYTES = 100000;
  int8_t dt_scale = dt_lut[128];

  // Warmup
  for (int i = 0; i < 1000; i++) {
    ssm_step_neon((int8_t)(i % 100), state_neon, A, B, C, D_param, dt_scale,
                  y_neon);
  }
  state_neon.reset();

  // Benchmark NEON
  std::cout << "\nBenchmarking " << N_BYTES << " bytes..." << std::endl;

  auto start_neon = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < N_BYTES; t++) {
    int8_t x = (int8_t)(std::sin(t * 0.05) * 80);
    ssm_step_neon(x, state_neon, A, B, C, D_param, dt_scale, y_neon);
  }
  auto end_neon = std::chrono::high_resolution_clock::now();

  // Benchmark scalar
  auto start_scalar = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < N_BYTES; t++) {
    int8_t x = (int8_t)(std::sin(t * 0.05) * 80);
    ssm_step_scalar(x, h_scalar, A, B, C, D_param, dt_scale, y_scalar);
  }
  auto end_scalar = std::chrono::high_resolution_clock::now();

  // Results
  auto neon_us = std::chrono::duration_cast<std::chrono::microseconds>(
                     end_neon - start_neon)
                     .count();
  auto scalar_us = std::chrono::duration_cast<std::chrono::microseconds>(
                       end_scalar - start_scalar)
                       .count();

  double neon_per_byte = (double)neon_us / N_BYTES;
  double scalar_per_byte = (double)scalar_us / N_BYTES;
  double speedup = scalar_per_byte / neon_per_byte;

  std::cout << "\n=== Results ===" << std::endl;
  std::cout << "  Scalar:  " << scalar_us << " µs (" << scalar_per_byte
            << " µs/byte)" << std::endl;
  std::cout << "  NEON:    " << neon_us << " µs (" << neon_per_byte
            << " µs/byte)" << std::endl;
  std::cout << "  Speedup: " << speedup << "x" << std::endl;

  // State energy check
  float neon_energy = 0, scalar_energy = 0;
  for (int d = 0; d < D_INNER; d++) {
    for (int n = 0; n < D_STATE; n++) {
      neon_energy += std::abs(state_neon.h[d][n]);
      scalar_energy += std::abs(h_scalar[d][n]);
    }
  }

  std::cout << "\n=== Validation ===" << std::endl;
  std::cout << "  NEON state energy:   " << neon_energy << std::endl;
  std::cout << "  Scalar state energy: " << scalar_energy << std::endl;

  if (neon_per_byte < 1.0) {
    std::cout << "\n✓ PASS: Sub-microsecond latency achieved!" << std::endl;
  } else if (neon_per_byte < 10.0) {
    std::cout << "\n✓ GOOD: Under 10µs per byte" << std::endl;
  }

  std::cout << "\n>>> LIQUID FLOW ACTIVE <<<" << std::endl;

  return 0;
}
