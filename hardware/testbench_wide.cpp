/**
 * Wide Accumulator Testbench
 *
 * Tests the full 128×16 SSM kernel with int32 accumulators.
 */

#include "ssm_kernel_wide.cpp"
#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
  std::cout << "Wide Accumulator SSM Testbench" << std::endl;
  std::cout << "==============================" << std::endl;
  std::cout << "D_INNER: " << D_INNER << ", D_STATE: " << D_STATE << std::endl;
  std::cout << "State size: " << (D_INNER * D_STATE)
            << " bytes = " << (D_INNER * D_STATE / 1024.0) << " KB per layer"
            << std::endl;

  // Initialize exp LUT
  init_dt_lut(0.001f, 0.1f);
  std::cout << "Initialized dt exp LUT" << std::endl;

  // Allocate (these would be in distributed RAM on FPGA)
  narrow_t h[D_INNER][D_STATE] = {0};
  narrow_t A[D_INNER][D_STATE];
  narrow_t B[D_STATE];
  narrow_t C[D_STATE];
  narrow_t D_param[D_INNER];
  narrow_t y[D_INNER];

  srand(42);

  // Initialize weights
  for (int d = 0; d < D_INNER; d++) {
    for (int n = 0; n < D_STATE; n++) {
      A[d][n] = -3 - (rand() % 5); // Small negative
    }
    D_param[d] = 64; // 0.5 in Q7
  }
  for (int n = 0; n < D_STATE; n++) {
    B[n] = 20 + (rand() % 20);
    C[n] = 20 + (rand() % 20);
  }

  // Run on sine wave input
  std::cout << "\nProcessing sine wave (100 bytes)..." << std::endl;

  int overflow_count = 0;
  float y_sum = 0;

  for (int t = 0; t < 100; t++) {
    // Sine wave input
    float sine_val = std::sin(t * 0.1f);
    narrow_t x = (narrow_t)(sine_val * 100);

    // dt varies with input (the "Liquid" property)
    uint8_t dt_idx = 128 + (uint8_t)(sine_val * 50); // Center ± 50

    ssm_step_wide(x, h, A, B, C, D_param, dt_idx, y);

    // Check outputs
    for (int d = 0; d < D_INNER; d++) {
      y_sum += y[d];
      if (y[d] == 127 || y[d] == -127)
        overflow_count++;
    }
  }

  // Statistics
  float y_mean = y_sum / (100 * D_INNER);
  float h_energy = 0;
  for (int d = 0; d < D_INNER; d++) {
    for (int n = 0; n < D_STATE; n++) {
      h_energy += std::abs(h[d][n]);
    }
  }

  std::cout << "\nResults:" << std::endl;
  std::cout << "  Output mean:    " << y_mean << std::endl;
  std::cout << "  State energy:   " << h_energy << std::endl;
  std::cout << "  Saturated:      " << overflow_count << "/" << (100 * D_INNER)
            << " (" << (100.0f * overflow_count / (100 * D_INNER)) << "%)"
            << std::endl;

  // Pass/Fail
  if (h_energy > 0 && overflow_count < 100 * D_INNER / 10) {
    std::cout << "\n✓ PASS: Full 128x16 kernel working" << std::endl;
    return 0;
  } else {
    std::cout << "\n✗ FAIL: Check saturation or state evolution" << std::endl;
    return 1;
  }
}
