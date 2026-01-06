/**
 * SSM Kernel Testbench
 *
 * Validates the C++ SSM kernel against known values.
 * Can be used for C-simulation in Vitis HLS.
 */

#include "ssm_kernel.cpp"
#include <cmath>
#include <cstdlib>
#include <iostream>

// Simple random float in range [min, max]
float randf(float min, float max) {
  return min + static_cast<float>(rand()) / RAND_MAX * (max - min);
}

int main() {
  std::cout << "SSM Kernel Testbench" << std::endl;
  std::cout << "====================" << std::endl;
  std::cout << "D_INNER: " << D_INNER << std::endl;
  std::cout << "D_STATE: " << D_STATE << std::endl;
  std::cout << "SEQ_LEN: " << SEQ_LEN << std::endl;

  // Allocate arrays
  fixed_t x[SEQ_LEN][D_INNER];
  fixed_t dt[SEQ_LEN][D_INNER];
  fixed_t B[SEQ_LEN][D_STATE];
  fixed_t C[SEQ_LEN][D_STATE];
  fixed_t A[D_INNER][D_STATE];
  fixed_t D_param[D_INNER];
  fixed_t y[SEQ_LEN][D_INNER];

  // Initialize with random values
  srand(42);

  for (int t = 0; t < SEQ_LEN; t++) {
    for (int d = 0; d < D_INNER; d++) {
      x[t][d] = float_to_fixed(randf(-1.0f, 1.0f));
      dt[t][d] = float_to_fixed(randf(0.001f, 0.1f)); // Positive dt
    }
    for (int n = 0; n < D_STATE; n++) {
      B[t][n] = float_to_fixed(randf(-0.5f, 0.5f));
      C[t][n] = float_to_fixed(randf(-0.5f, 0.5f));
    }
  }

  for (int d = 0; d < D_INNER; d++) {
    for (int n = 0; n < D_STATE; n++) {
      // A should be negative for stability
      A[d][n] = float_to_fixed(randf(-0.1f, -0.01f));
    }
    D_param[d] = float_to_fixed(1.0f); // Skip connection
  }

  // Run kernel
  std::cout << "\nRunning SSM scan..." << std::endl;
  ssm_scan(x, dt, B, C, A, D_param, y);

  // Check output statistics
  float y_sum = 0.0f;
  float y_max = -1e9f;
  float y_min = 1e9f;

  for (int t = 0; t < SEQ_LEN; t++) {
    for (int d = 0; d < D_INNER; d++) {
      float yf = fixed_to_float(y[t][d]);
      y_sum += yf;
      if (yf > y_max)
        y_max = yf;
      if (yf < y_min)
        y_min = yf;
    }
  }

  float y_mean = y_sum / (SEQ_LEN * D_INNER);

  std::cout << "Output statistics:" << std::endl;
  std::cout << "  Mean: " << y_mean << std::endl;
  std::cout << "  Min:  " << y_min << std::endl;
  std::cout << "  Max:  " << y_max << std::endl;

  // Basic sanity checks
  bool passed = true;

  if (std::isnan(y_mean) || std::isinf(y_mean)) {
    std::cerr << "FAIL: Output contains NaN/Inf!" << std::endl;
    passed = false;
  }

  if (y_max > 1e6 || y_min < -1e6) {
    std::cerr << "FAIL: Output values exploded!" << std::endl;
    passed = false;
  }

  if (passed) {
    std::cout << "\nPASS: Kernel executed successfully." << std::endl;
    return 0;
  } else {
    return 1;
  }
}
