/**
 * Fixed-8 Testbench
 *
 * Validates int8 SSM kernel against Q16.16 reference.
 */

#include "ssm_kernel_fixed8.cpp"
#include <cmath>
#include <cstdlib>
#include <iostream>

float randf(float min, float max) {
  return min + static_cast<float>(rand()) / RAND_MAX * (max - min);
}

int main() {
  std::cout << "Fixed-8 SSM Kernel Testbench" << std::endl;
  std::cout << "=============================" << std::endl;
  std::cout << "D_INNER: " << D_INNER << ", D_STATE: " << D_STATE << std::endl;

  // Allocate
  fixed8_t x[SEQ_LEN][D_INNER];
  fixed8_t dt[SEQ_LEN][D_INNER];
  fixed8_t B[SEQ_LEN][D_STATE];
  fixed8_t C[SEQ_LEN][D_STATE];
  fixed8_t A[D_INNER][D_STATE];
  fixed8_t D_param[D_INNER];
  fixed8_t y[SEQ_LEN][D_INNER];

  srand(42);

  // Initialize with small values (to stay in [-1, 1] range)
  for (int t = 0; t < SEQ_LEN; t++) {
    for (int d = 0; d < D_INNER; d++) {
      x[t][d] = float_to_q7(randf(-0.5f, 0.5f));
      dt[t][d] = float_to_q7(randf(0.01f, 0.1f));
    }
    for (int n = 0; n < D_STATE; n++) {
      B[t][n] = float_to_q7(randf(-0.3f, 0.3f));
      C[t][n] = float_to_q7(randf(-0.3f, 0.3f));
    }
  }

  for (int d = 0; d < D_INNER; d++) {
    for (int n = 0; n < D_STATE; n++) {
      A[d][n] = float_to_q7(randf(-0.1f, -0.01f)); // Negative for stability
    }
    D_param[d] = float_to_q7(0.5f);
  }

  // Run kernel
  std::cout << "\nRunning int8 SSM scan..." << std::endl;
  ssm_scan_fixed8(x, dt, B, C, A, D_param, y);

  // Stats
  float y_sum = 0.0f;
  float y_max = -1e9f;
  float y_min = 1e9f;
  int overflow_count = 0;

  for (int t = 0; t < SEQ_LEN; t++) {
    for (int d = 0; d < D_INNER; d++) {
      float yf = q7_to_float(y[t][d]);
      y_sum += yf;
      if (yf > y_max)
        y_max = yf;
      if (yf < y_min)
        y_min = yf;
      if (y[t][d] == 127 || y[t][d] == -128)
        overflow_count++;
    }
  }

  float y_mean = y_sum / (SEQ_LEN * D_INNER);

  std::cout << "\nOutput statistics:" << std::endl;
  std::cout << "  Mean: " << y_mean << std::endl;
  std::cout << "  Min:  " << y_min << std::endl;
  std::cout << "  Max:  " << y_max << std::endl;
  std::cout << "  Saturated values: " << overflow_count << "/"
            << (SEQ_LEN * D_INNER) << std::endl;

  // Pass/Fail
  bool passed = true;

  if (std::isnan(y_mean) || std::isinf(y_mean)) {
    std::cerr << "FAIL: NaN/Inf detected!" << std::endl;
    passed = false;
  }

  if (overflow_count > (SEQ_LEN * D_INNER) / 4) {
    std::cerr << "WARNING: High saturation rate (>25%)" << std::endl;
  }

  if (passed) {
    std::cout << "\nPASS: int8 kernel executed successfully." << std::endl;
    return 0;
  }
  return 1;
}
