/**
 * Stochastic Rounding Testbench
 *
 * Compares standard rounding vs stochastic rounding over long sequences.
 */

#include "ssm_kernel_sr.cpp"
#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
  std::cout << "Stochastic Rounding Testbench" << std::endl;
  std::cout << "=============================" << std::endl;

  init_dt_lut(0.001f, 0.1f);

  // Initialize weights
  narrow_t A[D_INNER][D_STATE];
  narrow_t B[D_STATE];
  narrow_t C[D_STATE];
  narrow_t D_param[D_INNER];
  narrow_t h_std[D_INNER][D_STATE] = {0};
  narrow_t h_sr[D_INNER][D_STATE] = {0};
  narrow_t y_std[D_INNER], y_sr[D_INNER];

  srand(42);
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

  // Process 10,000 bytes (stress test)
  const int NUM_BYTES = 10000;
  std::cout << "\nProcessing " << NUM_BYTES << " bytes..." << std::endl;

  double std_sum = 0, sr_sum = 0;
  int std_sat = 0, sr_sat = 0;

  for (int t = 0; t < NUM_BYTES; t++) {
    float sine = std::sin(t * 0.05f);
    narrow_t x = (narrow_t)(sine * 80);
    uint8_t dt_idx = 128 + (uint8_t)(sine * 40);

    // Run both versions
    ssm_step_sr(x, h_std, A, B, C, D_param, dt_idx, y_std, false);
    ssm_step_sr(x, h_sr, A, B, C, D_param, dt_idx, y_sr, true);

    for (int d = 0; d < D_INNER; d++) {
      std_sum += y_std[d];
      sr_sum += y_sr[d];
      if (y_std[d] == 127 || y_std[d] == -127)
        std_sat++;
      if (y_sr[d] == 127 || y_sr[d] == -127)
        sr_sat++;
    }
  }

  // Compare state energies
  double h_std_energy = 0, h_sr_energy = 0;
  for (int d = 0; d < D_INNER; d++) {
    for (int n = 0; n < D_STATE; n++) {
      h_std_energy += std::abs(h_std[d][n]);
      h_sr_energy += std::abs(h_sr[d][n]);
    }
  }

  std::cout << "\nResults:" << std::endl;
  std::cout << "  Standard Rounding:" << std::endl;
  std::cout << "    Output sum:    " << std_sum << std::endl;
  std::cout << "    State energy:  " << h_std_energy << std::endl;
  std::cout << "    Saturated:     " << std_sat << std::endl;

  std::cout << "  Stochastic Rounding:" << std::endl;
  std::cout << "    Output sum:    " << sr_sum << std::endl;
  std::cout << "    State energy:  " << h_sr_energy << std::endl;
  std::cout << "    Saturated:     " << sr_sat << std::endl;

  // Check for bias (SR should have sum closer to 0)
  double std_bias = std::abs(std_sum / (NUM_BYTES * D_INNER));
  double sr_bias = std::abs(sr_sum / (NUM_BYTES * D_INNER));

  std::cout << "\n  Avg bias (std):  " << std_bias << std::endl;
  std::cout << "  Avg bias (SR):   " << sr_bias << std::endl;

  if (sr_sat <= std_sat && h_sr_energy > 0) {
    std::cout << "\n✓ PASS: Stochastic rounding working" << std::endl;
    return 0;
  }
  return 1;
}
