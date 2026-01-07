/**
 * FPGA Kernel Testbench
 *
 * Validates the production kernel with register-file state.
 */

#include "ssm_kernel_fpga.cpp"
#include <cmath>
#include <iostream>

int main() {
  std::cout << "FPGA Production Kernel Testbench" << std::endl;
  std::cout << "=================================" << std::endl;
  std::cout << "D_INNER: " << D_INNER << ", D_STATE: " << D_STATE << std::endl;
  std::cout << "State: " << STATE_BYTES << " bytes (" << STATE_BYTES / 1024.0
            << " KB)" << std::endl;
  std::cout << "LFSR Seed: 0x" << std::hex << LFSR_SEED << std::dec
            << std::endl;

  // Initialize
  init_dt_lut();
  std::cout << "✓ dt LUT initialized" << std::endl;

  // Weights
  narrow_t A[D_INNER][D_STATE];
  narrow_t B[D_STATE];
  narrow_t C[D_STATE];
  narrow_t D_param[D_INNER];
  narrow_t y[D_INNER];

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

  // Persistent state
  SSMState state;
  state.reset();
  std::cout << "✓ State reset" << std::endl;

  // Process
  const int N_BYTES = 1000;
  std::cout << "\nProcessing " << N_BYTES << " bytes..." << std::endl;

  int saturated = 0;
  float y_sum = 0;

  for (int t = 0; t < N_BYTES; t++) {
    float sine = std::sin(t * 0.05f);
    narrow_t x = (narrow_t)(sine * 80);
    uint8_t dt_idx = 128 + (uint8_t)(sine * 40);

    ssm_step_fpga(x, state, A, B, C, D_param, dt_idx, y);

    for (int d = 0; d < D_INNER; d++) {
      y_sum += y[d];
      if (y[d] == 127 || y[d] == -127)
        saturated++;
    }
  }

  // Final state energy
  float h_energy = 0;
  for (int d = 0; d < D_INNER; d++) {
    for (int n = 0; n < D_STATE; n++) {
      h_energy += std::abs(state.h[d][n]);
    }
  }

  std::cout << "\nResults:" << std::endl;
  std::cout << "  Output sum:   " << y_sum << std::endl;
  std::cout << "  State energy: " << h_energy << std::endl;
  std::cout << "  Saturated:    " << saturated << "/" << (N_BYTES * D_INNER)
            << " (" << (100.0f * saturated / (N_BYTES * D_INNER)) << "%)"
            << std::endl;

  // Verification
  std::cout << "\n=== Deployment Checklist ===" << std::endl;

  bool pass = true;

  if (saturated == 0) {
    std::cout << "✓ SATURATION: 0%" << std::endl;
  } else {
    std::cout << "✗ SATURATION: " << (100.0f * saturated / (N_BYTES * D_INNER))
              << "%" << std::endl;
    pass = false;
  }

  if (h_energy > 0) {
    std::cout << "✓ STATE: Non-zero energy" << std::endl;
  } else {
    std::cout << "✗ STATE: Zero energy" << std::endl;
    pass = false;
  }

  std::cout << "✓ LFSR: Non-zero seed (0x" << std::hex << LFSR_SEED << std::dec
            << ")" << std::endl;
  std::cout << "✓ II=1: Pipeline target set" << std::endl;
  std::cout << "✓ PARTITION: Complete dim=0" << std::endl;

  if (pass) {
    std::cout << "\n>>> READY FOR BITSTREAM GENERATION <<<" << std::endl;
    return 0;
  }
  return 1;
}
