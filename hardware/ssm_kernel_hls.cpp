/**
 * SSM Selective Scan Kernel - HLS Version
 *
 * This version includes Xilinx Vitis HLS pragmas for FPGA synthesis.
 * Target: AMD/Xilinx FPGAs (Alveo, Zynq, etc.)
 */

#include "ssm_kernel.cpp"

// Top-level function for HLS synthesis
void ssm_scan_hls(const fixed_t x[SEQ_LEN][D_INNER],
                  const fixed_t dt[SEQ_LEN][D_INNER],
                  const fixed_t B[SEQ_LEN][D_STATE],
                  const fixed_t C[SEQ_LEN][D_STATE],
                  const fixed_t A[D_INNER][D_STATE],
                  const fixed_t D_param[D_INNER], fixed_t y[SEQ_LEN][D_INNER]) {
// HLS Interface pragmas
#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = dt offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = D_param offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = y offset = slave bundle = gmem3
#pragma HLS INTERFACE s_axilite port = return bundle = control

  // Local buffers for on-chip processing
  fixed_t x_local[SEQ_LEN][D_INNER];
  fixed_t dt_local[SEQ_LEN][D_INNER];
  fixed_t B_local[SEQ_LEN][D_STATE];
  fixed_t C_local[SEQ_LEN][D_STATE];
  fixed_t A_local[D_INNER][D_STATE];
  fixed_t D_local[D_INNER];
  fixed_t y_local[SEQ_LEN][D_INNER];

// Partition arrays for parallel access
#pragma HLS ARRAY_PARTITION variable = A_local dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = D_local dim = 1 complete

  // Hidden state (on-chip BRAM)
  fixed_t h[D_INNER][D_STATE];
#pragma HLS ARRAY_PARTITION variable = h dim = 2 complete

// Initialize hidden state
init_h:
  for (int d = 0; d < D_INNER; d++) {
#pragma HLS UNROLL
    for (int n = 0; n < D_STATE; n++) {
#pragma HLS UNROLL
      h[d][n] = 0;
    }
  }

// Load A and D (static weights)
load_A:
  for (int d = 0; d < D_INNER; d++) {
    for (int n = 0; n < D_STATE; n++) {
      A_local[d][n] = A[d][n];
    }
    D_local[d] = D_param[d];
  }

// Main scan loop (sequential in time, pipelined per step)
scan_loop:
  for (int t = 0; t < SEQ_LEN; t++) {
  // Load time-step data
  load_t:
    for (int d = 0; d < D_INNER; d++) {
#pragma HLS PIPELINE II = 1
      x_local[t][d] = x[t][d];
      dt_local[t][d] = dt[t][d];
    }
    for (int n = 0; n < D_STATE; n++) {
      B_local[t][n] = B[t][n];
      C_local[t][n] = C[t][n];
    }

  // Compute inner dimensions in parallel
  inner_loop:
    for (int d = 0; d < D_INNER; d++) {
#pragma HLS PIPELINE II = 1

      fixed_t y_t = 0;

    state_loop:
      for (int n = 0; n < D_STATE; n++) {
#pragma HLS UNROLL

        fixed_t dt_val = dt_local[t][d];
        fixed_t A_val = A_local[d][n];

        // Discretize A (first-order approximation)
        fixed_t dA = float_to_fixed(1.0f) + fixed_mul(dt_val, A_val);
        fixed_t dB = fixed_mul(dt_val, B_local[t][n]);

        // State update
        h[d][n] = fixed_mul(dA, h[d][n]) + fixed_mul(dB, x_local[t][d]);

        // Output accumulation
        y_t += fixed_mul(C_local[t][n], h[d][n]);
      }

      // Skip connection
      y_local[t][d] = y_t + fixed_mul(D_local[d], x_local[t][d]);
    }
  }

// Write output
write_y:
  for (int t = 0; t < SEQ_LEN; t++) {
    for (int d = 0; d < D_INNER; d++) {
#pragma HLS PIPELINE II = 1
      y[t][d] = y_local[t][d];
    }
  }
}
