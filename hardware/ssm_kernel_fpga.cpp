/**
 * SSM Kernel - FPGA Production Ready
 *
 * Final architecture with:
 * - Persistent Register-File state (LUTRAM/FFs)
 * - Complete ARRAY_PARTITION for parallel access
 * - Streamed weights via AXI
 * - Hadamard transform ready for scaling
 * - II=1 target for single-cycle throughput
 */

#ifndef SSM_KERNEL_FPGA_H
#define SSM_KERNEL_FPGA_H

#include <cmath>
#include <cstdint>

// Full dimensions matching Python model
constexpr int D_INNER = 128;
constexpr int D_STATE = 16;
constexpr int STATE_BYTES = D_INNER * D_STATE; // 2048 bytes = 2KB

typedef int8_t narrow_t;
typedef int32_t wide_t;
constexpr int SCALE_SHIFT = 7;
constexpr narrow_t NARROW_MAX = 127;
constexpr narrow_t NARROW_MIN = -127;

// ============================================
// LFSR with verified non-zero seed
// ============================================
#define LFSR_SEED 0xACE1u // Non-zero seed (verified)

class LFSR {
private:
  uint16_t state;

public:
  LFSR() : state(LFSR_SEED) {}

  void reset() { state = LFSR_SEED; }

  uint16_t next() {
#pragma HLS INLINE
    uint16_t bit =
        ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1;
    state = (state >> 1) | (bit << 15);
    return state;
  }
};

// Global LFSR instance
static LFSR lfsr;

// ============================================
// Stochastic Rounding
// ============================================
inline narrow_t stochastic_round(wide_t val, int shift = SCALE_SHIFT) {
#pragma HLS INLINE
  wide_t mask = (1 << shift) - 1;
  wide_t fraction = val & mask;
  uint16_t threshold = lfsr.next() & mask;
  wide_t rounded = val + (fraction > threshold ? (1 << shift) : 0);
  rounded >>= shift;
  if (rounded > NARROW_MAX)
    return NARROW_MAX;
  if (rounded < NARROW_MIN)
    return NARROW_MIN;
  return (narrow_t)rounded;
}

// ============================================
// Delta Lookup Table (ROM)
// ============================================
constexpr int DT_LUT_SIZE = 256;
static narrow_t dt_lut[DT_LUT_SIZE];
#pragma HLS BIND_STORAGE variable = dt_lut type = rom_1p impl = lutram

inline void init_dt_lut(float dt_min = 0.001f, float dt_max = 0.1f) {
  for (int i = 0; i < DT_LUT_SIZE; i++) {
    float x = (float)(i - 128) / 128.0f;
    float dt = dt_min + (dt_max - dt_min) * (x + 1.0f) / 2.0f;
    dt_lut[i] = (narrow_t)(std::exp(-dt) * NARROW_MAX);
  }
}

// ============================================
// Hadamard Transform (for scaling beyond 128×16)
// ============================================
inline void hadamard_2(narrow_t &a, narrow_t &b) {
#pragma HLS INLINE
  wide_t sum = (wide_t)a + (wide_t)b;
  wide_t diff = (wide_t)a - (wide_t)b;
  a = stochastic_round(sum);
  b = stochastic_round(diff);
}

// Apply Hadamard to input (prevents outliers at scaling)
inline void hadamard_input(narrow_t x[D_INNER]) {
#pragma HLS INLINE
  // Fast Walsh-Hadamard on pairs
  for (int i = 0; i < D_INNER; i += 2) {
#pragma HLS UNROLL
    hadamard_2(x[i], x[i + 1]);
  }
}

// ============================================
// Persistent Register-File State
// ============================================
struct SSMState {
  // State lives in Flip-Flops (2KB)
  narrow_t h[D_INNER][D_STATE];
#pragma HLS ARRAY_PARTITION variable = h complete dim = 0

  void reset() {
#pragma HLS INLINE
    for (int d = 0; d < D_INNER; d++) {
#pragma HLS UNROLL
      for (int n = 0; n < D_STATE; n++) {
#pragma HLS UNROLL
        h[d][n] = 0;
      }
    }
  }
};

// ============================================
// SSM Step - Production Kernel (II=1 target)
// ============================================
void ssm_step_fpga(narrow_t x_byte, SSMState &state,
                   const narrow_t A[D_INNER][D_STATE],
                   const narrow_t B[D_STATE], const narrow_t C[D_STATE],
                   const narrow_t D_param[D_INNER], uint8_t dt_idx,
                   narrow_t y_out[D_INNER]) {
// Target: II=1 (one byte per clock cycle)
#pragma HLS PIPELINE II = 1
#pragma HLS LATENCY min = 1 max = 10

// Partition all arrays for parallel access
#pragma HLS ARRAY_PARTITION variable = A complete dim = 0
#pragma HLS ARRAY_PARTITION variable = B complete
#pragma HLS ARRAY_PARTITION variable = C complete
#pragma HLS ARRAY_PARTITION variable = D_param complete
#pragma HLS ARRAY_PARTITION variable = y_out complete

  narrow_t dA_scale = dt_lut[dt_idx];

// All 128 dimensions computed in parallel
inner_loop:
  for (int d = 0; d < D_INNER; d++) {
#pragma HLS UNROLL

    wide_t y_acc = 0;

  // All 16 state dimensions in parallel
  state_loop:
    for (int n = 0; n < D_STATE; n++) {
#pragma HLS UNROLL

      wide_t h_wide = (wide_t)state.h[d][n];
      // dA is the decay factor (should be positive)
      wide_t A_abs = (A[d][n] < 0) ? -A[d][n] : A[d][n];
      wide_t dA = ((wide_t)dA_scale * A_abs) >> SCALE_SHIFT;
      dA = (dA > NARROW_MAX) ? NARROW_MAX
                             : (dA < 1 ? 1 : dA); // Min 1 for stability
      wide_t dB = ((wide_t)dA_scale * (wide_t)B[n]) >> SCALE_SHIFT;

      wide_t h_new = dA * h_wide + dB * (wide_t)x_byte;
      state.h[d][n] = stochastic_round(h_new);

      y_acc += (wide_t)C[n] * state.h[d][n];
    }

    wide_t skip = (wide_t)D_param[d] * (wide_t)x_byte;
    y_out[d] = stochastic_round(y_acc + skip);
  }
}

// ============================================
// Top-Level AXI-Stream Wrapper
// ============================================
#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_stream.h>

void ssm_top(hls::stream<ap_uint<8>> &input_stream,
             hls::stream<ap_int<8>> &output_stream,
             // Weights streamed from DDR via AXI
             const narrow_t A[D_INNER][D_STATE], const narrow_t B[D_STATE],
             const narrow_t C[D_STATE], const narrow_t D_param[D_INNER]) {
#pragma HLS INTERFACE axis port = input_stream
#pragma HLS INTERFACE axis port = output_stream
#pragma HLS INTERFACE m_axi port = A bundle = weights
#pragma HLS INTERFACE m_axi port = B bundle = weights
#pragma HLS INTERFACE m_axi port = C bundle = weights
#pragma HLS INTERFACE m_axi port = D_param bundle = weights
#pragma HLS INTERFACE ap_ctrl_hs port = return

  // Persistent state in Flip-Flops (never leaves fabric)
  static SSMState state;
#pragma HLS RESET variable = state.h

  narrow_t y[D_INNER];
#pragma HLS ARRAY_PARTITION variable = y complete

  // Continuous stream processing
  while (!input_stream.empty()) {
#pragma HLS PIPELINE II = 1

    ap_uint<8> in_byte = input_stream.read();
    narrow_t x = (narrow_t)(in_byte - 128);

    uint8_t dt_idx = 128; // Could be made dynamic

    ssm_step_fpga(x, state, A, B, C, D_param, dt_idx, y);

    // Output first dimension (or reduce)
    output_stream.write(y[0]);
  }
}
#endif

#endif // SSM_KERNEL_FPGA_H
