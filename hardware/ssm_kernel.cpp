/**
 * SSM Selective Scan Kernel - C++ Reference Implementation
 *
 * This is a hardware-synthesizable implementation of the core SSM logic.
 * Designed for High-Level Synthesis (HLS) targeting FPGAs.
 *
 * Uses fixed-point arithmetic for hardware compatibility.
 */

#ifndef SSM_KERNEL_H
#define SSM_KERNEL_H

#include <cmath>
#include <cstdint>

// Configuration (match Python model)
constexpr int D_INNER = 128;   // d_model * expand
constexpr int D_STATE = 16;    // SSM state dimension
constexpr int SEQ_LEN = 128;   // Sequence length

// Fixed-point type (Q16.16 format for precision)
typedef int32_t fixed_t;
constexpr int FRAC_BITS = 16;

inline fixed_t float_to_fixed(float x) {
    return static_cast<fixed_t>(x * (1 << FRAC_BITS));
}

inline float fixed_to_float(fixed_t x) {
    return static_cast<float>(x) / (1 << FRAC_BITS);
}

inline fixed_t fixed_mul(fixed_t a, fixed_t b) {
    int64_t temp = static_cast<int64_t>(a) * b;
    return static_cast<fixed_t>(temp >> FRAC_BITS);
}

// Softplus approximation (used for dt)
inline fixed_t softplus_fixed(fixed_t x) {
    // softplus(x) = log(1 + exp(x))
    // Approximation: for x > 4, softplus(x) ≈ x
    const fixed_t THRESHOLD = float_to_fixed(4.0f);
    if (x > THRESHOLD) {
        return x;
    }
    // For small x, use exp(x) approximation
    // exp(x) ≈ 1 + x for small x
    float xf = fixed_to_float(x);
    float result = std::log(1.0f + std::exp(xf));
    return float_to_fixed(result);
}

/**
 * SSM Selective Scan
 *
 * Computes: h_t = dA * h_{t-1} + dB * x_t
 *           y_t = C_t * h_t
 *
 * Where:
 *   dA = exp(dt * A)
 *   dB = dt * B
 */
void ssm_scan(
    const fixed_t x[SEQ_LEN][D_INNER],      // Input (after conv)
    const fixed_t dt[SEQ_LEN][D_INNER],     // Delta time (after softplus)
    const fixed_t B[SEQ_LEN][D_STATE],      // B matrix
    const fixed_t C[SEQ_LEN][D_STATE],      // C matrix
    const fixed_t A[D_INNER][D_STATE],      // A matrix (log-space, negative)
    const fixed_t D_param[D_INNER],         // D skip connection
    fixed_t y[SEQ_LEN][D_INNER]             // Output
) {
    // Hidden state (persistent across time)
    fixed_t h[D_INNER][D_STATE] = {0};

    // Scan loop (sequential in time)
    for (int t = 0; t < SEQ_LEN; t++) {
        // For each inner dimension
        for (int d = 0; d < D_INNER; d++) {
            fixed_t y_t = 0;

            // For each state dimension
            for (int n = 0; n < D_STATE; n++) {
                // Discretize A: dA = exp(dt * A)
                // Since A is stored as log(-A), we compute exp(dt * (-exp(A)))
                // Simplified: dA ≈ exp(-dt * exp_A) where exp_A = exp(A_log)
                // For HLS, we pre-compute exp(A) or use lookup tables.

                // Approximation: dA = 1 - dt * |A| (first-order Taylor)
                fixed_t dt_val = dt[t][d];
                fixed_t A_val = A[d][n]; // Stored as -exp(A_log), i.e., negative

                // dA = exp(dt * A) ≈ 1 + dt * A (for small dt * A)
                fixed_t dA = float_to_fixed(1.0f) + fixed_mul(dt_val, A_val);

                // dB = dt * B
                fixed_t dB = fixed_mul(dt_val, B[t][n]);

                // Update state: h = dA * h + dB * x
                h[d][n] = fixed_mul(dA, h[d][n]) + fixed_mul(dB, x[t][d]);

                // Output: y += C * h
                y_t += fixed_mul(C[t][n], h[d][n]);
            }

            // Add skip connection: y = y + D * x
            y[t][d] = y_t + fixed_mul(D_param[d], x[t][d]);
        }
    }
}

#endif // SSM_KERNEL_H
