"""
drift_test_sr.py - Extended Drift Test with Stochastic Rounding Simulation

Tests correlation and MSE over 10K+ bytes to verify stability.
"""

import numpy as np

def run_sr_drift_test(n_bytes=10000):
    """Extended drift test simulating stochastic rounding."""
    print(f"Extended Drift Test ({n_bytes} bytes)")
    print("=" * 50)

    d_inner, d_state = 128, 16

    # Initialize
    h_std = np.zeros((d_inner, d_state), dtype=np.float32)  # Python float
    h_sr = np.zeros((d_inner, d_state), dtype=np.int8)       # C++ with SR

    # Weights
    A = np.full((d_inner, d_state), -3, dtype=np.int8)
    B = np.full(d_state, 25, dtype=np.int8)
    C = np.full(d_state, 25, dtype=np.int8)
    D = np.full(d_inner, 64, dtype=np.int8)
    dt_scale = 50

    py_outputs = []
    sr_outputs = []

    # LFSR state
    lfsr = 0xACE1

    def lfsr_next():
        nonlocal lfsr
        bit = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1
        lfsr = (lfsr >> 1) | (bit << 15)
        return lfsr

    def stochastic_round(val, shift):
        mask = (1 << shift) - 1
        fraction = int(val) & mask
        threshold = lfsr_next() & mask
        rounded = (val + (mask + 1 if fraction > threshold else 0)) >> shift
        return np.clip(rounded, -127, 127).astype(np.int8)

    # Process bytes
    for t in range(n_bytes):
        sine = np.sin(t * 0.05)
        x = int(sine * 80)
        x_float = sine * 80 / 127.0  # Normalized

        # Python float version
        for d in range(d_inner):
            for n in range(d_state):
                dA = np.exp(0.05 * A[d, n] / 127.0)
                dB = 0.05 * B[n] / 127.0
                h_std[d, n] = dA * h_std[d, n] + dB * x_float
        py_out = np.sum(h_std) * 0.01 + 0.5 * x_float  # Simplified output
        py_outputs.append(py_out)

        # C++ SR version
        for d in range(d_inner):
            y_acc = 0
            for n in range(d_state):
                dA = (int(dt_scale) * int(A[d, n])) >> 7
                dA = np.clip(dA, -127, 127)
                dB = (int(dt_scale) * int(B[n])) >> 7

                h_new = int(dA) * int(h_sr[d, n]) + int(dB) * x
                h_sr[d, n] = stochastic_round(h_new, 7)  # SR instead of truncation

                y_acc += int(C[n]) * int(h_sr[d, n])

            skip = (int(D[d]) * x) >> 7
            y_total = (y_acc >> 7) + skip
        sr_outputs.append(stochastic_round(y_total, 0))

    # Metrics
    py_out = np.array(py_outputs)
    sr_out = np.array(sr_outputs, dtype=np.float32)

    # Normalize
    py_norm = py_out / (np.max(np.abs(py_out)) + 1e-8)
    sr_norm = sr_out / 127.0

    mse = np.mean((py_norm - sr_norm) ** 2)
    corr = np.corrcoef(py_norm.flatten(), sr_norm.flatten())[0, 1]

    print(f"\nResults ({n_bytes} bytes):")
    print(f"  MSE:         {mse:.6f}")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Py state energy:  {np.sum(h_std**2):.2f}")
    print(f"  SR state energy:  {np.sum(h_sr.astype(np.float32)**2):.2f}")

    if mse < 0.05:
        print(f"\n✓ PASS: MSE below 0.05 threshold")
    elif mse < 0.17:
        print(f"\n⚠ IMPROVED: MSE reduced from 0.17")
    else:
        print(f"\n✗ MSE still high")

    if corr > 0.999:
        print(f"✓ State-Stable: Correlation > 0.999")

    return mse, corr

if __name__ == "__main__":
    run_sr_drift_test(10000)
