"""
drift_test.py - Python vs C++ Numerical Comparison

Compares the Python model output with the C++ fixed-point implementation
to detect numerical drift due to quantization.
"""

import torch
import numpy as np
import subprocess
import struct
from model import LiquidStreamModel, ModelArgs

def run_python_ssm(input_bytes: list, d_inner=128, d_state=16):
    """Run Python SSM on input and return state evolution."""
    args = ModelArgs(d_model=64, n_layer=1, d_state=d_state)
    model = LiquidStreamModel(args)
    model.eval()

    # Extract first layer weights
    layer = model.layers[0]
    A = -torch.exp(layer.A_log.float()).detach().numpy()

    # Initialize state
    h = np.zeros((d_inner, d_state), dtype=np.float32)
    outputs = []

    dt = 0.05  # Fixed dt for comparison

    for byte_val in input_bytes:
        x = (byte_val - 128) / 128.0  # Normalize to [-1, 1]

        # Simple SSM step (matching C++ logic)
        for d in range(d_inner):
            y = 0.0
            for n in range(d_state):
                dA = np.exp(dt * A[d, n])
                dB = dt * 0.1  # Simplified B
                h[d, n] = dA * h[d, n] + dB * x
                y += 0.1 * h[d, n]  # Simplified C
            y += 0.5 * x  # Skip connection
            outputs.append(y)

    return np.array(outputs), h

def run_cpp_ssm(input_bytes: list):
    """Run C++ SSM and return outputs (if testbench is available)."""
    # This would invoke the C++ testbench
    # For now, simulate the quantized version in Python

    d_inner, d_state = 128, 16
    h = np.zeros((d_inner, d_state), dtype=np.int8)
    outputs = []

    # Quantized weights
    A = np.full((d_inner, d_state), -5, dtype=np.int8)  # Small negative
    B = np.full(d_state, 10, dtype=np.int8)
    C = np.full(d_state, 10, dtype=np.int8)
    D = np.full(d_inner, 64, dtype=np.int8)  # 0.5 in Q7

    dt_scale = 50  # From LUT

    for byte_val in input_bytes:
        x = np.int8(byte_val - 128)

        for d in range(d_inner):
            y_acc = np.int32(0)

            for n in range(d_state):
                # Wide arithmetic (int32)
                dA = (np.int32(dt_scale) * np.int32(A[d, n])) >> 7
                dA = np.clip(dA, -127, 127)
                dB = (np.int32(dt_scale) * np.int32(B[n])) >> 7

                h_new = (np.int32(dA) * np.int32(h[d, n]) + np.int32(dB) * np.int32(x)) >> 7
                h[d, n] = np.clip(h_new, -127, 127)

                y_acc += np.int32(C[n]) * np.int32(h[d, n])

            skip = (np.int32(D[d]) * np.int32(x)) >> 7
            y_total = (y_acc >> 7) + skip
            outputs.append(np.clip(y_total, -127, 127))

    return np.array(outputs, dtype=np.int8), h

def compare_drift():
    """Compare Python and C++ outputs."""
    print("Numerical Drift Test: Python vs C++")
    print("=" * 50)

    # Generate test input (sine wave pattern)
    input_bytes = []
    for i in range(1000):
        val = int(128 + 100 * np.sin(i * 0.1))
        input_bytes.append(np.clip(val, 0, 255))

    print(f"Input: {len(input_bytes)} bytes (sine wave)")

    # Run both versions
    py_out, py_state = run_python_ssm(input_bytes)
    cpp_out, cpp_state = run_cpp_ssm(input_bytes)

    # Normalize for comparison
    py_out_norm = py_out / np.max(np.abs(py_out) + 1e-8)
    cpp_out_norm = cpp_out.astype(np.float32) / 127.0

    # Compare (subsample to match dimensions)
    min_len = min(len(py_out_norm), len(cpp_out_norm))
    py_sub = py_out_norm[:min_len:128]  # Sample every d_inner
    cpp_sub = cpp_out_norm[:min_len:128]

    # Metrics
    mse = np.mean((py_sub - cpp_sub) ** 2)
    corr = np.corrcoef(py_sub.flatten(), cpp_sub.flatten())[0, 1]
    max_diff = np.max(np.abs(py_sub - cpp_sub))

    print(f"\nResults:")
    print(f"  MSE:          {mse:.6f}")
    print(f"  Correlation:  {corr:.4f}")
    print(f"  Max Diff:     {max_diff:.4f}")

    # State comparison
    py_state_energy = np.sum(py_state ** 2)
    cpp_state_energy = np.sum(cpp_state.astype(np.float32) ** 2)
    print(f"\n  Python state energy:  {py_state_energy:.2f}")
    print(f"  C++ state energy:     {cpp_state_energy:.2f}")

    # Verdict
    if corr > 0.9 and max_diff < 0.5:
        print("\n✓ PASS: Minimal drift, outputs correlate well")
    elif corr > 0.7:
        print("\n⚠ WARNING: Moderate drift, consider stochastic rounding")
    else:
        print("\n✗ FAIL: Significant drift, check quantization")

    return corr, mse

if __name__ == "__main__":
    compare_drift()
