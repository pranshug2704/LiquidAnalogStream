"""
latency_test.py - Time-to-First-Byte vs Time-to-Last-Byte

Tests the constant-time property of the Liquid SSM.
In a Transformer, later bytes take longer (O(L) attention).
In Mamba/SSM, every byte should take ~same time (O(1) per step).
"""

import time
import torch
import numpy as np
from model import LiquidStreamModel, ModelArgs

def measure_latency(model, input_bytes: list, num_warmup: int = 5):
    """Measure per-byte generation latency."""
    model.eval()
    context = torch.tensor([input_bytes], dtype=torch.long)

    latencies = []

    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            _ = model(context)

        # Measure
        for i in range(100):
            start = time.perf_counter()
            logits = model(context)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            end = time.perf_counter()

            latencies.append(end - start)

            context = torch.cat([context, next_token], dim=1)

    return latencies

def stress_test():
    """Run latency stress test."""
    print("Latency Stress Test")
    print("=" * 50)

    args = ModelArgs(d_model=64, n_layer=2)
    model = LiquidStreamModel(args)

    try:
        model.load_state_dict(torch.load('model.pt', map_location='cpu', weights_only=True))
        print("Loaded trained model")
    except:
        print("Using random weights")

    # Initial context
    initial = list(b"val: 0.123\n")

    print(f"\nMeasuring 100 tokens generation...")
    latencies = measure_latency(model, initial)

    # Split into first 10, middle, last 10
    first_10 = latencies[:10]
    last_10 = latencies[-10:]

    print(f"\nResults:")
    print(f"  Time to FIRST byte:    {latencies[0]*1000:.2f} ms")
    print(f"  Time to LAST byte:     {latencies[-1]*1000:.2f} ms")
    print(f"  First 10 avg:          {np.mean(first_10)*1000:.2f} ms")
    print(f"  Last 10 avg:           {np.mean(last_10)*1000:.2f} ms")
    print(f"  Overall avg:           {np.mean(latencies)*1000:.2f} ms")
    print(f"  Overall std:           {np.std(latencies)*1000:.2f} ms")

    # Ratio check (should be close to 1.0 for constant-time)
    ratio = np.mean(last_10) / np.mean(first_10)
    print(f"\n  Last/First ratio:      {ratio:.2f}x")

    if ratio < 1.5:
        print("  ✓ PASS: Near-constant latency (< 1.5x slowdown)")
    else:
        print("  ⚠ WARNING: Latency increased significantly")

    # Note about context growth
    print(f"\n[Note: Context grew from {len(initial)} to {len(initial)+100} tokens]")
    print("[In current implementation, we recompute full sequence each step]")
    print("[True SSM would maintain hidden state for O(1) per-step]")

if __name__ == "__main__":
    stress_test()
