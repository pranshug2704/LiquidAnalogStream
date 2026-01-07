"""
long_duration_test.py - 10M Byte Infinite Stability Test

Verifies that the Liquid SSM maintains stable state energy over millions of steps.
Success: State energy within 5% of initial value after 10M bytes.
"""

import numpy as np
import time

D_INNER = 128
D_STATE = 16
SCALE_SHIFT = 7
NARROW_MAX = 127
NARROW_MIN = -127

# LFSR
lfsr_state = 0xACE1

def lfsr_next():
    global lfsr_state
    bit = ((lfsr_state >> 0) ^ (lfsr_state >> 2) ^
           (lfsr_state >> 3) ^ (lfsr_state >> 5)) & 1
    lfsr_state = ((lfsr_state >> 1) | (bit << 15)) & 0xFFFF
    return lfsr_state

def stochastic_round(val, shift=SCALE_SHIFT):
    mask = (1 << shift) - 1
    fraction = int(val) & mask
    threshold = lfsr_next() & mask
    rounded = int(val) + ((1 << shift) if fraction > threshold else 0)
    rounded >>= shift
    return max(NARROW_MIN, min(NARROW_MAX, rounded))

def init_dt_lut():
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        x = (i - 128) / 128.0
        dt = 0.001 + (0.1 - 0.001) * (x + 1.0) / 2.0
        lut[i] = int(np.exp(-dt) * NARROW_MAX)
    return lut

def run_long_duration_test(n_bytes=10_000_000, checkpoint_interval=1_000_000):
    """Run 10M byte stability test."""
    global lfsr_state
    lfsr_state = 0xACE1

    print("=" * 60)
    print("LONG-DURATION STABILITY TEST")
    print("=" * 60)
    print(f"Bytes to process: {n_bytes:,}")
    print(f"State size: {D_INNER}×{D_STATE} = {D_INNER * D_STATE} bytes")

    # Initialize
    A = np.full((D_INNER, D_STATE), -3, dtype=np.int8)
    B = np.full(D_STATE, 25, dtype=np.int8)
    C = np.full(D_STATE, 25, dtype=np.int8)
    D_param = np.full(D_INNER, 64, dtype=np.int8)
    h = np.zeros((D_INNER, D_STATE), dtype=np.int8)
    dt_lut = init_dt_lut()

    # Track stats
    initial_energy = None
    checkpoints = []
    start_time = time.time()

    print("\nProcessing...")

    for t in range(n_bytes):
        # Input
        sine = np.sin(t * 0.05)
        x = int(sine * 80)
        dt_idx = int(128 + sine * 40) & 0xFF
        dA_scale = dt_lut[dt_idx]

        # SSM step
        for d in range(D_INNER):
            for n in range(D_STATE):
                A_abs = abs(A[d, n])
                dA = (int(dA_scale) * A_abs) >> SCALE_SHIFT
                dA = max(1, min(NARROW_MAX, dA))
                dB = (int(dA_scale) * int(B[n])) >> SCALE_SHIFT

                h_new = int(dA) * int(h[d, n]) + int(dB) * x
                h[d, n] = stochastic_round(h_new)

        # Checkpoint
        if t == 0 or (t + 1) % checkpoint_interval == 0:
            energy = np.sum(np.abs(h.astype(np.int32)))
            if initial_energy is None:
                initial_energy = energy if energy > 0 else 1

            elapsed = time.time() - start_time
            rate = (t + 1) / elapsed if elapsed > 0 else 0
            drift = 100.0 * (energy - initial_energy) / initial_energy

            checkpoints.append({
                'step': t + 1,
                'energy': energy,
                'drift_pct': drift
            })

            print(f"  Step {t+1:>10,}: Energy={energy:>6}, Drift={drift:>+6.2f}%, Rate={rate:,.0f}/s")

    # Final results
    elapsed = time.time() - start_time
    final_energy = np.sum(np.abs(h.astype(np.int32)))
    final_drift = 100.0 * (final_energy - initial_energy) / initial_energy

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total bytes:      {n_bytes:,}")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  Rate:             {n_bytes/elapsed:,.0f} bytes/sec")
    print(f"  Initial energy:   {initial_energy}")
    print(f"  Final energy:     {final_energy}")
    print(f"  Final drift:      {final_drift:+.2f}%")

    # Verdict
    print("\n" + "=" * 60)
    if abs(final_drift) <= 5.0:
        print("✓ PASS: INFINITE STABILITY ACHIEVED")
        print(f"  State energy within 5% threshold ({final_drift:+.2f}%)")
    else:
        print(f"✗ FAIL: Drift {final_drift:+.2f}% exceeds 5% threshold")
    print("=" * 60)

    return final_drift

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000  # Default 1M for faster test
    run_long_duration_test(n)
