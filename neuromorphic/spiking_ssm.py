"""
Spiking SSM - Neuromorphic Version

Maps the Liquid Mamba SSM to a Spiking Neural Network using Intel's Lava framework.
Can run on Loihi 2 or in simulation mode.

The SSM recurrence h_t = dA * h_{t-1} + dB * x_t maps to:
- h_t → Leaky Integrate-and-Fire (LIF) neuron membrane potential
- dA → Leak factor (decay)
- dB * x_t → Input current

Note: This is a conceptual prototype. Full Loihi 2 deployment requires hardware access.
"""

try:
    from lava.proc.lif.process import LIF
    from lava.proc.dense.process import Dense
    from lava.proc.io.source import RingBuffer as Source
    from lava.proc.io.sink import RingBuffer as Sink
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi2SimCfg
    LAVA_AVAILABLE = True
except ImportError:
    LAVA_AVAILABLE = False
    print("Lava not installed. Run: pip install lava-nc")

import numpy as np

# Configuration
D_INNER = 32  # Reduced for simulation
D_STATE = 8
SEQ_LEN = 64

def create_spiking_ssm():
    """
    Create a spiking neural network that emulates SSM dynamics.
    """
    if not LAVA_AVAILABLE:
        print("Lava not available. Exiting.")
        return

    # Input source (simulated byte stream)
    input_data = np.random.randint(0, 256, size=(SEQ_LEN, D_INNER)).astype(np.float32)
    input_data = (input_data - 128) / 128  # Normalize to [-1, 1]

    source = Source(data=input_data)

    # Input projection (like B matrix in SSM)
    B_weights = np.random.randn(D_STATE, D_INNER).astype(np.float32) * 0.1
    input_proj = Dense(weights=B_weights)

    # LIF neurons representing SSM state
    # du = leak rate (maps to dA decay)
    # vth = threshold for spiking
    state_neurons = LIF(
        shape=(D_STATE,),
        du=0.1,    # Decay factor (higher = more leak = faster forget)
        dv=0.1,    # Voltage decay
        vth=1.0,   # Spike threshold
        bias_mant=0,
        bias_exp=0
    )

    # Output projection (like C matrix in SSM)
    C_weights = np.random.randn(D_INNER, D_STATE).astype(np.float32) * 0.1
    output_proj = Dense(weights=C_weights)

    # Output sink
    sink = Sink(shape=(D_INNER,), buffer=SEQ_LEN)

    # Connect the network
    source.s_out.connect(input_proj.s_in)
    input_proj.a_out.connect(state_neurons.a_in)
    state_neurons.s_out.connect(output_proj.s_in)
    output_proj.a_out.connect(sink.a_in)

    return source, state_neurons, sink

def run_simulation():
    """
    Run the spiking SSM simulation.
    """
    if not LAVA_AVAILABLE:
        print("=" * 40)
        print("LAVA SDK NOT INSTALLED")
        print("This is a conceptual prototype.")
        print("Install with: pip install lava-nc")
        print("=" * 40)

        # Fallback: numpy simulation of the concept
        print("\nRunning NumPy simulation instead...\n")

        # Simulate LIF dynamics
        input_data = np.random.randn(SEQ_LEN, D_INNER).astype(np.float32)
        B = np.random.randn(D_STATE, D_INNER).astype(np.float32) * 0.1
        C = np.random.randn(D_INNER, D_STATE).astype(np.float32) * 0.1

        # LIF state
        v = np.zeros(D_STATE)  # Membrane potential
        du = 0.1  # Leak
        threshold = 1.0

        outputs = []
        spike_counts = np.zeros(D_STATE)

        for t in range(SEQ_LEN):
            # Input current
            current = B @ input_data[t]

            # LIF update: v = v * (1 - du) + current
            v = v * (1 - du) + current

            # Spike detection
            spikes = (v >= threshold).astype(np.float32)
            spike_counts += spikes

            # Reset after spike
            v = v * (1 - spikes)

            # Output
            out = C @ spikes
            outputs.append(out)

        outputs = np.array(outputs)

        print("Simulation complete!")
        print(f"  Time steps: {SEQ_LEN}")
        print(f"  Total spikes: {int(spike_counts.sum())}")
        print(f"  Avg spikes/neuron: {spike_counts.mean():.2f}")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Output mean: {outputs.mean():.4f}")

        return outputs

    # Lava simulation
    source, state_neurons, sink = create_spiking_ssm()

    print("Running Lava simulation...")
    run_cfg = Loihi2SimCfg()
    state_neurons.run(condition=RunSteps(num_steps=SEQ_LEN), run_cfg=run_cfg)

    output_data = sink.data.get()
    state_neurons.stop()

    print("Simulation complete!")
    print(f"Output shape: {output_data.shape}")

    return output_data

if __name__ == "__main__":
    print("Spiking SSM - Neuromorphic Prototype")
    print("=" * 40)
    output = run_simulation()
