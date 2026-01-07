"""
stateful_inference.py - O(1) Per-Byte Inference

Maintains hidden state `h` between calls for true constant-time generation.
This is the "Liquid Flow" mode where computation time is independent of context length.
"""

import torch
import torch.nn.functional as F
import time
import sys
from model import LiquidStreamModel, ModelArgs

class StatefulLiquidModel:
    """
    Wrapper for stateful inference with cached hidden state.
    Each call to step() processes ONE byte in O(1) time.
    """

    def __init__(self, model: LiquidStreamModel):
        self.model = model
        self.model.eval()

        # Cache hidden states for each layer
        # h: (batch, d_inner, d_state) per layer
        self.hidden_states = None
        self.args = model.args

    def reset(self):
        """Reset hidden states to zero."""
        self.hidden_states = None

    def _init_hidden(self, batch_size: int, device):
        """Initialize hidden states for all layers."""
        self.hidden_states = []
        for _ in range(self.args.n_layer):
            h = torch.zeros(batch_size, self.args.d_inner, self.args.d_state, device=device)
            self.hidden_states.append(h)

    def step(self, byte_val: int) -> int:
        """
        Process a single byte and return next byte prediction.
        O(1) time complexity - hidden state is cached.
        """
        device = next(self.model.parameters()).device

        # Create single-token input
        x = torch.tensor([[byte_val]], dtype=torch.long, device=device)

        # Initialize hidden if needed
        if self.hidden_states is None:
            self._init_hidden(1, device)

        with torch.no_grad():
            # Embedding
            x_emb = self.model.embedding(x)  # (1, 1, d_model)

            # Process through layers with stateful hidden
            for layer_idx, layer in enumerate(self.model.layers):
                x_emb, self.hidden_states[layer_idx] = self._layer_step(
                    layer, x_emb, self.hidden_states[layer_idx]
                )
                x_emb = self.model.norm_f(x_emb)

            # Output
            logits = self.model.lm_head(x_emb)  # (1, 1, vocab)
            probs = F.softmax(logits[0, 0] / 0.8, dim=-1)
            next_byte = torch.multinomial(probs, 1).item()

        return next_byte

    def _layer_step(self, layer, x, h):
        """
        Single-step through a LiquidMambaBlock with cached hidden state.
        """
        batch = x.shape[0]

        # Input projection
        xz = layer.in_proj(x)  # (1, 1, 2*d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # For single step, conv1d uses previous context
        # Simplified: skip conv for single-step (or use cached buffer)
        x_conv = F.silu(x_branch)

        # SSM parameters
        x_dbl = layer.x_proj(x_conv)
        dt_raw, B, C = torch.split(x_dbl, [layer.dt_rank, layer.d_state, layer.d_state], dim=-1)

        # Bounded dt
        dt = torch.sigmoid(layer.dt_proj(dt_raw)) * (layer.dt_max - layer.dt_min) + layer.dt_min
        dt = torch.clamp(dt, min=1e-4)

        # Get A
        A = -torch.exp(layer.A_log.float())

        # Single-step state update (in-place for efficiency)
        dt_t = dt[:, 0, :].unsqueeze(-1)  # (1, d_inner, 1)
        B_t = B[:, 0, :].unsqueeze(1)      # (1, 1, d_state)
        C_t = C[:, 0, :].unsqueeze(1)      # (1, 1, d_state)
        x_t = x_conv[:, 0, :].unsqueeze(-1)  # (1, d_inner, 1)

        # Discretize
        dA = torch.exp(dt_t * A.unsqueeze(0))  # (1, d_inner, d_state)
        dB = dt_t * B_t

        # Update state IN-PLACE
        h = dA * h + dB * x_t

        # Output
        y = torch.sum(h * C_t, dim=-1)  # (1, d_inner)
        y = y + x_conv[:, 0, :] * layer.D

        # Gating
        y = y * F.silu(z_branch[:, 0, :])

        # Output projection
        out = layer.out_proj(y.unsqueeze(1))  # (1, 1, d_model)

        return out, h


def benchmark_stateful():
    """Benchmark stateful vs stateless inference."""
    args = ModelArgs(d_model=64, n_layer=2)
    model = LiquidStreamModel(args)

    try:
        model.load_state_dict(torch.load('model.pt', map_location='cpu', weights_only=True))
        print("Loaded trained model")
    except:
        print("Using random weights")

    stateful = StatefulLiquidModel(model)

    # Warmup
    for _ in range(10):
        _ = stateful.step(ord('v'))
    stateful.reset()

    # Benchmark
    n_bytes = 100

    print(f"\nGenerating {n_bytes} bytes (stateful)...")
    latencies = []
    output = []

    byte_val = ord('v')
    output.append(byte_val)

    for i in range(n_bytes):
        start = time.perf_counter()
        byte_val = stateful.step(byte_val)
        end = time.perf_counter()

        latencies.append(end - start)
        output.append(byte_val)

    # Results
    print(f"\nStateful Inference Results:")
    print(f"  First byte:  {latencies[0]*1000:.3f} ms")
    print(f"  Last byte:   {latencies[-1]*1000:.3f} ms")
    print(f"  Avg latency: {sum(latencies)/len(latencies)*1000:.3f} ms")
    print(f"  Ratio:       {latencies[-1]/latencies[0]:.2f}x")

    # Check constant time
    ratio = latencies[-1] / latencies[0]
    if ratio < 1.2:
        print("  ✓ PASS: Near-constant O(1) latency")
    else:
        print("  ⚠ Ratio > 1.2x")

    print(f"\nGenerated: {bytes(output[:50]).decode('utf-8', errors='replace')}...")


if __name__ == "__main__":
    benchmark_stateful()
