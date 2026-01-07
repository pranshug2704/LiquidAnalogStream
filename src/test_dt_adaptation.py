"""
test_dt_adaptation.py - Verify dt tracks input complexity

Tests that the model's dt (time-step) adapts to:
- Low entropy (simple patterns) -> Higher dt (faster)
- High entropy (complex patterns) -> Lower dt (slower)
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import LiquidStreamModel, ModelArgs

def extract_dt_values(model, input_bytes):
    """Extract dt values for a given input sequence."""
    model.eval()
    x = torch.tensor([input_bytes], dtype=torch.long)

    with torch.no_grad():
        # Get embeddings
        x_emb = model.embedding(x)

        # Process through first layer to get dt
        layer = model.layers[0]
        x_emb = model.norm_f(x_emb)

        # Input projection
        xz = layer.in_proj(x_emb)
        x_branch, _ = xz.chunk(2, dim=-1)

        # Convolution
        x_conv = x_branch.transpose(1, 2)
        x_conv = layer.conv1d(x_conv)[:, :, :x.shape[1]]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Get dt
        x_dbl = layer.x_proj(x_conv)
        dt_raw = x_dbl[..., :layer.dt_rank]
        dt = torch.sigmoid(layer.dt_proj(dt_raw)) * (layer.dt_max - layer.dt_min) + layer.dt_min
        dt = torch.clamp(dt, min=1e-4)

        # Average across d_inner for each position
        dt_mean = dt[0].mean(dim=-1).numpy()

    return dt_mean

def test_adaptation():
    """Test dt adaptation on different input patterns."""
    # Model with continuous embedding
    args = ModelArgs(
        d_model=64,
        n_layer=2,
        continuous_embed=True,
        multi_scale=True,
        n_substeps=2
    )
    model = LiquidStreamModel(args)

    # Try loading trained weights
    try:
        model.load_state_dict(torch.load('model.pt', map_location='cpu', weights_only=True))
        print("Loaded trained weights")
    except:
        print("Using random weights")

    # Test patterns
    patterns = {
        'Constant (low entropy)': [ord('a')] * 64,
        'Alternating (medium)': [ord('a'), ord('b')] * 32,
        'Random (high entropy)': list(np.random.randint(0, 256, 64)),
        'Sine wave text': [ord(c) for c in "val: 0.123\nval: 0.456\nval: 0.789\nval: 0.012\nval: 0.345\nval: 0.67"]
    }

    fig, axes = plt.subplots(len(patterns), 1, figsize=(12, 2.5*len(patterns)))

    for idx, (name, pattern) in enumerate(patterns.items()):
        dt_vals = extract_dt_values(model, pattern)

        ax = axes[idx]
        ax.plot(dt_vals, 'b-', linewidth=1.5, label='dt')
        ax.axhline(y=args.dt_min, color='r', linestyle='--', alpha=0.5, label='dt_min')
        ax.axhline(y=args.dt_max, color='g', linestyle='--', alpha=0.5, label='dt_max')
        ax.fill_between(range(len(dt_vals)), dt_vals, alpha=0.3)
        ax.set_title(f'{name} | Mean dt: {np.mean(dt_vals):.4f}')
        ax.set_ylabel('dt')
        ax.legend(loc='upper right')
        ax.set_xlim(0, len(dt_vals)-1)

    axes[-1].set_xlabel('Position')
    plt.tight_layout()
    plt.savefig('dt_adaptation_test.png', dpi=150)
    print("\nSaved: dt_adaptation_test.png")

    # Summary
    print("\ndt Adaptation Summary:")
    print("=" * 40)
    for name, pattern in patterns.items():
        dt_vals = extract_dt_values(model, pattern)
        print(f"{name:25s} | Mean dt: {np.mean(dt_vals):.4f} | Std: {np.std(dt_vals):.4f}")

if __name__ == "__main__":
    test_adaptation()
