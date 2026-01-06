"""
visualize_dt.py - Real-time Delta Visualization

Shows the model's "viscosity" (dt values) as it generates text.
- Low dt (slow thinking) = red
- High dt (fast processing) = green

Usage:
    python visualize_dt.py
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from model import LiquidStreamModel, ModelArgs

class DtCaptureLiquidMamba:
    """Wrapper to capture dt values during forward pass."""

    def __init__(self, model):
        self.model = model
        self.dt_history = []
        self._hook_handles = []
        self._install_hooks()

    def _install_hooks(self):
        """Install forward hooks on all LiquidMambaBlock layers."""
        for name, module in self.model.named_modules():
            if 'LiquidMambaBlock' in type(module).__name__:
                # We need to capture dt inside forward
                # Since dt is computed inside forward, we'll modify approach
                pass

        # Alternative: Patch the forward method
        # For simplicity, we'll compute dt separately

    def generate_with_dt(self, initial_tokens, max_len=100, device='cpu'):
        """Generate text while capturing dt values."""
        self.model.eval()

        tokens = initial_tokens.clone()
        generated = []
        dt_values = []

        with torch.no_grad():
            for _ in range(max_len):
                logits = self.model(tokens)

                # Sample next
                probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

                generated.append(next_tok.item())

                # Capture dt from first layer
                # We need to manually compute what dt would be
                layer = self.model.layers[0]
                x = self.model.embedding(tokens)
                for l in self.model.layers:
                    # Get x_conv
                    xz = l.in_proj(self.model.norm_f(x))
                    x_branch, _ = xz.chunk(2, dim=-1)
                    x_conv = x_branch.transpose(1, 2)
                    x_conv = l.conv1d(x_conv)[:, :, :tokens.shape[1]]
                    x_conv = x_conv.transpose(1, 2)
                    x_conv = F.silu(x_conv)

                    # Get dt
                    x_dbl = l.x_proj(x_conv)
                    dt_raw = x_dbl[..., :l.dt_rank]
                    dt = torch.sigmoid(l.dt_proj(dt_raw)) * (l.dt_max - l.dt_min) + l.dt_min

                    # Store mean dt for this step
                    dt_values.append(dt[:, -1, :].mean().item())
                    break  # Just first layer for simplicity

                tokens = torch.cat([tokens, next_tok], dim=1)
                if tokens.shape[1] > 256:
                    tokens = tokens[:, -256:]

        return generated, dt_values

def visualize():
    """Main visualization routine."""
    # Setup
    args = ModelArgs(d_model=64, n_layer=2, dt_min=0.001, dt_max=0.1)
    model = LiquidStreamModel(args)

    # Try loading weights
    try:
        model.load_state_dict(torch.load('model.pt', map_location='cpu', weights_only=True))
        print("Loaded model weights.")
    except:
        print("Using random weights.")

    wrapper = DtCaptureLiquidMamba(model)

    # Generate
    initial = torch.tensor([[ord(c) for c in "val: "]], dtype=torch.long)
    tokens, dt_vals = wrapper.generate_with_dt(initial, max_len=100)

    # Decode text
    text = bytes(tokens).decode('utf-8', errors='replace')

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # dt timeline
    ax1.plot(dt_vals, 'b-', linewidth=1)
    ax1.axhline(y=args.dt_min, color='r', linestyle='--', label=f'dt_min={args.dt_min}')
    ax1.axhline(y=args.dt_max, color='g', linestyle='--', label=f'dt_max={args.dt_max}')
    ax1.fill_between(range(len(dt_vals)), dt_vals, alpha=0.3)
    ax1.set_xlabel('Generation Step')
    ax1.set_ylabel('dt (Viscosity)')
    ax1.set_title('Model "Thinking Speed" Over Generation')
    ax1.legend()

    # Color-coded text
    ax2.axis('off')
    ax2.set_title('Generated Text (Red=Slow, Green=Fast)')

    # Normalize dt for coloring
    dt_norm = np.array(dt_vals)
    dt_norm = (dt_norm - dt_norm.min()) / (dt_norm.max() - dt_norm.min() + 1e-8)

    # Create colored text
    x_pos = 0.05
    y_pos = 0.8
    for i, (char, dt_n) in enumerate(zip(text, dt_norm)):
        # Red for slow (low dt), green for fast (high dt)
        color = (1 - dt_n, dt_n, 0)  # RGB
        ax2.text(x_pos, y_pos, char, fontsize=10, color=color,
                 fontfamily='monospace', transform=ax2.transAxes)
        x_pos += 0.01
        if x_pos > 0.95:
            x_pos = 0.05
            y_pos -= 0.15

    plt.tight_layout()
    plt.savefig('dt_visualization.png', dpi=150)
    print(f"\nVisualization saved to dt_visualization.png")
    print(f"Generated: {text[:50]}...")
    plt.show()

if __name__ == "__main__":
    visualize()
