import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelArgs:
    d_model: int = 256
    n_layer: int = 4
    vocab_size: int = 256  # Byte-level
    d_state: int = 16
    expand: int = 2
    dt_rank: int = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    # Liquid refinements
    dt_min: float = 0.001    # Min time-step (prevents vanishing gradient)
    dt_max: float = 0.1      # Max time-step (prevents instability)
    n_substeps: int = 1      # Sub-steps per input byte
    multi_scale: bool = False # Split state into slow/fast
    continuous_embed: bool = False  # Use sin/cos embedding instead of lookup

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class ContinuousEmbedding(nn.Module):
    """
    Continuous sin/cos embedding for bytes.
    Maps byte values to points on a circle, ensuring nearby bytes are mathematically similar.
    """
    def __init__(self, d_model: int, vocab_size: int = 256):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Learnable frequency multipliers for each dimension
        self.freq = nn.Parameter(torch.randn(d_model // 2) * 0.1 + 1.0)
        self.phase = nn.Parameter(torch.zeros(d_model // 2))
        # Linear projection to full d_model
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len) of byte values 0-255
        # Normalize to [0, 2π]
        theta = x.float() / (self.vocab_size - 1) * 2 * math.pi  # (B, L)
        theta = theta.unsqueeze(-1)  # (B, L, 1)

        # Apply learned frequencies
        angles = theta * self.freq + self.phase  # (B, L, d_model//2)

        # Sin/cos encoding
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, L, d_model)
        return self.proj(emb)

class LiquidMambaBlock(nn.Module):
    """
    A 'Liquid' Mamba Block.
    This implements the Selective State Space Model (S6) logic in pure PyTorch.
    The 'Liquid' aspect is the input-dependent delta (dt).
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_inner = args.d_inner
        self.d_conv = args.d_conv
        self.d_model = args.d_model
        self.d_state = args.d_state
        self.dt_rank = args.dt_rank
        # Liquid refinements
        self.dt_min = args.dt_min
        self.dt_max = args.dt_max
        self.n_substeps = args.n_substeps

        # Projects input to d_inner * 2 (for x and z)
        # We process both paths, then gate them at the end
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=args.bias)

        # 1D Convolution (Standard SSM conv)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=self.d_inner,
            padding=args.d_conv - 1,
        )

        # Projects x_t to dt, B, C (the "Selective" mechanism)
        # x_proj maps to (dt_rank + 2*d_state)
        # i.e., project to dt, B, and C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # Projects dt from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D real initialization (A parameter)
        # A is (d_inner, d_state)
        # Multi-scale: First half of state dims are "slow" (small A = long memory)
        #              Second half are "fast" (large A = local context)
        if hasattr(args, 'multi_scale') and args.multi_scale:
            half = self.d_state // 2
            A_slow = torch.arange(1, half + 1, dtype=torch.float32) * 0.1  # Slow decay
            A_fast = torch.arange(1, self.d_state - half + 1, dtype=torch.float32) * 2.0  # Fast decay
            A = torch.cat([A_slow, A_fast]).repeat(self.d_inner, 1)
        else:
            A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=args.bias)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # 1. Input Projection
        xz = self.in_proj(x) # (B, L, 2*d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1) # (B, L, d_inner), (B, L, d_inner)

        # 2. Convolution
        # Rearrange for Conv1d: (B, d_inner, L)
        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len] # Causal padding
        x_conv = x_conv.transpose(1, 2) # (B, L, d_inner)
        x_conv = F.silu(x_conv) # Activation

        # 3. State Space Model (SSM) - The "Liquid" Core
        # Calculate input-dependent parameters
        x_dbl = self.x_proj(x_conv) # (B, L, dt_rank + 2*d_state)

        # Split into dt, B, C
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # dt: (B, L, dt_rank) -> (B, L, d_inner)
        # Bounded sigmoid gating with epsilon floor (Lipschitz constraint)
        dt = torch.sigmoid(self.dt_proj(dt)) * (self.dt_max - self.dt_min) + self.dt_min
        dt = torch.clamp(dt, min=1e-4)  # Epsilon floor for stability

        # B, C: (B, L, d_state)

        # Selective Scan (Recursive or Parallel)
        # For this prototype, we use the sequential scan (slow in PyTorch but correct)
        # or a parallel scan if we implement it.
        # Let's do a sequential scan loop for clarity and correctness without kernels.

        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)

        # Discretize
        # dt: (B, L, d_inner)
        # A: (d_inner, d_state) -> broadcast to (B, L, d_inner, d_state)
        # But wait, usually we do element-wise.
        # dA = exp(dt * A)

        # We need to run the scan.
        # y_t = C_t * h_t + D * x_t
        # h_t = A_bar * h_{t-1} + B_bar * x_t

        # A_bar = exp(dt * A)
        # B_bar = (dt * A)^-1 (exp(dt * A) - I) * dt * B  ~= dt * B (approx)

        # This is the "Liquid" logic: A_bar and B_bar change every step (t).

        # Prepare for scan
        y = []
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device) # Hidden state

        # Unsqueezing for broadcasting
        # B: (batch, L, d_state)
        # x_conv: (batch, L, d_inner)
        # dt: (batch, L, d_inner)

        # Optimize by precomputing B_bar and A_bar maybe? No, depends on t.

        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1) # (batch, d_inner, 1)
            A_t = A.unsqueeze(0) # (1, d_inner, d_state)
            B_t = B[:, t, :].unsqueeze(1) # (batch, 1, d_state)
            C_t = C[:, t, :].unsqueeze(1) # (batch, 1, d_state)
            x_t = x_conv[:, t, :].unsqueeze(-1) # (batch, d_inner, 1)

            # Sub-stepping: divide dt by n_substeps for finer resolution
            dt_sub = dt_t / self.n_substeps

            for _ in range(self.n_substeps):
                # Discretize A -> A_bar = exp(dt_sub * A)
                dA = torch.exp(dt_sub * A_t) # (batch, d_inner, d_state)

                # Discretize B -> B_bar = dt_sub * B
                dB = dt_sub * B_t # (batch, d_inner, d_state)

                # Update state: h_t = dA * h_{t-1} + dB * x_t
                h = dA * h + dB * x_t

            # Output: y_t = C_t * h_t
            # C_t: (batch, 1, d_state)
            # h: (batch, d_inner, d_state)
            # sum over state dim
            y_t = torch.sum(h * C_t, dim=-1) # (batch, d_inner)

            y.append(y_t)

        y = torch.stack(y, dim=1) # (B, L, d_inner)

        # Add residual D * x
        y = y + x_conv * self.D

        # 4. Gating
        y = y * F.silu(z_branch)

        # 5. Output Projection
        out = self.out_proj(y)

        return out


class LiquidStreamModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # Choose embedding type
        if hasattr(args, 'continuous_embed') and args.continuous_embed:
            self.embedding = ContinuousEmbedding(args.d_model, args.vocab_size)
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([
            LiquidMambaBlock(args) for _ in range(args.n_layer)
        ])
        self.norm_f = RMSNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len)
        """
        x = self.embedding(x)

        for layer in self.layers:
            # Simple residual connection
            x = x + layer(x)
            x = self.norm_f(x) # Norm after each layer? Or before? Mamba usually pre-norm.
            # Let's assume pre-norm usage in block or here.
            # Standard Transformer is: x = x + layer(norm(x))
            # Let's adjust to Standard Pre-Norm

        logits = self.lm_head(x)
        return logits

def test_model():
    args = ModelArgs(d_model=64, n_layer=2)
    model = LiquidStreamModel(args)
    x = torch.randint(0, 256, (2, 32)) # Batch 2, Seq 32
    y = model(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
    # Expected output: (2, 32, 256)
    assert y.shape == (2, 32, 256)
    print("Forward pass successful!")

if __name__ == "__main__":
    test_model()
