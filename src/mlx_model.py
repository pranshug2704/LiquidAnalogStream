import math
import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass

@dataclass
class ModelArgs:
    d_model: int = 256
    n_layer: int = 4
    vocab_size: int = 256
    d_state: int = 16
    expand: int = 2
    dt_rank: int = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x):
        rsqrt = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return self.weight * x * rsqrt

class LiquidMambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_inner = args.d_inner
        self.d_conv = args.d_conv
        self.d_model = args.d_model
        self.d_state = args.d_state
        self.dt_rank = args.dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=args.d_conv,
            stride=1,
            padding=args.d_conv - 1,
            groups=self.d_inner,
            bias=args.conv_bias
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Mamba Init: dt initialization
        # dt ~ exp(uniform(log(0.001), log(0.1)))
        # bias = inv_softplus(dt)
        dt_init_min, dt_init_max = 0.001, 0.1
        dt = mx.exp(
            mx.random.uniform(
                low=math.log(dt_init_min),
                high=math.log(dt_init_max),
                shape=(self.d_inner,)
            )
        )
        inv_dt = dt + mx.log(-mx.expm1(-dt)) # approx inv_softplus
        self.dt_proj.bias = inv_dt # Assign to bias (nn.Linear bias is mx.array)

        # A initialization
        A = mx.arange(1, self.d_state + 1, dtype=mx.float32)
        self._A_log = mx.log(mx.tile(A[None, :], (self.d_inner, 1)))

        self.D = mx.ones((self.d_inner,))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=args.bias)

    def __call__(self, x):
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x_branch, z_branch = mx.split(xz, 2, axis=-1)

        x_conv = self.conv1d(x_branch)
        x_conv = x_conv[:, :L, :]
        x_conv = nn.silu(x_conv)

        x_dbl = self.x_proj(x_conv)
        dt = x_dbl[..., :self.dt_rank]
        B_ssm = x_dbl[..., self.dt_rank : self.dt_rank + self.d_state]
        C_ssm = x_dbl[..., self.dt_rank + self.d_state :]

        dt = nn.softplus(self.dt_proj(dt))

        A = -mx.exp(self._A_log)

        # Recurrence
        dA = mx.exp(dt[..., None] * A)

        # dB computation: Check shape broadcasting carefully
        # dt: (B, L, d_inner, 1)
        # B_ssm: (B, L, 1, d_state)
        # x_conv: (B, L, d_inner, 1)

        dB_x = (dt[..., None] * B_ssm[..., None, :]) * x_conv[..., None]

        h_state = mx.zeros((B, self.d_inner, self.d_state))
        ys = []

        for t in range(L):
            h_state = dA[:, t] * h_state + dB_x[:, t]

            # C_ssm[:, t] -> (B, d_state) -> (B, 1, d_state)
            ctx = C_ssm[:, t, None, :] * h_state
            y_t = mx.sum(ctx, axis=-1)
            ys.append(y_t)

        y = mx.stack(ys, axis=1)

        y = y + x_conv * self.D
        y = y * nn.silu(z_branch)
        out = self.out_proj(y)
        return out

class LiquidStreamModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = [
            LiquidMambaBlock(args) for _ in range(args.n_layer)
        ]
        self.norm_f = RMSNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def __call__(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)
