import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.scale = d ** 0.5
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        rms = nn.RMSNorm(normalized_shape=1024, eps=1e-5)
        out = rms(x)
        return out


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 32, ker_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)

        # matches official (expand=1 => d_inner=d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.conv = nn.Conv1d(
            d_model, d_model, ker_size,
            padding=ker_size - 1, groups=d_model, bias=True
        )

        # unfused equivalents of official x_proj slices
        self.s_dt = nn.Linear(d_model, self.dt_rank, bias=False)
        self.s_B  = nn.Linear(d_model, d_state, bias=False)
        self.s_C  = nn.Linear(d_model, d_state, bias=False)

        # official dt expand + bias
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # official parameterization of A
        # store A_log like official (or store A and convert in forward)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # official skip
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, seq: Tensor) -> Tensor:
        B, L, D = seq.shape

        a, z = self.in_proj(seq).chunk(2, dim=-1)

        # depthwise conv on a
        x = rearrange(a, "b l d -> b d l")
        x = self.conv(x)[..., :L]
        x = F.silu(x)  # official uses SiLU activation

        # compute dt, B, C from x (note: x currently BDL)
        x_bl_d = rearrange(x, "b d l -> (b l) d")

        dt_low = self.s_dt(x_bl_d)                  # (B*L, dt_rank)
        dt = self.dt_proj(dt_low)                   # (B*L, D) includes bias
        dt = rearrange(dt, "(b l) d -> b l d", b=B, l=L)

        Bv = self.s_B(x_bl_d)                       # (B*L, d_state)
        Cv = self.s_C(x_bl_d)                       # (B*L, d_state)

        # reshape to match scan math
        Bv = rearrange(Bv, "(b l) s -> b l s", b=B, l=L)
        Cv = rearrange(Cv, "(b l) s -> b l s", b=B, l=L)

        # delta is softplus(dt) (dt already has bias)
        delta = F.softplus(dt)                      # (B, L, D)

        A = -torch.exp(self.A_log.float())          # (D, S)

        # x currently is (B, D, L) after conv + silu
        x_bld = rearrange(x, "b d l -> b l d")  # (B, L, D)

        # delta is (B, L, D)
        # A is (D, S)
        # Bv, Cv are (B, L, S)

        # A_bar = exp(delta * A)  -> (B, L, D, S)
        A_bar = torch.exp(einsum(A, delta, 'd s, b l d -> b l d s'))

        # B_bar = delta * B  -> (B, L, D, S)
        # (broadcast B over D)
        B_bar = einsum(Bv, delta, 'b l s, b l d -> b l d s')

        # X_bar = B_bar * x -> (B, L, D, S)
        X_bar = einsum(B_bar, x_bld, 'b l d s, b l d -> b l d s')

        # hid via old-baseline scan algorithm
        hid = self._hid_states(A_bar, X_bar)  # (B, L, D, S)

        # output: sum over state with C, then add skip
        y = einsum(hid, Cv, 'b l d s, b l s -> b l d')  # (B, L, D)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_bld
        # (B, L, D)

        # gate by silu(z)
        y = y * F.silu(z)

        # out proj
        out = self.out_proj(y)
        return out

    @staticmethod
    def _hid_states(A_bar: Tensor, X_bar: Tensor) -> Tensor:
        # A_bar, X_bar: (B, L, D, S)
        b, l, d, s = A_bar.shape
        A = rearrange(A_bar, 'b l d s -> l b d s')
        X = rearrange(X_bar, 'b l d s -> l b d s')

        h = torch.zeros(b, d, s, device=A_bar.device, dtype=torch.float32)
        return torch.stack([h := A_t * h + X_t for A_t, X_t in zip(A, X)], dim=1)


class Model(nn.Module):
    def __init__(
            self,
            vocab_size: int = 16384,
            num_layers: int = 8,
            d_model: int = 1024,
            d_state: int = 32,
            ker_size: int = 4,
    ) -> None:
        super().__init__()
        d_input = d_model
        self.embedding = nn.Embedding(vocab_size, d_input)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                MambaBlock(d_model=d_model, d_state=d_state, ker_size=ker_size),
                RMSNorm(d_input)
            ])
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(d_input, vocab_size, bias=False)

    def forward(self, tok: Tensor) -> Tuple[Tensor, Tuple]:
        tok = torch.atleast_2d(tok)
        seq = self.embedding(tok)

        for mamba, norm in self.layers:
            out = mamba(norm(seq))
            seq = out + seq

        return self.head(seq)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


if __name__ == "__main__":
    torch.set_default_device("cuda")
    model = Model()

    # Make sure inputs are fixed and deterministic
    B, L = 2, 256
    x = torch.randint(0, 16384, (B, L), device="cuda")

    torch.cuda.synchronize()  # Ensure all kernels have finished

    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        model(x)
    p.export_chrome_trace("trace_eager.json")

    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log")
    ) as prof:
        model(x)

    print(prof.key_averages().table(sort_by="cuda_time", row_limit=50))
