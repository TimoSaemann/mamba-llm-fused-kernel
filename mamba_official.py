import torch
import torch.nn as nn
from torch import Tensor
from mamba_ssm import Mamba


class RMSNorm(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.scale = d ** 0.5
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        rms = nn.RMSNorm(normalized_shape=1024, eps=1e-5)
        out = rms(x)
        return out


# ---- Single block using official Mamba ----
class OfficialMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        ker_size: int,
    ):
        super().__init__()

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=ker_size,
            expand=1,
        )

    def forward(self, x: Tensor):
        # official block returns only output (no cache)
        return self.mamba(x)


# ---- Full model equivalent ----
class OfficialMambaModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16384,
        num_layers: int = 8,
        d_input: int = 1024,
        d_model: int = 1024,
        d_state: int = 32,
        ker_size: int = 4,
    ):
        super().__init__()

        assert d_input == d_model, \
            "Official Mamba requires d_input == d_model for fair comparison"

        self.embedding = nn.Embedding(vocab_size, d_input)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                OfficialMambaBlock(d_model, d_state, ker_size),
                RMSNorm(d_model),
            ])
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tok: Tensor):
        tok = torch.atleast_2d(tok)
        seq = self.embedding(tok)

        for mamba, norm in self.layers:
            out = mamba(norm(seq))
            seq = seq + out   # residual

        return self.head(seq)

    @property
    def device(self):
        return next(self.parameters()).device
