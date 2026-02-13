import time
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Dict, Tuple

import numpy as np
import torch
from torch.amp import autocast

from mamba_baseline import Model as MambaBaselineAdapted
from mamba_cuda import Model as MambaC
from mamba_official import OfficialMambaModel


# ---------------- Config ----------------
@dataclass(frozen=True)
class BenchConfig:
    B: int = 1
    L: int = 512
    vocab_size: int = 16384
    d_model: int = 1024
    d_state: int = 32
    num_layers: int = 8
    ker_size: int = 4
    seed: int = 0


# ---------------- Utilities ----------------
@torch.no_grad()
def run_forward_and_time(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    num_warmup: int = 10,
    num_iters: int = 100,
    use_amp: bool = False,
) -> float:
    """Returns avg forward time in ms."""
    model.eval()
    torch.cuda.synchronize()

    ctx = autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()

    # Warmup
    for _ in range(num_warmup):
        with ctx:
            _ = model(x)
    torch.cuda.synchronize()

    # Timed
    start = time.time()
    for _ in range(num_iters):
        with ctx:
            _ = model(x)
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) * 1000.0 / num_iters


@torch.no_grad()
def measure_peak_vram(model: torch.nn.Module, x: torch.Tensor, *, use_amp: bool = False) -> int:
    """Returns peak allocated bytes during a single forward."""
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    ctx = autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
    with ctx:
        _ = model(x)

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.2f} MB"
    return f"{num_bytes / 1024**3:.2f} GB"


def to_np(y: torch.Tensor) -> np.ndarray:
    return y.detach().float().cpu().numpy()


def compare_outputs(name_a: str, y_a: np.ndarray, name_b: str, y_b: np.ndarray) -> None:
    diff = np.abs(y_a - y_b)
    denom = np.maximum(np.abs(y_b), 1e-3)
    rel_diff = diff / denom
    match = np.allclose(y_a, y_b, atol=1e-2, rtol=1e-2)

    print(f"\n=== {name_a} vs {name_b} ===")
    print(f"Output ranges: [{y_a.min():.2f}, {y_a.max():.2f}] vs [{y_b.min():.2f}, {y_b.max():.2f}]")
    print(f"Max abs diff: {diff.max():.4f}, Max rel diff: {rel_diff.max():.2e}")
    print(f"Outputs match? {'Yes ✅' if match else 'No ❌'}")


# ---------------- Weight copy (official -> custom) ----------------
@torch.no_grad()
def copy_official_model_to_custom(custom_model: torch.nn.Module, official_model: torch.nn.Module) -> None:
    """
    Copy weights from OfficialMambaModel -> custom model.

    Works for:
      - adapted PyTorch baseline blocks (have s_dt, s_B, s_C)
      - CUDA blocks (have s_BCD)
    """
    # Embedding + head
    custom_model.embedding.weight.copy_(
        official_model.embedding.weight.to(custom_model.embedding.weight.dtype)
    )
    custom_model.head.weight.copy_(
        official_model.head.weight.to(custom_model.head.weight.dtype)
    )

    # Per-layer blocks
    for (c_block, _c_norm), (o_block, _o_norm) in zip(custom_model.layers, official_model.layers):
        o = o_block.mamba

        # Common params
        c_block.in_proj.weight.copy_(o.in_proj.weight.to(c_block.in_proj.weight.dtype))
        c_block.out_proj.weight.copy_(o.out_proj.weight.to(c_block.out_proj.weight.dtype))

        c_block.conv.weight.copy_(o.conv1d.weight.to(c_block.conv.weight.dtype))
        c_block.conv.bias.copy_(o.conv1d.bias.to(c_block.conv.bias.dtype))

        c_block.A_log.copy_(o.A_log.float())
        c_block.D.copy_(o.D.to(c_block.D.dtype))

        c_block.dt_proj.weight.copy_(o.dt_proj.weight.to(c_block.dt_proj.weight.dtype))
        c_block.dt_proj.bias.copy_(o.dt_proj.bias.to(c_block.dt_proj.bias.dtype))

        # Projection packing differs
        W = o.x_proj.weight  # (dt_rank + 2*d_state, d_model)

        if hasattr(c_block, "s_BCD"):
            c_block.s_BCD.weight.copy_(W.to(c_block.s_BCD.weight.dtype))
        else:
            dt_rank = o.dt_rank
            d_state = o.d_state
            c_block.s_dt.weight.copy_(W[:dt_rank, :].to(c_block.s_dt.weight.dtype))
            c_block.s_B.weight.copy_(W[dt_rank:dt_rank + d_state, :].to(c_block.s_B.weight.dtype))
            c_block.s_C.weight.copy_(W[dt_rank + d_state:dt_rank + 2 * d_state, :].to(c_block.s_C.weight.dtype))


# ---------------- Main ----------------
def build_models(cfg: BenchConfig) -> Dict[str, torch.nn.Module]:
    torch.manual_seed(cfg.seed)
    eager = MambaBaselineAdapted(d_state=cfg.d_state).cuda().eval()

    torch.manual_seed(cfg.seed)
    compiled = torch.compile(MambaBaselineAdapted(d_state=cfg.d_state).cuda().eval())

    torch.manual_seed(cfg.seed)
    cuda_fp32 = MambaC(dtype=torch.float32, big_fuse=True, d_state=cfg.d_state).cuda().eval()

    torch.manual_seed(cfg.seed)
    cuda_fp16 = MambaC(dtype=torch.float16, big_fuse=True, d_state=cfg.d_state).cuda().eval()

    torch.manual_seed(cfg.seed)
    official = OfficialMambaModel(
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_layers,
        d_input=cfg.d_model,
        d_model=cfg.d_model,
        d_state=cfg.d_state,
        ker_size=cfg.ker_size,
    ).cuda().eval()

    return {
        "eager": eager,
        "compiled": compiled,
        "cuda_fp32": cuda_fp32,
        "cuda_fp16": cuda_fp16,
        "official": official,
    }


def main() -> None:
    torch.set_default_device("cuda")
    cfg = BenchConfig()

    # Input
    x = torch.randint(0, cfg.vocab_size, (cfg.B, cfg.L), device="cuda", dtype=torch.long)

    # Models
    models = build_models(cfg)
    eager = models["eager"]
    compiled = models["compiled"]
    cuda_fp32 = models["cuda_fp32"]
    cuda_fp16 = models["cuda_fp16"]
    official = models["official"]

    # Copy official -> others
    for m in (eager, compiled, cuda_fp32, cuda_fp16):
        copy_official_model_to_custom(m, official)

    # Correctness
    with torch.no_grad():
        y_eager = eager(x)
        y_compiled = compiled(x)
        y_cuda_fp32 = cuda_fp32(x)
        y_cuda_fp16 = cuda_fp16(x)
        with autocast("cuda", dtype=torch.float16):
            y_amp = eager(x)
        y_official = official(x)

    compare_outputs("Eager FP32", to_np(y_eager), "Compiled FP32", to_np(y_compiled))
    compare_outputs("Eager FP32", to_np(y_eager), "CUDA FP32", to_np(y_cuda_fp32))
    compare_outputs("Eager FP32", to_np(y_eager), "CUDA FP16", to_np(y_cuda_fp16))
    compare_outputs("Eager FP32", to_np(y_eager), "AMP FP16", to_np(y_amp))
    compare_outputs("Eager FP32", to_np(y_eager), "Mamba lib", to_np(y_official))

    # Timing
    print("\n=== Runtime Benchmark (ms) ===")
    t_eager = run_forward_and_time(eager, x)
    t_compiled = run_forward_and_time(compiled, x)
    t_cuda_fp32 = run_forward_and_time(cuda_fp32, x)
    t_cuda_fp16 = run_forward_and_time(cuda_fp16, x)
    t_amp = run_forward_and_time(eager, x, use_amp=True)
    t_official = run_forward_and_time(official, x)

    print(f"PyTorch eager FP32:   {t_eager:.3f} ms")
    print(f"torch.compile FP32:   {t_compiled:.3f} ms")
    print(f"Custom CUDA FP32:     {t_cuda_fp32:.3f} ms")
    print(f"Custom CUDA FP16:     {t_cuda_fp16:.3f} ms")
    print(f"PyTorch AMP FP16:     {t_amp:.3f} ms")
    print(f"Official Mamba:  {t_official:.3f} ms")

    print(
        f"\nSpeedup vs eager FP32:\n"
        f"  compile     = {t_eager / t_compiled:.2f}×\n"
        f"  cuda FP32   = {t_eager / t_cuda_fp32:.2f}×\n"
        f"  cuda FP16   = {t_eager / t_cuda_fp16:.2f}×\n"
        f"  AMP FP16    = {t_eager / t_amp:.2f}×\n"
        f"  Off. Mamba  = {t_eager / t_official:.2f}×"
    )

    # VRAM
    print("\n=== Peak VRAM per forward ===")
    vram_eager = measure_peak_vram(eager, x)
    vram_compiled = measure_peak_vram(compiled, x)
    vram_cuda_fp32 = measure_peak_vram(cuda_fp32, x)
    vram_cuda_fp16 = measure_peak_vram(cuda_fp16, x)
    vram_amp = measure_peak_vram(eager, x, use_amp=True)
    vram_official = measure_peak_vram(official, x)

    print(f"PyTorch eager FP32:   {format_bytes(vram_eager)}")
    print(f"torch.compile FP32:   {format_bytes(vram_compiled)}")
    print(f"Custom CUDA FP32:     {format_bytes(vram_cuda_fp32)}")
    print(f"Custom CUDA FP16:     {format_bytes(vram_cuda_fp16)}")
    print(f"PyTorch AMP FP16:     {format_bytes(vram_amp)}")
    print(f"Official Mamba FP32:  {format_bytes(vram_official)}")

    print(
        f"\nVRAM reduction vs eager FP32:\n"
        f"  cuda FP32   = {vram_eager / vram_cuda_fp32:.2f}×\n"
        f"  cuda FP16   = {vram_eager / vram_cuda_fp16:.2f}×\n"
        f"  AMP FP16    = {vram_eager / vram_amp:.2f}×"
    )


if __name__ == "__main__":
    main()
