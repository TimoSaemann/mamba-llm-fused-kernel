import torch
import numpy as np
import time

from mamba_baseline import Model as MambaBaseline
from mamba_cuda import Model as MambaC
from torch.amp import autocast


# ---------------- Utility: Timing helper ----------------
@torch.no_grad()
def run_forward_and_time(model, x, num_warmup=10, num_iters=100, use_amp=False):
    model.eval()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(num_warmup):
        if use_amp:
            with autocast("cuda", dtype=torch.float16):
                _ = model(x)
        else:
            _ = model(x)
    torch.cuda.synchronize()

    # Timed runs
    start = time.time()
    for _ in range(num_iters):
        if use_amp:
            with autocast("cuda", dtype=torch.float16):
                _ = model(x)
        else:
            _ = model(x)
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) * 1000 / num_iters


# ---------------- Utility: VRAM helper ----------------
@torch.no_grad()
def measure_peak_vram(model, x, use_amp=False):
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    if use_amp:
        with autocast("cuda", dtype=torch.float16):
            _ = model(x)
    else:
        _ = model(x)

    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    return peak_bytes


def format_bytes(num_bytes):
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024 ** 3:
        return f"{num_bytes / 1024 ** 2:.2f} MB"
    else:
        return f"{num_bytes / 1024 ** 3:.2f} GB"


# ---------------- Main benchmark ----------------
def main():
    torch.set_default_device("cuda")

    # ---------------- Input setup ----------------
    B, L = 1, 512
    vocab_size = 16384
    x = torch.randint(0, vocab_size, (B, L), device="cuda", dtype=torch.long)

    # ---------------- Load models ----------------
    torch.manual_seed(0)
    model_eager = MambaBaseline().cuda().eval()

    torch.manual_seed(0)
    model_compiled = torch.compile(MambaBaseline().cuda().eval())

    torch.manual_seed(0)
    model_cuda_fp32 = MambaC(dtype=torch.float32).cuda().eval()

    torch.manual_seed(0)
    model_cuda_fp16 = MambaC(dtype=torch.float16).cuda().eval()

    # ---------------- Forward pass (correctness) ----------------
    with torch.no_grad():
        y_eager = model_eager(x)
        y_compiled = model_compiled(x)
        y_cuda_fp32 = model_cuda_fp32(x)
        y_cuda_fp16 = model_cuda_fp16(x)

        with autocast("cuda", dtype=torch.float16):
            y_amp = model_eager(x)

    def to_np(y):
        return y.detach().cpu().numpy()

    def compare_outputs(name_a, y_a, name_b, y_b):
        diff = np.abs(y_a - y_b)
        denom = np.maximum(np.abs(y_b), 1e-3)
        rel_diff = diff / denom
        match = np.allclose(y_a, y_b, atol=1e-2, rtol=1e-2)

        print(f"\n=== {name_a} vs {name_b} ===")
        print(f"Output ranges: [{y_a.min():.2f}, {y_a.max():.2f}] vs [{y_b.min():.2f}, {y_b.max():.2f}]")
        print(f"Max abs diff: {diff.max():.4f}, Max rel diff: {rel_diff.max():.2e}")
        print(f"Outputs match? {'Yes ✅' if match else 'No ❌'}")

    compare_outputs("Eager FP32", to_np(y_eager), "Compiled FP32", to_np(y_compiled))
    compare_outputs("Eager FP32", to_np(y_eager), "CUDA FP32", to_np(y_cuda_fp32))
    compare_outputs("Eager FP32", to_np(y_eager), "CUDA FP16", to_np(y_cuda_fp16))
    compare_outputs("Eager FP32", to_np(y_eager), "AMP FP16", to_np(y_amp))

    # ---------------- Runtime benchmark ----------------
    print("\n=== Runtime Benchmark (ms) ===")
    t_eager = run_forward_and_time(model_eager, x)
    t_compiled = run_forward_and_time(model_compiled, x)
    t_cuda_fp32 = run_forward_and_time(model_cuda_fp32, x)
    t_cuda_fp16 = run_forward_and_time(model_cuda_fp16, x)
    t_amp = run_forward_and_time(model_eager, x, use_amp=True)

    print(f"PyTorch eager FP32:   {t_eager:.3f} ms")
    print(f"torch.compile FP32:   {t_compiled:.3f} ms")
    print(f"Custom CUDA FP32:     {t_cuda_fp32:.3f} ms")
    print(f"Custom CUDA FP16:     {t_cuda_fp16:.3f} ms")
    print(f"PyTorch AMP FP16:     {t_amp:.3f} ms")

    print(
        f"\nSpeedup vs eager FP32:\n"
        f"  compile     = {t_eager / t_compiled:.2f}×\n"
        f"  cuda FP32   = {t_eager / t_cuda_fp32:.2f}×\n"
        f"  cuda FP16   = {t_eager / t_cuda_fp16:.2f}×\n"
        f"  AMP FP16    = {t_eager / t_amp:.2f}×"
    )

    # ---------------- VRAM benchmark ----------------
    print("\n=== Peak VRAM per forward ===")
    vram_eager = measure_peak_vram(model_eager, x)
    vram_compiled = measure_peak_vram(model_compiled, x)
    vram_cuda_fp32 = measure_peak_vram(model_cuda_fp32, x)
    vram_cuda_fp16 = measure_peak_vram(model_cuda_fp16, x)
    vram_amp = measure_peak_vram(model_eager, x, use_amp=True)

    print(f"PyTorch eager FP32:   {format_bytes(vram_eager)}")
    print(f"torch.compile FP32:   {format_bytes(vram_compiled)}")
    print(f"Custom CUDA FP32:     {format_bytes(vram_cuda_fp32)}")
    print(f"Custom CUDA FP16:     {format_bytes(vram_cuda_fp16)}")
    print(f"PyTorch AMP FP16:     {format_bytes(vram_amp)}")

    print(
        f"\nVRAM reduction vs eager FP32:\n"
        f"  cuda FP32   = {vram_eager / vram_cuda_fp32:.2f}×\n"
        f"  cuda FP16   = {vram_eager / vram_cuda_fp16:.2f}×\n"
        f"  AMP FP16    = {vram_eager / vram_amp:.2f}×"
    )


if __name__ == "__main__":
    main()
