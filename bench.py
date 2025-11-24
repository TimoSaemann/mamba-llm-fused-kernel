import torch
import numpy as np
import time

from mamba_baseline import Model as MambaBaseline
from mamba_cuda import Model as MambaC


# ---------------- Utility: Timing helper ----------------
@torch.no_grad()
def run_forward_and_time(model, x, num_warmup=10, num_iters=100):
    """Runs model forward several times and measures average runtime in ms."""
    model.eval()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(num_warmup):
        _ = model(x)
    torch.cuda.synchronize()

    # Timed runs
    start = time.time()
    for _ in range(num_iters):
        _ = model(x)
    torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) * 1000 / num_iters
    return avg_ms


# ---------------- Main benchmark ----------------
def main():
    # ---------------- Environment setup ----------------
    torch.set_default_device("cuda")
    # torch.backends.cuda.matmul.fp32_precision = 'high'  # or 'ieee'
    # torch.backends.cudnn.conv.fp32_precision = 'high'

    # ---------------- Input setup ----------------
    B, L = 1, 256  # batch size, sequence length
    vocab_size = 16384
    x = torch.randint(0, vocab_size, (B, L))

    # ---------------- Load models ----------------
    torch.manual_seed(0)
    model_eager = MambaBaseline().eval()
    torch.manual_seed(0)
    model_compiled = torch.compile(MambaBaseline().eval())
    torch.manual_seed(0)
    model_cuda = MambaC().eval()

    # ---------------- Forward pass ----------------
    with torch.no_grad():
        y_eager = model_eager(x)
        y_compiled = model_compiled(x)
        y_cuda = model_cuda(x)

    # Convert to numpy for comparison
    y_eager_np = y_eager.detach().cpu().numpy()
    y_compiled_np = y_compiled.detach().cpu().numpy()
    y_cuda_np = y_cuda.detach().cpu().numpy()

    def compare_outputs(name_a, y_a, name_b, y_b):
        diff = np.abs(y_a - y_b)
        rel_diff = diff / (np.abs(y_b) + 1e-8)
        match = np.allclose(y_a, y_b, atol=1e-2, rtol=1e-2)

        print(f"\n=== {name_a} vs {name_b} ===")
        print(f"Output ranges: [{y_a.min():.2f}, {y_a.max():.2f}] vs [{y_b.min():.2f}, {y_b.max():.2f}]")
        print(f"Max abs diff: {diff.max():.4f}, Max rel diff: {rel_diff.max():.2e}")
        print(f"Outputs match? {'Yes ✅' if match else 'No ❌'}")

    # ---------------- Compare outputs ----------------
    compare_outputs("Eager", y_eager_np, "Compiled", y_compiled_np)
    compare_outputs("Eager", y_eager_np, "CUDA", y_cuda_np)
    compare_outputs("Compiled", y_compiled_np, "CUDA", y_cuda_np)

    # ---------------- Runtime benchmark ----------------
    print("\n=== Runtime Benchmark (ms) ===")
    t_eager = run_forward_and_time(model_eager, x)
    t_compiled = run_forward_and_time(model_compiled, x)
    t_cuda = run_forward_and_time(model_cuda, x)

    print(f"PyTorch eager:   {t_eager:.3f} ms")
    print(f"torch.compile:   {t_compiled:.3f} ms")
    print(f"Custom CUDA:     {t_cuda:.3f} ms")
    print(f"Speedup vs eager: compile={t_eager / t_compiled:.2f}×, cuda={t_eager / t_cuda:.2f}×")


if __name__ == "__main__":
    main()
