"""
Reproduces the memory usage plot for the Mamba FP16 vs FP32 benchmark.

The numeric values are extracted from bench.py runs and are intentionally
inlined for figure reproducibility. This script is not meant to be a
general-purpose plotting utility.
"""
import matplotlib.pyplot as plt
import numpy as np


batch_sizes = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16])

speedup_eager = np.ones_like(batch_sizes) # baseline, always 1
speedup_torch_compile = np.array([4.85, 3.49, 2.63, 1.97, 1.64, 1.4, 1.25, 1.16, 1.09])
speedup_amp = np.array([0.93, 0.96, 1.08, 1.0, 1.09, 1.06, 1.09, 1.07, 1.08])
speedup_llm_cuda_32 = np.array([4.9, 3.88, 2.54, 2.28, 1.9, 1.83, 1.62, 1.59, 1.48])
speedup_llm_cuda_16 = np.array([8.02, 5.74, 3.79, 3.69, 3.04, 2.99, 2.6, 2.64, 2.41])

plt.figure(figsize=(8, 5))
plt.plot(batch_sizes, speedup_eager, marker='o', linestyle='--', color='gray', label='PyTorch Eager')
plt.plot(batch_sizes, speedup_torch_compile, marker='s', label='torch.compile', color='tab:blue')
plt.plot(batch_sizes, speedup_amp, marker='^', label='PyTorch AMP FP16', color='tab:red')
plt.plot(batch_sizes, speedup_llm_cuda_32, marker='x', label='LLM-Optimized CUDA FP32', color='tab:orange')
plt.plot(batch_sizes, speedup_llm_cuda_16, marker='v', label='LLM-Optimized CUDA FP16', color='tab:green')

plt.xlabel('Batch size', fontsize=12)
plt.ylabel('Speedup', fontsize=12)
plt.title('Mamba Forward Pass: Speedup vs Batch Size', fontsize=14)
plt.xticks(batch_sizes)
plt.yticks(np.arange(1, 9))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig('mamba_speedup_fp16.png', dpi=300)
plt.show()