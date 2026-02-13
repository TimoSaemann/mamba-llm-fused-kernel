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
speedup_torch_compile = np.array([5.40, 2.66, 2.07, 1.7, 1.43, 1.31, 1.11, 1.08, 0.98])
speedup_llm_cuda_32 = np.array([8.88, 7.93, 5.93, 5.43, 4.78, 4.82, 4.12, 4.29, 3.6])
speedup_llm_cuda_16 = np.array([12.26, 12.41, 9.48, 9.52, 8.11, 8.4, 7.05, 7.54, 6.56])
speedup_amp = np.array([0.95, 0.94, 1.01, 1.03, 1.07, 1.08, 1.04, 1.08, 1.03])
speedup_mamba_lib = np.array([16.37, 12.93, 10.39, 8.41, 7.69, 7.43, 6.28, 6.25, 5.63])

plt.figure(figsize=(8, 5))
plt.plot(batch_sizes, speedup_eager, marker='o', linestyle='--', color='gray', label='PyTorch Eager')
plt.plot(batch_sizes, speedup_torch_compile, marker='s', label='torch.compile', color='tab:blue')
plt.plot(batch_sizes, speedup_amp, marker='^', label='PyTorch AMP FP16', color='tab:red')
plt.plot(batch_sizes, speedup_llm_cuda_32, marker='x', label='LLM-Optimized CUDA FP32', color='tab:orange')
plt.plot(batch_sizes, speedup_llm_cuda_16, marker='v', label='LLM-Optimized CUDA FP16', color='tab:green')
plt.plot(batch_sizes, speedup_mamba_lib, marker='p', label='Official Mamba Repo', color='tab:purple')

plt.xlabel('Batch size', fontsize=12)
plt.ylabel('Speedup', fontsize=12)
plt.title('Mamba Full-Sequence: Speedup vs Batch Size', fontsize=14)
plt.xticks(batch_sizes)
plt.yticks(np.arange(1, 9))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig('mamba_speedup_bench.png', dpi=300)
plt.show()