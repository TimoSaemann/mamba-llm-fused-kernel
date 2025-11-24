import matplotlib.pyplot as plt
import numpy as np


batch_sizes = np.array([1, 2, 4, 6, 8, 10, 12, 14])

speedup_eager = np.ones_like(batch_sizes) # baseline, always 1
speedup_torch_compile = np.array([4.54, 4.6, 4.4, 2.2, 1.98, 1.7, 1.46, 1.31])
speedup_llm_cuda = np.array([6.99, 4.9, 2.7, 2.4, 2.04, 1.97, 1.71, 1.67])

plt.figure(figsize=(8, 5))
plt.plot(batch_sizes, speedup_eager, marker='o', linestyle='--', color='gray', label='PyTorch Eager (baseline)')
plt.plot(batch_sizes, speedup_llm_cuda, marker='^', label='LLM Optimized CUDA', color='tab:orange')
#plt.plot(batch_sizes, speedup_torch_compile, marker='s', label='torch.compile', color='tab:blue')

plt.xlabel('Batch size', fontsize=12)
plt.ylabel('Speedup', fontsize=12)
plt.title('SSM Forward Pass: Speedup vs Batch Size', fontsize=14)
plt.xticks(batch_sizes)
plt.yticks(np.arange(1, 9))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig('speedup_plot.png', dpi=300)
plt.show()