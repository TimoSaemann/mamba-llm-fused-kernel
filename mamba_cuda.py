import argparse

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import silu, softplus
from torch.utils.cpp_extension import load_inline


################################################################################
# Python Helper: pscan (used for the "parallel" mode of the SSM)
################################################################################
def pscan(A: Tensor, X: Tensor) -> Tensor:
    # A and X are assumed to be of shape (l, b, d, s)
    # Multiply elementwise then permute back to (b, l, d, s)
    return (A * X).permute(1, 0, 2, 3).contiguous()


################################################################################
# CUDA Kernels: Fused RMSNorm, Fused SSM Scan-Reduce, and Fused Residual Addition.
################################################################################
cuda_source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuda_fp16.h>

extern "C" {

// ----------------------------------------------------------------
// Fused RMSNorm Kernel (unchanged, kept for completeness)
__global__ void fused_rmsnorm_kernel(const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     float* __restrict__ out,
                                     int d, float scale, float eps) {
    int row = blockIdx.x;  // one block per row
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float val = x[row * d + i];
        sum += val * val;
    }
    // Warp-level reduction within each warp.
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // Use shared memory for inter-warp reduction.
    __shared__ float warp_sum[32];  // up to 32 warps per block.
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    if (lane == 0) {
        warp_sum[wid] = sum;
    }
    __syncthreads();
    sum = (tid < blockDim.x / warpSize) ? warp_sum[lane] : 0.0f;
    if (tid < warpSize) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            float norm = sqrtf(sum);
            // Save the scaling factor in shared memory.
            warp_sum[0] = scale / (norm + eps);
        }
    }
    __syncthreads();
    float factor = warp_sum[0];
    // Apply RMSNorm: each thread updates its assigned elements.
    for (int i = tid; i < d; i += blockDim.x) {
        float x_val = x[row * d + i];
        float w_val = weight[i];
        out[row * d + i] = x_val * factor * w_val;
    }
}

// ----------------------------------------------------------------
// Fused SSM Scan-Reduce Kernel (FP32)
// All input pointers marked const and __restrict__ for alias analysis.
// Precompute base offsets to reduce integer arithmetic in the inner loop.
__global__ void fused_ssm_scan_reduce_kernel(const float* __restrict__ A,
                                             const float* __restrict__ X,
                                             const float* __restrict__ C,
                                             const float* __restrict__ seq,
                                             const float* __restrict__ D,
                                             float* __restrict__ out,
                                             float* __restrict__ hid,
                                             int B, int L, int d_model, int d_state) {
    // Each block corresponds to one chain: a unique (b, d) pair.
    int chain = blockIdx.x;
    const int b_idx = chain / d_model;
    const int d_idx = chain % d_model;
    const int tid = threadIdx.x;  // each thread handles one element of the hidden state.

    float h_val = 0.0f;  // initial hidden state.
    const float D_val = D[d_idx];

    // Precompute repeated stride terms and base offsets
    const int stride_per_timestep_A = d_model * d_state;   // number of elements per timestep for A/X
    const int stride_per_timestep_C = d_state;            // number of elements per timestep for C
    const int stride_per_timestep_seq = d_model;          // number of elements per timestep for seq

    const int baseA = b_idx * (L * stride_per_timestep_A) + d_idx * d_state; // base for A/X for this chain
    const int baseC = b_idx * (L * stride_per_timestep_C);                   // base for C for this batch
    const int baseSeq = b_idx * (L * stride_per_timestep_seq) + d_idx;      // base for seq per (b,d)

    // Loop sequentially over time steps.
    for (int l = 0; l < L; ++l) {
        const int idx = baseA + l * stride_per_timestep_A + tid;   // A/X index for (b,l,d,s)
        const float a_val = A[idx];
        const float x_val = X[idx];

        // recurrence
        h_val = a_val * h_val + x_val;
        hid[idx] = h_val;

        // C index and reduction
        const int cidx = baseC + l * stride_per_timestep_C + tid; // C index for (b,l,s)
        const float c_val = C[cidx];
        float prod = h_val * c_val;

        // Warp-level reduction for the dot product
        for (int offset = min(16, d_state / 2); offset > 0; offset /= 2) {
            prod += __shfl_down_sync(0xffffffff, prod, offset);
        }

        if (d_state > 32) {
            // Cross-warp reduction: accumulate one partial sum per warp
            __shared__ float sdata[32];
            int lane = tid % 32;        // lane within the warp
            int warp_id = tid / 32;     // warp index within the block
        
            // Write one partial sum per warp
            if (lane == 0) sdata[warp_id] = prod;
            __syncthreads();
        
            // Final reduction across warps (single thread)
            if (tid == 0) {
                float final_prod = 0;
                for (int i = 0; i < (d_state + 31) / 32; i++) {
                    final_prod += sdata[i];
                }
                prod = final_prod;
            }
        }
        
        // final reduction
        if (tid == 0) {
            const int seqidx = baseSeq + l * stride_per_timestep_seq;
            const float seq_val = seq[seqidx];
            float out_val = prod + D_val * seq_val;
            out[seqidx] = out_val;
        }
    }
}


// ----------------------------------------------------------------
// Fused SSM Scan-Reduce Kernel (FP16)
// All input pointers marked const and __restrict__ for alias analysis.
// Precompute base offsets to reduce integer arithmetic in the inner loop.
__global__ void fused_ssm_scan_reduce_kernel_fp16(
    const __half* __restrict__ A,
    const __half* __restrict__ X,
    const __half* __restrict__ C,
    const __half* __restrict__ seq,
    const __half* __restrict__ D,
    __half* __restrict__ out,
    __half* __restrict__ hid,
    int B, int L, int d_model, int d_state
) {
    // Each block corresponds to one chain: a unique (b, d) pair.
    int chain = blockIdx.x;
    const int b_idx = chain / d_model;
    const int d_idx = chain % d_model;
    const int tid = threadIdx.x;  // each thread handles one element of the hidden state.

    __half h_val = __float2half(0.0f);  // initial hidden state.
    const __half D_val = D[d_idx];

    // Precompute repeated stride terms and base offsets
    const int stride_per_timestep_A = d_model * d_state;   // number of elements per timestep for A/X
    const int stride_per_timestep_C = d_state;            // number of elements per timestep for C
    const int stride_per_timestep_seq = d_model;          // number of elements per timestep for seq

    const int baseA = b_idx * (L * stride_per_timestep_A) + d_idx * d_state; // base for A/X for this chain
    const int baseC = b_idx * (L * stride_per_timestep_C);                   // base for C for this batch
    const int baseSeq = b_idx * (L * stride_per_timestep_seq) + d_idx;      // base for seq per (b,d)

    // Loop sequentially over time steps.
    for (int l = 0; l < L; ++l) {
        const int idx = baseA + l * stride_per_timestep_A + tid;   // A/X index for (b,l,d,s)
        const __half a_val = A[idx];
        const __half x_val = X[idx];

        // recurrence
        h_val = __hfma(a_val, h_val, x_val); // h_val = a_val * h_val + x_val
        hid[idx] = h_val;

        // C index and reduction
        const int cidx = baseC + l * stride_per_timestep_C + tid;
        const half c_val = C[cidx];
        
        // accumulate in FP32
        float prod = __half2float(h_val) * __half2float(c_val);
        
        // Warp-level reduction for the dot product
        for (int offset = min(16, d_state / 2); offset > 0; offset /= 2) {
            prod += __shfl_down_sync(0xffffffff, prod, offset);
        }
        
        if (d_state > 32) {
            // Cross-warp reduction: accumulate one partial sum per warp
            __shared__ float sdata[32];
            int lane = tid % 32;        // lane within the warp
            int warp_id = tid / 32;     // warp index within the block
        
            // Write one partial sum per warp
            if (lane == 0) sdata[warp_id] = prod;
            __syncthreads();
        
            // Final reduction across warps (single thread)
            if (tid == 0) {
                float final_prod = 0;
                for (int i = 0; i < (d_state + 31) / 32; i++) {
                    final_prod += sdata[i];
                }
                prod = final_prod;
            }
        }
        
        // Write output for this timestep and (b, d) chain
        if (tid == 0) {
            const int seqidx = baseSeq + l * stride_per_timestep_seq;
            const half seq_val = seq[seqidx];
            float out_val = prod + __half2float(D_val) * __half2float(seq_val);
            out[seqidx] = __float2half(out_val);
        }
    }
}


// ----------------------------------------------------------------
// Fused Residual Addition Kernel:
// Takes two tensors "a" and "b" (of the same size) and writes out elementwise a+b.
__global__ void fused_residual_add_kernel(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          float* __restrict__ out,
                                          int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// ----------------------------------------------------------------
// Wrapper for Fused RMSNorm Kernel.
torch::Tensor fused_rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, int d, float scale, float eps) {
    auto N = x.size(0);  // number of rows.
    auto out = torch::empty_like(x);
    int block_size = 256;
    if (d < block_size) {
        block_size = ((d + 31) / 32) * 32;
    }
    dim3 grid(N);
    dim3 block(block_size);
    fused_rmsnorm_kernel<<<grid, block>>>(x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), d, scale, eps);
    return out;
}


torch::Tensor fused_ssm_scan_reduce_cuda(
    torch::Tensor A, torch::Tensor X, torch::Tensor C,
    torch::Tensor seq, torch::Tensor D,
    int B, int L, int d_model, int d_state,
    torch::Tensor hid
) {
    auto dtype = A.scalar_type();
    auto out = torch::empty({B, L, d_model}, seq.options());

    dim3 grid(B * d_model);
    dim3 block(d_state);

    if (dtype == c10::kFloat) {
        fused_ssm_scan_reduce_kernel<<<grid, block>>>(
            A.data_ptr<float>(), X.data_ptr<float>(),
            C.data_ptr<float>(), seq.data_ptr<float>(),
            D.data_ptr<float>(), out.data_ptr<float>(),
            hid.data_ptr<float>(), B, L, d_model, d_state
        );
    } else if (dtype == c10::kHalf) {
        fused_ssm_scan_reduce_kernel_fp16<<<grid, block>>>(
            reinterpret_cast<__half*>(A.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(X.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(C.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(seq.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(D.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(out.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(hid.data_ptr<c10::Half>()),
            B, L, d_model, d_state
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for fused_ssm_scan_reduce_cuda");
    }

    return out;
}


// ----------------------------------------------------------------
// Wrapper for Fused Residual Addition Kernel.
torch::Tensor fused_residual_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::empty_like(a);
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    fused_residual_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

} // extern "C"
'''

cpp_source = r'''
extern "C" {
torch::Tensor fused_rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, int d, float scale, float eps);
torch::Tensor fused_ssm_scan_reduce_cuda(torch::Tensor A, torch::Tensor X, torch::Tensor C,
                                         torch::Tensor seq, torch::Tensor D,
                                         int B, int L, int d_model, int d_state,
                                         torch::Tensor hid, c10::ScalarType dtype);
torch::Tensor fused_residual_add_cuda(torch::Tensor a, torch::Tensor b);
}
'''

# Compile the inline CUDA module with our optimized kernels.
cuda_module = load_inline(name="cuda_kernels_optimized",
                          cpp_sources=cpp_source,
                          cuda_sources=cuda_source,
                          functions=["fused_rmsnorm_cuda", "fused_ssm_scan_reduce_cuda", "fused_residual_add_cuda"],
                          verbose=True,
                          extra_cflags=[''],
                          extra_ldflags=[''])


################################################################################
# Python Modules: RMSNorm, MambaBlock, and Model
################################################################################

# Fused RMSNorm module using our optimized CUDA kernel (keep in FP32 for numerical stability).
class RMSNorm(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.scale = d ** 0.5
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        # x has shape (..., d); flatten to 2D for row-wise processing.
        orig_dtype = x.dtype
        x = x.float()
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        out = cuda_module.fused_rmsnorm_cuda(x_2d, self.g, orig_shape[-1], self.scale, 1e-5)
        return out.reshape(orig_shape).to(orig_dtype)


# MambaBlock that fuses the state-space model (SSM) operations via our CUDA kernel.
class MambaBlock(nn.Module):
    def __init__(self, d_input: int, d_model: int, d_state: int = 16, d_discr: int = None,
                 ker_size: int = 4, parallel: bool = False, dtype=torch.float32) -> None:
        super().__init__()
        self.dtype = dtype
        d_discr = d_discr or d_model // 16
        self.in_proj = nn.Linear(d_input, 2 * d_model, bias=False).to(dtype)
        self.out_proj = nn.Linear(d_model, d_input, bias=False).to(dtype)
        self.s_B = nn.Linear(d_model, d_state, bias=False).to(dtype)
        self.s_C = nn.Linear(d_model, d_state, bias=False).to(dtype)
        self.s_D = nn.Sequential(
            nn.Linear(d_model, d_discr, bias=False),
            nn.Linear(d_discr, d_model, bias=False)
        ).to(dtype)
        # Depthwise conv1d with padding to preserve sequence length.
        self.conv = nn.Conv1d(d_model, d_model, ker_size, padding=ker_size - 1, groups=d_model, bias=True).to(dtype)
        # Parameter A: (d_model, d_state) initialized as [1, 2, ..., d_state] repeated for each d_model.
        self.A = nn.Parameter(torch.arange(1, d_state + 1, dtype=dtype).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model, dtype=dtype))
        self.parallel = parallel

    def forward(self, seq: Tensor, cache: tuple = None) -> tuple:
        b, l, d = seq.shape
        prev_hid, prev_inp = (None, None) if cache is None else cache
        a, b_proj = self.in_proj(seq).chunk(2, dim=-1)
        # Transpose for depthwise conv1d: (b, l, d) -> (b, d, l)
        x = a.transpose(1, 2).contiguous()
        if prev_inp is not None:
            x = torch.cat((prev_inp, x), dim=2)
        # Apply depthwise conv and slice to preserve original time length.
        a_conv = self.conv(x)[..., :l]
        a_conv = a_conv.transpose(1, 2).contiguous()  # back to (b, l, d)
        a_conv = silu(a_conv)
        a_out, hid = self.ssm(a_conv, prev_hid)
        b_proj = silu(b_proj)
        out = a_out * b_proj
        out = self.out_proj(out)
        if cache is not None:
            # Update cache: store hidden state and conv input (dropping the first time step).
            cache = (hid, x[..., 1:].contiguous())
        return out, cache

    def ssm(self, seq: Tensor, prev_hid: Tensor = None) -> tuple:
        # Compute recurrence factors.
        A_param = -self.A  # (d_model, d_state)
        D_param = self.D  # (d_model)
        delta = softplus(D_param + self.s_D(seq))  # shape: (b, l, d_model)
        B = self.s_B(seq)  # (b, l, d_state)
        C = self.s_C(seq)  # (b, l, d_state)
        # Compute A_bar: (b, l, d_model, d_state)
        A_bar = torch.exp(A_param)[None, None, :, :] * delta.unsqueeze(-1)
        # Compute X_bar: (b, l, d_model, d_state)
        B_bar = B.unsqueeze(2) * delta.unsqueeze(-1)
        X_bar = B_bar * seq.unsqueeze(-1)
        # Compute the recurrence: use fused CUDA kernel if not in parallel mode.
        hid, fused_out = self._hid_states(A_bar, X_bar, self.parallel, prev_hid, C=C, seq=seq)
        if fused_out is None:
            fused_out = torch.einsum("blds,bls->bld", hid, C) + D_param * seq
        return fused_out, hid


    def _hid_states(self, A: Tensor, X: Tensor, parallel: bool = False, prev_hid: Tensor = None,
                    C: Tensor = None, seq: Tensor = None) -> tuple:
        b, l, d, s = A.shape
        if prev_hid is not None:
            # With cached hidden state, compute recurrence elementwise.
            A_t = A.transpose(0, 1)  # (l, b, d, s)
            X_t = X.transpose(0, 1)
            hid_t = A_t * prev_hid.transpose(0, 1) + X_t
            hid = hid_t.transpose(0, 1).contiguous()  # (b, l, d, s)
            fused_out = torch.einsum("blds,bls->bld", hid, C) + self.D * seq
            return hid, fused_out
        if parallel:
            # In parallel mode, use the helper pscan.
            hid = pscan(A, X)
            return hid, None
        else:
            # Otherwise, use our fused CUDA kernel for the recurrence scan-reduce.
            hid_tensor = torch.empty((b, l, d, s), device=A.device, dtype=A.dtype)
            # Explicit measurement mode for the fused ssm kernel: "none" | "time" | "memory"
            MEASURE_MODE = "none"

            if MEASURE_MODE == "time":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                torch.cuda.synchronize()
                start.record()

                fused_out = cuda_module.fused_ssm_scan_reduce_cuda(
                    A.contiguous(), X.contiguous(), C.contiguous(), seq.contiguous(),
                    self.D, b, l, d, s, hid_tensor, self.dtype
                )

                end.record()
                torch.cuda.synchronize()
                print("SSM kernel time (ms):", start.elapsed_time(end))


            elif MEASURE_MODE == "memory":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                fused_out = cuda_module.fused_ssm_scan_reduce_cuda(
                    A.contiguous(), X.contiguous(), C.contiguous(), seq.contiguous(),
                    self.D, b, l, d, s, hid_tensor, self.dtype
                )

                torch.cuda.synchronize()
                peak_bytes = torch.cuda.max_memory_allocated()
                peak_mb = peak_bytes / (1024 ** 2)
                peak_gb = peak_bytes / (1024 ** 3)

                print(
                    f"SSM kernel peak memory: "
                    f"{peak_mb:,.2f} MB ({peak_gb:.2f} GB)"
                )

            else:  # MEASURE_MODE == "none"
                torch.cuda.nvtx.range_push("SSM_FUSED_KERNEL")
                fused_out = cuda_module.fused_ssm_scan_reduce_cuda(
                    A.contiguous(), X.contiguous(), C.contiguous(), seq.contiguous(),
                    self.D, b, l, d, s, hid_tensor, self.dtype
                )
                torch.cuda.nvtx.range_pop()

            return hid_tensor, fused_out

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


# The complete Model with the same interface.
class Model(nn.Module):
    def __init__(self, vocab_size: int = 16384, num_layers: int = 8, d_input: int = 1024, d_model: int = 1024,
                 d_state: int = 32, d_discr: int = None, ker_size: int = 4, parallel: bool = False, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype

        self.embedding = nn.Embedding(vocab_size, d_input).to(dtype)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MambaBlock(d_input, d_model, d_state, d_discr, ker_size, parallel, dtype),
                RMSNorm(d_input)
            ])
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_input, vocab_size, bias=False).to(dtype)


    def forward(self, tok: Tensor, cache: tuple = None) -> Tensor:
        tok = torch.atleast_2d(tok)
        seq = self.embedding(tok)
        for mamba, norm in self.layers:
            out, cache = mamba(norm(seq), cache)
            # Fused residual addition: seq = out + seq.
            seq = cuda_module.fused_residual_add_cuda(out.float(), seq.float()).to(self.dtype)  # keep FP32 for now
        return self.head(seq)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def get_inputs():
    return [torch.randint(0, 16384, (2, 256))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        type=str,
        default="none",
        choices=["none", "torch", "torch_tb"],
        help="Profiling mode: none | torch | torch_tb"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 (default: FP32)"
    )
    args = parser.parse_args()

    # -------------------------
    # Deterministic setup
    # -------------------------
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    dtype = torch.float16 if args.fp16 else torch.float32
    model = Model(dtype=dtype).cuda().eval()

    # Fixed inputs
    B, L = 16, 512
    x = torch.randint(
        0, 16384,
        (B, L),
        device="cuda",
        dtype=torch.long
    )

    # -------------------------
    # Warmup (always)
    # -------------------------
    for _ in range(3):
        model(x)

    torch.cuda.synchronize()

    # -------------------------
    # Run modes
    # -------------------------
    if args.profile == "none":
        # For Nsight Compute (ncu)
        model(x)
        torch.cuda.synchronize()

    elif args.profile == "torch":
        # Chrome trace
        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as prof:
            model(x)

        prof.export_chrome_trace("trace_cuda.json")
        print("Saved Chrome trace to trace_cuda.json")

    elif args.profile == "torch_tb":
        # TensorBoard-style detailed profiling
        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log")
        ) as prof:
            model(x)

        print(
            prof.key_averages()
            .table(sort_by="cuda_time_total", row_limit=20)
        )


if __name__ == "__main__":
    main()
