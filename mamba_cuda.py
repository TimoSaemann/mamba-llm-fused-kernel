import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>

extern "C" {

// Updated fast path specialized for d_state == 32 using half2 (16 threads).
__global__ void fused_ssm_precompute_scan_reduce_kernel_fp16_ds32_half2_official(
    const __half* __restrict__ A_param, // [d_model, 32]  where A_param = -exp(A_log)
    const __half* __restrict__ B_in,    // [B, L, 32]
    const __half* __restrict__ C_in,    // [B, L, 32]
    const __half* __restrict__ dt_in,   // [B, L, d_model] (pre-softplus)
    const __half* __restrict__ seq,     // [B, L, d_model] (post-conv+SiLU)
    const __half* __restrict__ D,       // [d_model]
    __half* __restrict__ out,           // [B, L, d_model]
    int Bsz, int L, int d_model
){
    // 1 block = one (b,d)
    int chain = blockIdx.x;
    int b_idx = chain / d_model;
    int d_idx = chain % d_model;
    int t = threadIdx.x; // 0..15 (each thread handles 2 state lanes)

    if (b_idx >= Bsz || t >= 16) return;

    constexpr int d_state = 32;
    const int stride_BC  = d_state;  // (b,l,s)
    const int stride_seq = d_model;  // (b,l,d)

    const int baseBC  = b_idx * (L * stride_BC);
    const int baseSeq = b_idx * (L * stride_seq) + d_idx;
    const int baseDT  = b_idx * (L * stride_seq) + d_idx;

    const float D_val = __half2float(D[d_idx]);

    // reinterpret A/B/C as half2 (packed pairs of state lanes)
    const __half2* __restrict__ A2 = reinterpret_cast<const __half2*>(A_param + d_idx * d_state);
    const __half2* __restrict__ B2 = reinterpret_cast<const __half2*>(B_in + baseBC);
    const __half2* __restrict__ C2 = reinterpret_cast<const __half2*>(C_in + baseBC);

    // Load A_param[d,:] once into registers (half2)
    __half2 aparam2 = A2[t];

    // Two independent hidden states per thread
    float2 h;
    h.x = 0.0f; h.y = 0.0f;

    for (int l = 0; l < L; ++l) {
        // Load scalar seq + dt once per timestep (one thread), broadcast.
        float seq_f, delta;
        if (t == 0) {
            seq_f = __half2float(seq[baseSeq + l * stride_seq]);
            float dt_f = __half2float(dt_in[baseDT + l * stride_seq]);
            // softplus(dt)
            // For stability: softplus(x)=log1p(exp(x)) if x<=0 else x+log1p(exp(-x))
            float x = dt_f;
            if (x <= 0.0f) delta = log1pf(expf(x));
            else           delta = x + log1pf(expf(-x));
        }
        seq_f = __shfl_sync(0xffffffff, seq_f, 0);
        delta = __shfl_sync(0xffffffff, delta, 0);

        // Vector loads for B and C for this timestep
        __half2 b2 = B2[l * 16 + t];   // 32 halves = 16 half2
        __half2 c2 = C2[l * 16 + t];

        // Convert half2 -> float2
        float2 b, c, aparam;
        b.x = __half2float(__low2half(b2));
        b.y = __half2float(__high2half(b2));
        c.x = __half2float(__low2half(c2));
        c.y = __half2float(__high2half(c2));
        aparam.x = __half2float(__low2half(aparam2));
        aparam.y = __half2float(__high2half(aparam2));

        // a = exp(delta * A_param)
        float2 a;
        a.x = __expf(delta * aparam.x);
        a.y = __expf(delta * aparam.y);

        // xterm = (B * delta) * seq
        float bd = delta * seq_f;
        float2 xterm;
        xterm.x = b.x * bd;
        xterm.y = b.y * bd;

        // recurrence (two lanes)
        h.x = fmaf(a.x, h.x, xterm.x);
        h.y = fmaf(a.y, h.y, xterm.y);

        // dot(h, C): partial sum per thread (two lanes)
        float prod = h.x * c.x + h.y * c.y;

        // reduce across 16 threads (covers 32 state lanes)
        for (int offset = 8; offset > 0; offset >>= 1) {
            prod += __shfl_down_sync(0xffffffff, prod, offset, 16);
        }

        if (t == 0) {
            float out_val = prod + D_val * seq_f;
            out[baseSeq + l * stride_seq] = __float2half(out_val);
        }
    }
}

// Generic FP16 precompute+scan+reduce kernel for arbitrary d_state.
//   A_param[d,s] = -exp(A_log[d,s])   (passed in as half)
//   delta = softplus(dt_in[b,l,d])    (dt_in is pre-softplus, includes dt_proj bias)
//   a_val = exp(delta * A_param[d,s])
//   x_val = (B[b,l,s] * delta) * seq[b,l,d]
//   h = a_val * h + x_val
//   out = sum_s h * C + D[d] * seq
__global__ void fused_ssm_precompute_scan_reduce_kernel_fp16_official(
    const __half* __restrict__ A_param, // [d_model, d_state]  (half) = -exp(A_log)
    const __half* __restrict__ B_in,    // [B, L, d_state]     (half)
    const __half* __restrict__ C_in,    // [B, L, d_state]     (half)
    const __half* __restrict__ dt_in,   // [B, L, d_model]     (half) pre-softplus
    const __half* __restrict__ seq,     // [B, L, d_model]     (half) post-conv+SiLU
    const __half* __restrict__ D,       // [d_model]           (half)
    __half* __restrict__ out,           // [B, L, d_model]     (half)
    int Bsz, int L, int d_model, int d_state
) {
    int chain = blockIdx.x;            // one (b,d) per block
    int b_idx = chain / d_model;
    int d_idx = chain % d_model;
    int s_idx = threadIdx.x;           // one thread per state lane

    if (b_idx >= Bsz || s_idx >= d_state) return;

    // Strides
    const int stride_BC  = d_state;   // per timestep for (b,l,s)
    const int stride_seq = d_model;   // per timestep for (b,l,d)

    // Base pointers
    const int baseBC  = b_idx * (L * stride_BC);           // (b,*,s)
    const int baseSeq = b_idx * (L * stride_seq) + d_idx;  // (b,*,d)
    const int baseDT  = b_idx * (L * stride_seq) + d_idx;  // (b,*,d)

    // Constants for this (b,d)
    const float D_val = __half2float(D[d_idx]);
    const int A_base = d_idx * d_state; // A_param[d_idx, s]

    float h = 0.0f;

    // shared for cross-warp reduction
    __shared__ float warp_sums[32];  // supports up to 32 warps => d_state up to 1024

    for (int l = 0; l < L; ++l) {
        // Load dt and seq for this (b,l,d)
        float dt_val  = __half2float(dt_in[baseDT + l * stride_seq]);
        float seq_val = __half2float(seq[baseSeq + l * stride_seq]);

        // delta = softplus(dt)
        float delta;
        if (dt_val <= 0.0f) delta = log1pf(expf(dt_val));
        else                delta = dt_val + log1pf(expf(-dt_val));

        // a_val = exp(delta * A_param[d,s])
        float Aps = __half2float(A_param[A_base + s_idx]);
        float a_val = __expf(delta * Aps);

        // x_val = (B * delta) * seq
        float b_val = __half2float(B_in[baseBC + l * stride_BC + s_idx]);
        float x_val = (b_val * delta) * seq_val;

        // recurrence
        h = fmaf(a_val, h, x_val);

        // reduce dot(h, C) across d_state lanes
        float c_val = __half2float(C_in[baseBC + l * stride_BC + s_idx]);
        float prod = h * c_val;

        int lane = s_idx & 31;
        int warp = s_idx >> 5;
        int nwarps = (d_state + 31) >> 5;

        float sum = prod;

        // intra-warp reduce
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // cross-warp reduce via shared
        if (lane == 0) warp_sums[warp] = sum;
        __syncthreads();

        if (warp == 0) {
            float total = (lane < nwarps) ? warp_sums[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                total += __shfl_down_sync(0xffffffff, total, offset);
            }
            if (lane == 0) {
                float out_val = total + D_val * seq_val;
                out[baseSeq + l * stride_seq] = __float2half(out_val);
            }
        }

        __syncthreads();
    }
}

__device__ __forceinline__ float softplus_f_stable(float x) {
    // stable softplus
    return (x <= 0.0f) ? log1pf(expf(x)) : (x + log1pf(expf(-x)));
}

// FP32 precompute+scan+reduce, arbitrary d_state
__global__ void fused_ssm_precompute_scan_reduce_kernel_fp32_official(
    const float* __restrict__ A_param, // [d_model, d_state]  A_param = -exp(A_log)
    const float* __restrict__ B_in,    // [B, L, d_state]
    const float* __restrict__ C_in,    // [B, L, d_state]
    const float* __restrict__ dt_in,   // [B, L, d_model]  pre-softplus (includes dt_proj bias)
    const float* __restrict__ seq,     // [B, L, d_model]  post-conv+SiLU
    const float* __restrict__ D,       // [d_model]
    float* __restrict__ out,           // [B, L, d_model]
    int Bsz, int L, int d_model, int d_state
){
    // one block per (b,d)
    int chain = blockIdx.x;
    int b_idx = chain / d_model;
    int d_idx = chain % d_model;
    int s_idx = threadIdx.x; // one thread per state lane

    if (b_idx >= Bsz || s_idx >= d_state) return;

    const int stride_BC  = d_state;  // (b,l,s)
    const int stride_seq = d_model;  // (b,l,d)

    const int baseBC  = b_idx * (L * stride_BC);
    const int baseSeq = b_idx * (L * stride_seq) + d_idx;
    const int baseDT  = b_idx * (L * stride_seq) + d_idx;

    const float D_val = D[d_idx];
    const int A_base  = d_idx * d_state;

    float h = 0.0f;

    // cross-warp reduction scratch
    __shared__ float warp_sums[32]; // supports up to 32 warps => d_state up to 1024

    for (int l = 0; l < L; ++l) {
        const float dt_val  = dt_in[baseDT + l * stride_seq];
        const float seq_val = seq[baseSeq + l * stride_seq];

        // delta = softplus(dt)
        const float delta = softplus_f_stable(dt_val);

        // a = exp(delta * A_param[d,s])
        const float Aps = A_param[A_base + s_idx];
        const float a_val = __expf(delta * Aps);

        // x = (B * delta) * seq
        const float b_val = B_in[baseBC + l * stride_BC + s_idx];
        const float x_val = (b_val * delta) * seq_val;

        // recurrence
        h = fmaf(a_val, h, x_val);

        // dot contribution
        const float c_val = C_in[baseBC + l * stride_BC + s_idx];
        float prod = h * c_val;

        // reduce across d_state lanes (same pattern as your fp16 generic kernel)
        int lane = s_idx & 31;
        int warp = s_idx >> 5;
        int nwarps = (d_state + 31) >> 5;

        float sum = prod;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) warp_sums[warp] = sum;
        __syncthreads();

        if (warp == 0) {
            float total = (lane < nwarps) ? warp_sums[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                total += __shfl_down_sync(0xffffffff, total, offset);
            }
            if (lane == 0) {
                out[baseSeq + l * stride_seq] = total + D_val * seq_val;
            }
        }
        __syncthreads();
    }
}

// ----------------------------------------------------------------
// Fused SSM Scan-Reduce Kernel (FP32)
__global__ void fused_ssm_scan_reduce_kernel(const float* __restrict__ A,
                                             const float* __restrict__ X,
                                             const float* __restrict__ C,
                                             const float* __restrict__ seq,
                                             const float* __restrict__ D,
                                             float* __restrict__ out,
                                             int B, int L, int d_model, int d_state) {
    int chain = blockIdx.x;
    const int b_idx = chain / d_model;
    const int d_idx = chain % d_model;
    const int tid = threadIdx.x;

    float h_val = 0.0f;
    const float D_val = D[d_idx];

    const int stride_per_timestep_A = d_model * d_state;
    const int stride_per_timestep_C = d_state;
    const int stride_per_timestep_seq = d_model;

    const int baseA = b_idx * (L * stride_per_timestep_A) + d_idx * d_state;
    const int baseC = b_idx * (L * stride_per_timestep_C);
    const int baseSeq = b_idx * (L * stride_per_timestep_seq) + d_idx;

    for (int l = 0; l < L; ++l) {
        const int idx = baseA + l * stride_per_timestep_A + tid;
        const float a_val = A[idx];
        const float x_val = X[idx];

        h_val = a_val * h_val + x_val;

        const int cidx = baseC + l * stride_per_timestep_C + tid;
        const float c_val = C[cidx];
        float prod = h_val * c_val;

        for (int offset = min(16, d_state / 2); offset > 0; offset /= 2) {
            prod += __shfl_down_sync(0xffffffff, prod, offset);
        }

        if (d_state > 32) {
            __shared__ float sdata[32];
            int lane = tid % 32;
            int warp_id = tid / 32;
            if (lane == 0) sdata[warp_id] = prod;
            __syncthreads();
            if (tid == 0) {
                float final_prod = 0;
                for (int i = 0; i < (d_state + 31) / 32; i++) final_prod += sdata[i];
                prod = final_prod;
            }
        }

        if (tid == 0) {
            const int seqidx = baseSeq + l * stride_per_timestep_seq;
            const float seq_val = seq[seqidx];
            out[seqidx] = prod + D_val * seq_val;
        }
    }
}

// ----------------------------------------------------------------
// Fused SSM Scan-Reduce Kernel (FP16)
__global__ void fused_ssm_scan_reduce_kernel_fp16(
    const __half* __restrict__ A,
    const __half* __restrict__ X,
    const __half* __restrict__ C,
    const __half* __restrict__ seq,
    const __half* __restrict__ D,
    __half* __restrict__ out,
    int B, int L, int d_model, int d_state
) {
    int chain = blockIdx.x;
    const int b_idx = chain / d_model;
    const int d_idx = chain % d_model;
    const int tid = threadIdx.x;

    __half h_val = __float2half(0.0f);
    const __half D_val = D[d_idx];

    const int stride_per_timestep_A = d_model * d_state;
    const int stride_per_timestep_C = d_state;
    const int stride_per_timestep_seq = d_model;

    const int baseA = b_idx * (L * stride_per_timestep_A) + d_idx * d_state;
    const int baseC = b_idx * (L * stride_per_timestep_C);
    const int baseSeq = b_idx * (L * stride_per_timestep_seq) + d_idx;

    for (int l = 0; l < L; ++l) {
        const int idx = baseA + l * stride_per_timestep_A + tid;
        const __half a_val = A[idx];
        const __half x_val = X[idx];

        h_val = __hfma(a_val, h_val, x_val);

        const int cidx = baseC + l * stride_per_timestep_C + tid;
        const half c_val = C[cidx];

        float prod = __half2float(h_val) * __half2float(c_val);

        for (int offset = min(16, d_state / 2); offset > 0; offset /= 2) {
            prod += __shfl_down_sync(0xffffffff, prod, offset);
        }

        if (d_state > 32) {
            __shared__ float sdata[32];
            int lane = tid % 32;
            int warp_id = tid / 32;
            if (lane == 0) sdata[warp_id] = prod;
            __syncthreads();
            if (tid == 0) {
                float final_prod = 0;
                for (int i = 0; i < (d_state + 31) / 32; i++) final_prod += sdata[i];
                prod = final_prod;
            }
        }

        if (tid == 0) {
            const int seqidx = baseSeq + l * stride_per_timestep_seq;
            const half seq_val = seq[seqidx];
            float out_val = prod + __half2float(D_val) * __half2float(seq_val);
            out[seqidx] = __float2half(out_val);
        }
    }
}


// Fused Residual Addition Kernel
__global__ void fused_residual_add_kernel(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          float* __restrict__ out,
                                          int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] + b[idx];
}


// Fused RMSNorm Kernel
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

// Wrapper for fused_ssm_precompute_scan_reduce_fp16_cuda
torch::Tensor fused_ssm_precompute_scan_reduce_fp16_cuda(
    torch::Tensor A_param, // [d_model, d_state] half  (A_param = -exp(A_log))
    torch::Tensor B_in,    // [B, L, d_state] half
    torch::Tensor C_in,    // [B, L, d_state] half
    torch::Tensor dt_in,   // [B, L, d_model] half (pre-softplus)
    torch::Tensor seq,     // [B, L, d_model] half
    torch::Tensor D,       // [d_model] half
    int Bsz, int L, int d_model, int d_state
) {
    auto out = torch::zeros({Bsz, L, d_model}, seq.options());

    dim3 grid(Bsz * d_model);

    if (d_state == 32) {
        dim3 block(16); // half2 path
        fused_ssm_precompute_scan_reduce_kernel_fp16_ds32_half2_official<<<grid, block>>>(
            (__half*)A_param.data_ptr<c10::Half>(),
            (__half*)B_in.data_ptr<c10::Half>(),
            (__half*)C_in.data_ptr<c10::Half>(),
            (__half*)dt_in.data_ptr<c10::Half>(),
            (__half*)seq.data_ptr<c10::Half>(),
            (__half*)D.data_ptr<c10::Half>(),
            (__half*)out.data_ptr<c10::Half>(),
            Bsz, L, d_model
        );
    } else {
        dim3 block(d_state);
        fused_ssm_precompute_scan_reduce_kernel_fp16_official<<<grid, block>>>(
            (__half*)A_param.data_ptr<c10::Half>(),
            (__half*)B_in.data_ptr<c10::Half>(),
            (__half*)C_in.data_ptr<c10::Half>(),
            (__half*)dt_in.data_ptr<c10::Half>(),
            (__half*)seq.data_ptr<c10::Half>(),
            (__half*)D.data_ptr<c10::Half>(),
            (__half*)out.data_ptr<c10::Half>(),
            Bsz, L, d_model, d_state
        );
    }

    return out;
}

// Wrapper for fused_ssm_precompute_scan_reduce_fp32_cuda
torch::Tensor fused_ssm_precompute_scan_reduce_fp32_cuda(
    torch::Tensor A_param, // [d_model, d_state] float
    torch::Tensor B_in,    // [B, L, d_state] float
    torch::Tensor C_in,    // [B, L, d_state] float
    torch::Tensor dt_in,   // [B, L, d_model] float (pre-softplus)
    torch::Tensor seq,     // [B, L, d_model] float
    torch::Tensor D,       // [d_model] float
    int Bsz, int L, int d_model, int d_state
) {
    auto out = torch::zeros({Bsz, L, d_model}, seq.options());

    dim3 grid(Bsz * d_model);
    dim3 block(d_state);
    
    fused_ssm_precompute_scan_reduce_kernel_fp32_official<<<grid, block>>>(
        A_param.data_ptr<float>(),
        B_in.data_ptr<float>(),
        C_in.data_ptr<float>(),
        dt_in.data_ptr<float>(),
        seq.data_ptr<float>(),
        D.data_ptr<float>(),
        out.data_ptr<float>(),
        Bsz, L, d_model, d_state
    );
    
    return out;
}

// Wrapper for fused_ssm_scan_reduce_cuda (fallback)
torch::Tensor fused_ssm_scan_reduce_cuda(
    torch::Tensor A, torch::Tensor X, torch::Tensor C,
    torch::Tensor seq, torch::Tensor D,
    int B, int L, int d_model, int d_state
) {
    auto dtype = A.scalar_type();
    auto out = torch::zeros({B, L, d_model}, seq.options());

    dim3 grid(B * d_model);
    dim3 block(d_state);
    if (dtype == c10::kFloat) {
            fused_ssm_scan_reduce_kernel<<<grid, block>>>(
            A.data_ptr<float>(), X.data_ptr<float>(),
            C.data_ptr<float>(), seq.data_ptr<float>(),
            D.data_ptr<float>(), out.data_ptr<float>(),
            B, L, d_model, d_state
        );
    
    } else if (dtype == c10::kHalf) {
        fused_ssm_scan_reduce_kernel_fp16<<<grid, block>>>(
            reinterpret_cast<__half*>(A.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(X.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(C.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(seq.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(D.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(out.data_ptr<c10::Half>()),
            B, L, d_model, d_state
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for fused_ssm_scan_reduce_cuda");
    }
    
    return out;
}

// Wrapper for Fused Residual Addition Kernel.
torch::Tensor fused_residual_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::empty_like(a);
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    fused_residual_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

// Wrapper for Fused RMSNorm Kernel.
torch::Tensor fused_rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, int d, float scale, float eps) {
    auto N = x.size(0);
    auto out = torch::empty_like(x);
    int block_size = 256;
    if (d < block_size) block_size = ((d + 31) / 32) * 32;
    dim3 grid(N);
    dim3 block(block_size);
    fused_rmsnorm_kernel<<<grid, block>>>(x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), d, scale, eps);
    return out;
}


} // extern "C"
'''

cpp_source = r'''
extern "C" {
torch::Tensor fused_ssm_precompute_scan_reduce_fp16_cuda(
    torch::Tensor A_exp, torch::Tensor B_in, torch::Tensor C_in,
    torch::Tensor sD_in, torch::Tensor seq, torch::Tensor D,
    int B, int L, int d_model, int d_state
);
torch::Tensor fused_ssm_precompute_scan_reduce_fp32_cuda(
    torch::Tensor A_exp, torch::Tensor B_in, torch::Tensor C_in,
    torch::Tensor sD_in, torch::Tensor seq, torch::Tensor D,
    int B, int L, int d_model, int d_state
);
torch::Tensor fused_ssm_scan_reduce_cuda(
    torch::Tensor A, torch::Tensor X, torch::Tensor C,
    torch::Tensor seq, torch::Tensor D,
    int B, int L, int d_model, int d_state
);
torch::Tensor fused_residual_add_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor fused_rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, int d, float scale, float eps);
}

'''

# Compile the inline CUDA module with our optimized kernels.
cuda_module = load_inline(name="cuda_kernels_optimized",
                          cpp_sources=cpp_source,
                          cuda_sources=cuda_source,
                          functions=["fused_ssm_precompute_scan_reduce_fp16_cuda",
                                     "fused_ssm_precompute_scan_reduce_fp32_cuda", "fused_ssm_scan_reduce_cuda",
                                     "fused_residual_add_cuda", "fused_rmsnorm_cuda"],
                          verbose=True,
                          extra_cflags=[''],
                          extra_ldflags=[''])


# ---------------- Fused RMSNorm ----------------
class RMSNorm(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.scale = d ** 0.5
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        orig_shape = x_f.shape
        x_2d = x_f.reshape(-1, orig_shape[-1]).contiguous()
        out_2d = cuda_module.fused_rmsnorm_cuda(x_2d, self.g, orig_shape[-1], self.scale, 1e-5)
        return out_2d.reshape(orig_shape).to(orig_dtype)


# ---------------- CUDA Mamba Block (official-matching math) ----------------
class MambaBlock(nn.Module):
    """
    Official-matching math, CUDA-optimized execution.

    - in_proj -> split (a, z)
    - depthwise conv + SiLU on a -> x
    - packed projection: s_BCD(x) -> [dt_lowrank | B | C]
      dt = dt_proj(dt_lowrank), delta = softplus(dt)
    - A_param = -exp(A_log)
    - scan recurrence uses exp(delta * A_param)
    - y = (C @ h) + D * x
    - gate: y *= silu(z)
    - out_proj
    """
    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_state: int = 32,
        ker_size: int = 4,
        dtype: torch.dtype = torch.float16,
        big_fuse: bool = True,
    ) -> None:
        super().__init__()
        assert d_input == d_model, "This implementation assumes d_input == d_model."

        self.dtype = dtype
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.big_fuse = big_fuse

        # Projections
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False).to(dtype)
        self.out_proj = nn.Linear(d_model, d_model, bias=False).to(dtype)

        # Depthwise conv
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            ker_size,
            padding=ker_size - 1,
            groups=d_model,
            bias=True,
        ).to(dtype)

        # Packed x_proj equivalent: (dt_rank + 2*d_state)
        self.s_BCD = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False).to(dtype)

        # dt_proj: dt_rank -> d_model (with bias)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True).to(dtype)

        # A_log like official (keep fp32)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip D
        self.D = nn.Parameter(torch.ones(d_model, dtype=dtype))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B, L, D = seq.shape

        # (a, z) split
        a, z = self.in_proj(seq).chunk(2, dim=-1)

        # Conv + SiLU on a  -> x is the SSM input
        x = a.transpose(1, 2).contiguous()           # (B, D, L)
        x = self.conv(x)[..., :L]                    # (B, D, L)
        x = F.silu(x).transpose(1, 2).contiguous()   # (B, L, D)

        # Packed projections from x: [dt_low | B | C]
        x_flat = x.reshape(B * L, D).contiguous()
        dbl = self.s_BCD(x_flat).view(B, L, -1)
        dt_low, Bv, Cv = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # dt (pre-softplus), delta (softplus)
        dt = self.dt_proj(dt_low.reshape(B * L, self.dt_rank)).view(B, L, D)
        delta = F.softplus(dt)

        y = self.ssm_forward_fp16_or_fp32(x=x, dt=dt, delta=delta, Bv=Bv, Cv=Cv)

        # Gate + out proj
        y = y * F.silu(z)
        return self.out_proj(y)

    def ssm_forward_fp16_or_fp32(
        self,
        x: torch.Tensor,      # (B, L, D) post-conv + SiLU
        dt: torch.Tensor,     # (B, L, D) pre-softplus
        delta: torch.Tensor,  # (B, L, D) softplus(dt)
        Bv: torch.Tensor,     # (B, L, S)
        Cv: torch.Tensor,     # (B, L, S)
    ) -> torch.Tensor:
        B, L, D = x.shape
        S = self.d_state

        # A_param = -exp(A_log)
        A_param = (-torch.exp(self.A_log.float())).to(dtype=x.dtype).contiguous()  # (D, S)

        if self.big_fuse:
            if x.dtype == torch.float16:
                return cuda_module.fused_ssm_precompute_scan_reduce_fp16_cuda(
                    A_param,
                    Bv.contiguous(),
                    Cv.contiguous(),
                    dt.contiguous(),
                    x.contiguous(),
                    self.D.contiguous(),
                    B, L, D, S,
                )
            if x.dtype == torch.float32:
                return cuda_module.fused_ssm_precompute_scan_reduce_fp32_cuda(
                    A_param.float(),
                    Bv.float().contiguous(),
                    Cv.float().contiguous(),
                    dt.float().contiguous(),
                    x.float().contiguous(),
                    self.D.float().contiguous(),
                    B, L, D, S,
                )

        # Fallback: materialize A_bar / X_bar then use fused scan-reduce
        A_bar = torch.exp(delta.unsqueeze(-1) * A_param.unsqueeze(0).unsqueeze(0))  # (B, L, D, S)
        X_bar = (Bv.unsqueeze(2) * delta.unsqueeze(-1)) * x.unsqueeze(-1)           # (B, L, D, S)

        return cuda_module.fused_ssm_scan_reduce_cuda(
            A_bar.contiguous(),
            X_bar.contiguous(),
            Cv.contiguous(),
            x.contiguous(),
            self.D,
            B, L, D, S,
        )

    @torch.no_grad()
    def copy_from_official_block(self, official_block) -> None:
        """official_block: OfficialMambaBlock from mamba_official.py"""
        o = official_block.mamba

        self.in_proj.weight.copy_(o.in_proj.weight.to(self.in_proj.weight.dtype))
        self.out_proj.weight.copy_(o.out_proj.weight.to(self.out_proj.weight.dtype))

        self.conv.weight.copy_(o.conv1d.weight.to(self.conv.weight.dtype))
        self.conv.bias.copy_(o.conv1d.bias.to(self.conv.bias.dtype))

        self.A_log.copy_(o.A_log.float())
        self.D.copy_(o.D.to(self.D.dtype))

        self.dt_proj.weight.copy_(o.dt_proj.weight.to(self.dt_proj.weight.dtype))
        self.dt_proj.bias.copy_(o.dt_proj.bias.to(self.dt_proj.bias.dtype))

        self.s_BCD.weight.copy_(o.x_proj.weight.to(self.s_BCD.weight.dtype))


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16384,
        num_layers: int = 8,
        d_model: int = 1024,
        d_state: int = 32,
        ker_size: int = 4,
        dtype: torch.dtype = torch.float16,
        big_fuse: bool = True,
    ) -> None:
        super().__init__()
        self.dtype = dtype

        self.embedding = nn.Embedding(vocab_size, d_model).to(dtype)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MambaBlock(d_model, d_model, d_state, ker_size, dtype=dtype, big_fuse=big_fuse),
                RMSNorm(d_model),
            ])
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=False).to(dtype)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        tok = torch.atleast_2d(tok)
        seq = self.embedding(tok)

        for mamba, norm in self.layers:
            out = mamba(norm(seq))
            seq = cuda_module.fused_residual_add_cuda(out.float(), seq.float()).to(self.dtype)

        return self.head(seq)

    @torch.no_grad()
    def copy_from_official(self, official_model) -> None:
        self.embedding.weight.copy_(official_model.embedding.weight.to(self.embedding.weight.dtype))
        self.head.weight.copy_(official_model.head.weight.to(self.head.weight.dtype))

        for (c_block, _), (o_block, _) in zip(self.layers, official_model.layers):
            c_block.copy_from_official_block(o_block)
