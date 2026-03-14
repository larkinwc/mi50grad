/**
 * Fused Skip-Connection + RMSNorm + INT4 GEMV kernel.
 *
 * Fuses the decode path in a SINGLE kernel launch:
 *   1. hidden_out[i] = hidden[i] + residual[i]  (skip connection, output to separate buffer)
 *   2. Each block independently computes RMSNorm from hidden+residual into per-block LDS
 *   3. GEMV: out = rmsnorm(hidden+residual) @ W_int4^T
 *
 * Eliminating the intermediate write of normalized activations to HBM saves
 * ~20KB of HBM traffic per layer (for dim=5120).
 *
 * Interface design:
 *   - `hidden` is READ-ONLY in this kernel (passed as const)
 *   - `hidden_out` receives the updated hidden state (hidden + residual)
 *   - `hidden_out` CAN be the same pointer as `hidden` (they are never both read+written
 *     by different blocks simultaneously: all blocks read `hidden` in Phase 1;
 *     block 0 writes `hidden_out` in Phase 3 AFTER Phase 2 is complete)
 *   - When hidden_out == hidden: blocks 128+ (if any) may read the written-back
 *     value from block 0 if they start Phase 1 late. For safety, callers should
 *     pass hidden_out != hidden (or use skip_rmsnorm_v3 for the separate update).
 *   - Simplest correct usage: pass hidden_out pointing to a separate buffer.
 *
 * ACTUAL SAFE DESIGN: Use a staging buffer approach within each block.
 *   Each block independently:
 *     Phase 1: Read hidden[K]+residual[K] from HBM (READ-ONLY), compute RMSNorm,
 *              store normalized activations in per-block LDS.
 *     Phase 2: GEMV from LDS (no HBM access for activations).
 *     Phase 3 (block 0 only): Write hidden_out[i] = hidden[i]+residual[i] from LDS.
 *              This is safe ONLY if hidden_out != hidden. If hidden_out == hidden,
 *              there may be races with blocks reading hidden in Phase 1.
 *
 * RECOMMENDATION: For safe in-place update, call skip_rmsnorm_v3 separately first,
 * then use gemv_int4_v3 on the norm output. OR pass hidden_out as a distinct buffer.
 *
 * LDS layout per block (for K=5120):
 *   [0, K*2):           lds_hval (FP16) — h+r pre-normalization (for hidden_out writeback)
 *   [K*2, K*4):         lds_A (FP16) — normalized activations (for GEMV)
 *   [K*4, K*4+16):      s_warp (float[4]) — RMSNorm cross-warp reduction
 *   [K*4+16, K*4+1040): s_reduce (float[256]) — GEMV K-split reduction
 *   Total: 21536 bytes < 64KB ✓
 *
 * Weight format (GPTQ-compatible):
 *   B_q4[K/8, N] uint32, scales[K/gs, N] FP16, zeros[K/gs, N] FP16
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---------------------------------------------------------------------------
// INT4 dequant+dot using v_bfe_u32 — reads FP16 activation from LDS.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dequant_dot8_lds(
    unsigned int packed,
    float cur_scale,
    float cur_zero,
    const __half* __restrict__ A_lds,
    unsigned int a_base
) {
    __half2 a01 = *reinterpret_cast<const __half2*>(A_lds + a_base);
    __half2 a23 = *reinterpret_cast<const __half2*>(A_lds + a_base + 2);
    __half2 a45 = *reinterpret_cast<const __half2*>(A_lds + a_base + 4);
    __half2 a67 = *reinterpret_cast<const __half2*>(A_lds + a_base + 6);

    float a0 = __half2float(__low2half(a01));
    float a1 = __half2float(__high2half(a01));
    float a2 = __half2float(__low2half(a23));
    float a3 = __half2float(__high2half(a23));
    float a4 = __half2float(__low2half(a45));
    float a5 = __half2float(__high2half(a45));
    float a6 = __half2float(__low2half(a67));
    float a7 = __half2float(__high2half(a67));

    float zs    = cur_zero * cur_scale;
    float a_sum = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;

    float acc = 0.0f;
    acc += (float)__builtin_amdgcn_ubfe(packed,  0, 4) * a0;
    acc += (float)__builtin_amdgcn_ubfe(packed,  4, 4) * a1;
    acc += (float)__builtin_amdgcn_ubfe(packed,  8, 4) * a2;
    acc += (float)__builtin_amdgcn_ubfe(packed, 12, 4) * a3;
    acc += (float)__builtin_amdgcn_ubfe(packed, 16, 4) * a4;
    acc += (float)__builtin_amdgcn_ubfe(packed, 20, 4) * a5;
    acc += (float)__builtin_amdgcn_ubfe(packed, 24, 4) * a6;
    acc += (float)__builtin_amdgcn_ubfe(packed, 28, 4) * a7;

    return acc * cur_scale - zs * a_sum;
}

// ---------------------------------------------------------------------------
// Fused kernel template.
//
// IMPORTANT: `hidden` is CONST (READ-ONLY). `hidden_out` receives h+r update.
// For correct behavior, hidden_out MUST be a different buffer from hidden
// (or at minimum, no other concurrent block reads `hidden_out` during Phase 1).
//
// A SAFE caller pattern:
//   static half hidden_buf_A[K], hidden_buf_B[K];
//   // alternating: even layers write to buf_A, odd layers write to buf_B
//   fused_skip_rmsnorm_gemv(..., hidden_out=buf_A, hidden=buf_B, ...)
//
// ---------------------------------------------------------------------------
template <int THREADS_PER_COL>
__device__ __forceinline__ void fused_skip_rmsnorm_gemv_impl(
    __half* __restrict__ out_gemv,
    __half* __restrict__ hidden_out,        // write: updated hidden state
    const __half* __restrict__ hidden,      // read: original hidden (const)
    const __half* __restrict__ residual,
    const __half* __restrict__ weight,
    float eps,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    const unsigned int COLS_PER_WG = 256 / THREADS_PER_COL;

    extern __shared__ char lds_raw[];
    __half* lds_hval   = reinterpret_cast<__half*>(lds_raw);              // [0, K*2)
    __half* lds_A      = reinterpret_cast<__half*>(lds_raw + K * 2);      // [K*2, K*4)
    float*  s_warp     = reinterpret_cast<float*>(lds_raw + K * 4);       // [K*4, K*4+16)
    float*  s_reduce   = reinterpret_cast<float*>(lds_raw + K * 4 + 16);  // [K*4+16, K*4+1040)

    const unsigned int wf_id = threadIdx.x / 64;
    const unsigned int lane  = threadIdx.x % 64;

    // ====================================================================
    // PHASE 1: Skip + RMSNorm — reads hidden and residual (READ-ONLY)
    // All blocks execute Phase 1 independently using per-block LDS.
    // ====================================================================
    float sum_sq = 0.0f;

    for (unsigned int i = threadIdx.x; i < K; i += 256) {
        float h   = __half2float(hidden[i]);
        float r   = __half2float(residual[i]);
        float val = h + r;
        lds_hval[i] = __float2half(val);   // store pre-norm h+r for writeback
        lds_A[i]    = __float2half(val);   // will be overwritten by normalized value
        sum_sq += val * val;
    }

    // Warp butterfly reduction (wave64)
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);

    // Cross-warp reduction
    if (lane == 0) s_warp[wf_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < 4) {
        float total = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
        s_warp[threadIdx.x] = total;
    }
    __syncthreads();

    float rms_inv = rsqrtf(s_warp[0] / (float)K + eps);

    // Normalize into lds_A
    for (unsigned int i = threadIdx.x; i < K; i += 256) {
        float val = __half2float(lds_A[i]);
        float w   = __half2float(weight[i]);
        lds_A[i]  = __float2half(val * rms_inv * w);
    }
    __syncthreads();   // lds_A fully written

    // ====================================================================
    // PHASE 2: INT4 GEMV — activation from lds_A (no HBM read)
    // ====================================================================
    const unsigned int col_in_wg  = threadIdx.x % COLS_PER_WG;
    const unsigned int k_split_id = threadIdx.x / COLS_PER_WG;
    const unsigned int col        = blockIdx.x * COLS_PER_WG + col_in_wg;

    float acc = 0.0f;

    if (col < N) {
        const unsigned int num_k_groups    = K >> 3;
        const unsigned int groups_per_scale = group_size >> 3;
        const unsigned int gps_log2 = 31u - __builtin_clz(groups_per_scale);

        const unsigned int kg_start = (num_k_groups * k_split_id) / THREADS_PER_COL;
        const unsigned int kg_end   = (num_k_groups * (k_split_id + 1)) / THREADS_PER_COL;

        float cur_scale = 0.0f, cur_zero = 0.0f;
        unsigned int last_sg = 0xFFFFFFFF;

        for (unsigned int kg = kg_start; kg < kg_end; kg++) {
            const unsigned int sg = kg >> gps_log2;
            if (sg != last_sg) {
                cur_scale = __half2float(scales[sg * N + col]);
                cur_zero  = __half2float(zeros[sg * N + col]);
                last_sg = sg;
            }
            const unsigned int packed = B_q4[kg * N + col];
            acc += dequant_dot8_lds(packed, cur_scale, cur_zero, lds_A, kg << 3);
        }
    }

    // K-split reduction
    s_reduce[threadIdx.x] = acc;
    __syncthreads();

    if (k_split_id == 0 && col < N) {
        float sum = s_reduce[col_in_wg];
        #pragma unroll
        for (int s = 1; s < THREADS_PER_COL; s++) {
            sum += s_reduce[col_in_wg + s * COLS_PER_WG];
        }
        out_gemv[col] = __float2half(sum);
    }

    // ====================================================================
    // PHASE 3: Hidden writeback — only block 0
    // hidden_out MUST be a different buffer from hidden for race-free behavior.
    // If hidden_out == hidden, this write races with other blocks' Phase 1 reads.
    // ====================================================================
    if (blockIdx.x == 0) {
        for (unsigned int i = threadIdx.x; i < K; i += 256) {
            hidden_out[i] = lds_hval[i];
        }
    }
}

// Public kernel variants

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void fused_skip_rmsnorm_gemv_t4(
    __half* __restrict__ out_gemv,
    __half* __restrict__ hidden_out,
    const __half* __restrict__ hidden,
    const __half* __restrict__ residual,
    const __half* __restrict__ weight,
    float eps,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    fused_skip_rmsnorm_gemv_impl<4>(
        out_gemv, hidden_out, hidden, residual, weight, eps,
        B_q4, scales, zeros, K, N, group_size);
}

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void fused_skip_rmsnorm_gemv_t8(
    __half* __restrict__ out_gemv,
    __half* __restrict__ hidden_out,
    const __half* __restrict__ hidden,
    const __half* __restrict__ residual,
    const __half* __restrict__ weight,
    float eps,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    fused_skip_rmsnorm_gemv_impl<8>(
        out_gemv, hidden_out, hidden, residual, weight, eps,
        B_q4, scales, zeros, K, N, group_size);
}

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void fused_skip_rmsnorm_gemv_t16(
    __half* __restrict__ out_gemv,
    __half* __restrict__ hidden_out,
    const __half* __restrict__ hidden,
    const __half* __restrict__ residual,
    const __half* __restrict__ weight,
    float eps,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    fused_skip_rmsnorm_gemv_impl<16>(
        out_gemv, hidden_out, hidden, residual, weight, eps,
        B_q4, scales, zeros, K, N, group_size);
}
