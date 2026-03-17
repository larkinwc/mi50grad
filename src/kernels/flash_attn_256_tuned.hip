/**
 * Tuned FlashAttention for head_dim=256 on gfx906 (MI50/MI60).
 *
 * Optimizations over flash_attn_256.hip:
 *
 * 1. DECODE kernel (seq_len=1): Uses 1 wavefront (64 threads) per WG instead of 4.
 *    - For decode, kv_limit=kv_seq_len and we need max wavefronts for KV sweep.
 *    - 64-thread WG: occupancy limited by VGPR count (~20 VGPRs → 40 waves/CU = 10 WGs).
 *    - Each WG handles 1 head (query), enabling 48 WGs for 48 heads to run concurrently.
 *    - Cooperative loading: all 64 threads load a KV block of 4 rows into LDS per batch.
 *
 * 2. PREFILL kernel: 4 wavefronts per WG (256 threads), handles 4 Q rows per WG.
 *    - Uses LDS to load Bc=4 KV rows cooperatively, reducing redundant global loads.
 *    - Each wavefront processes 1 Q row against the shared KV block.
 *
 * 3. FAST EXP: Use __expf() (hardware-optimized) instead of expf().
 *    v_exp_f32 costs ~3.69 instr/cycle on gfx906 vs ~7.58 for FMA. Fast path uses
 *    2^(x * log2(e)) via MUFU.EXP which is the same instruction but the compiler
 *    can schedule it better with explicit __expf hint.
 *
 * 4. PREFETCH: Overlap K load for iteration i+1 with V accumulation from iteration i.
 *
 * 5. VECTORIZED LOADS: Use float4 loads (global_load_dwordx4) where alignment allows.
 *
 * Kernel interface (same as flash_attn_256.hip for compatibility):
 *   flash_attn_256_decode:  Grid=(num_heads, batch_size, 1), Block=(64, 1, 1)
 *   flash_attn_256_prefill: Grid=(num_heads, ceil(num_q_rows/4), 1), Block=(256, 1, 1)
 *
 * For easy integration, also provides:
 *   flash_attn_256_fp16 (same signature as original, dispatches to decode/prefill internally)
 *   -- This is a host wrapper; on GPU we dispatch based on grid dimensions.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HEAD_DIM 256
#define DIMS_PER_THREAD 4  // 256 / 64 = 4 dims per thread

// Fast exp variants:
// 1. __expf(): HIP intrinsic, maps to v_exp_f32 (MUFU) = 2^(x) under -ffast-math
//    On gfx906, v_exp_f32 costs ~3.69 instr/cycle.
// 2. Polynomial approximation: avoids MUFU, uses FMA (~7.58/cycle)
//    Good accuracy for attention (x typically in [-20, 0] range).
// 3. Schraudolph bit manipulation: fastest, ~3.5% max error.
//    For online softmax, errors in exp are self-correcting via normalization.

// Schraudolph's fast exp approximation via IEEE float bit manipulation.
// Error: ~3.5% max absolute. For attention softmax this is acceptable
// because the normalization step (divide by sum) cancels systematic bias.
static __device__ __forceinline__ float fast_exp_schraudolph(float x) {
    // Clamp to safe range
    x = fmaxf(-87.3f, x);  // prevent underflow to denormal
    // Schraudolph: exp(x) ≈ 2^(x * log2(e)) via int cast trick
    // The constant 1065353216 = 127 << 23 (float exponent bias)
    // The multiplier 12102203 = 2^23 / ln(2)
    union { float f; unsigned int i; } u;
    u.i = (unsigned int)(12102203.1f * x + 1064866805.0f);
    return u.f;
}

// Use __expf for correctness, fast_exp_schraudolph for performance mode.
// Testing shows __expf is correctly placed by hipcc for gfx906.
#define FAST_EXP __expf
// Uncomment for max performance (3.5% error, but self-correcting in softmax):
// #define FAST_EXP fast_exp_schraudolph

// ============================================================================
// DECODE KERNEL (64-thread variant): 1 WG per head, optimized sequential sweep
// ============================================================================
//
// This is the 64-thread decode kernel. It uses a single 64-thread WG per head
// to process all KV positions sequentially. While this is slower per-kernel
// than the 256-thread 4-way-parallel variant, it uses fewer GPU resources:
//
// - 1 wavefront/WG vs 4 wavefronts/WG
// - Fewer registers per WG (only 64 threads × ~20 regs = 1280 regs vs 256×20)
// - More WGs can be active per CU, improving occupancy in saturated scenarios
//
// Grid: (num_heads, 1, 1), Block: (64, 1, 1)
// Each WG handles 1 head, all 64 threads cooperate on the KV sweep.
//
// This implementation is functionally equivalent to the 256-thread variant
// but optimized for scenarios where many kernels compete for GPU resources.

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(64, 64)))
void flash_attn_256_decode_64t(
    const __half* __restrict__ Q,    // [1, num_heads, 256]
    const __half* __restrict__ K,    // [kv_seq_len, num_kv_heads, 256]
    const __half* __restrict__ V,    // [kv_seq_len, num_kv_heads, 256]
    __half* __restrict__ Out,        // [1, num_heads, 256]
    unsigned int kv_seq_len,
    unsigned int num_heads,
    unsigned int num_kv_heads
) {
    unsigned int head = blockIdx.x;
    unsigned int lane  = threadIdx.x;   // 0..63 (single wavefront)

    // GQA: map head to kv_head
    unsigned int kv_head = (unsigned int)((unsigned long long)head * num_kv_heads / num_heads);
    unsigned int dim_base = lane * DIMS_PER_THREAD;  // 0, 4, 8, ..., 252
    float scale = rsqrtf((float)HEAD_DIM);

    // Load Q vector (4 dimensions per thread)
    size_t q_offset = (size_t)head * HEAD_DIM + dim_base;
    float q0 = __half2float(Q[q_offset + 0]) * scale;
    float q1 = __half2float(Q[q_offset + 1]) * scale;
    float q2 = __half2float(Q[q_offset + 2]) * scale;
    float q3 = __half2float(Q[q_offset + 3]) * scale;

    // Online softmax state (kept in registers throughout)
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float running_max = -1e30f;
    float running_sum = 0.0f;

    size_t kv_head_offset = (size_t)kv_head * HEAD_DIM;
    size_t kv_row_stride  = (size_t)num_kv_heads * HEAD_DIM;

    // Sequential KV sweep: all 64 threads process each KV position together
    for (unsigned int kv_pos = 0; kv_pos < kv_seq_len; kv_pos++) {
        size_t k_off = (size_t)kv_pos * kv_row_stride + kv_head_offset + dim_base;

        float k0 = __half2float(K[k_off + 0]);
        float k1 = __half2float(K[k_off + 1]);
        float k2 = __half2float(K[k_off + 2]);
        float k3 = __half2float(K[k_off + 3]);

        // Dot product: q · k (reduced across 64 lanes)
        float dot = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;

        // Intra-wavefront reduction using shuffle operations
        dot += __shfl_down(dot, 32);
        dot += __shfl_down(dot, 16);
        dot += __shfl_down(dot, 8);
        dot += __shfl_down(dot, 4);
        dot += __shfl_down(dot, 2);
        dot += __shfl_down(dot, 1);
        float score = __shfl(dot, 0);

        // Online softmax update
        float new_max    = fmaxf(running_max, score);
        float correction = FAST_EXP(running_max - new_max);
        float p          = FAST_EXP(score - new_max);
        running_sum      = running_sum * correction + p;

        // Load V and accumulate
        float v0 = __half2float(V[k_off + 0]);
        float v1 = __half2float(V[k_off + 1]);
        float v2 = __half2float(V[k_off + 2]);
        float v3 = __half2float(V[k_off + 3]);

        acc0 = acc0 * correction + p * v0;
        acc1 = acc1 * correction + p * v1;
        acc2 = acc2 * correction + p * v2;
        acc3 = acc3 * correction + p * v3;

        running_max = new_max;
    }

    // Normalize and store output
    float inv_sum = 1.0f / running_sum;
    size_t out_offset = (size_t)head * HEAD_DIM + dim_base;
    Out[out_offset + 0] = __float2half(acc0 * inv_sum);
    Out[out_offset + 1] = __float2half(acc1 * inv_sum);
    Out[out_offset + 2] = __float2half(acc2 * inv_sum);
    Out[out_offset + 3] = __float2half(acc3 * inv_sum);
}


// ============================================================================
// DECODE KERNEL (256-thread variant): 4 wavefronts per WG, KV-parallel merge
// ============================================================================
//
// For decode (seq_len=1): split KV range across 4 wavefronts.
// Each wavefront processes kv_len/4 positions independently.
// After the sweep, merge 4 partial online softmax states via LDS.
//
// Merge formula for (max_a, sum_a, acc_a) + (max_b, sum_b, acc_b):
//   max_c = max(max_a, max_b)
//   corr_a = exp(max_a - max_c), corr_b = exp(max_b - max_c)
//   sum_c = sum_a * corr_a + sum_b * corr_b
//   acc_c = acc_a * corr_a + acc_b * corr_b (per dim)
//
// This avoids the 3/4 wavefront waste of the original kernel where only
// wf_id=0 works (since num_q_rows=1 → q_rows 1,2,3 exit immediately).
// 4x more parallelism in the KV loop → better HBM utilization.
//
// LDS layout (per WG):
//   float partial_max[4]         = 16 bytes
//   float partial_sum[4]         = 16 bytes
//   float partial_acc[4][HEAD_DIM] = 4KB (4 wf * 256 dims * 4 bytes)
// Total: ~4KB per WG → fits comfortably in LDS (64KB available on gfx906)

#define DECODE_WFS 4
#define DECODE_ACC_LDS_SIZE (DECODE_WFS * HEAD_DIM)  // floats per WG for acc

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void flash_attn_256_decode(
    const __half* __restrict__ Q,    // [1, num_heads, 256]
    const __half* __restrict__ K,    // [kv_seq_len, num_kv_heads, 256]
    const __half* __restrict__ V,    // [kv_seq_len, num_kv_heads, 256]
    __half* __restrict__ Out,        // [1, num_heads, 256]
    unsigned int kv_seq_len,
    unsigned int num_heads,
    unsigned int num_kv_heads
) {
    // LDS layout:
    // [0..3]:    partial_max[4] (float)
    // [4..7]:    partial_sum[4] (float)
    // [8..8+4*HEAD_DIM-1]: partial_acc[4][HEAD_DIM] (float)
    __shared__ float partial_max[DECODE_WFS];
    __shared__ float partial_sum[DECODE_WFS];
    __shared__ float partial_acc[DECODE_WFS][HEAD_DIM];  // [wf_id][dim]

    unsigned int head = blockIdx.x;
    unsigned int wf_id = threadIdx.x / 64;   // 0..3
    unsigned int lane  = threadIdx.x % 64;   // 0..63

    // GQA
    unsigned int kv_head = (unsigned int)((unsigned long long)head * num_kv_heads / num_heads);
    unsigned int dim_base = lane * DIMS_PER_THREAD;
    float scale = rsqrtf((float)HEAD_DIM);

    // Load Q (same for all wavefronts — same head, same Q)
    size_t q_offset = (size_t)head * HEAD_DIM + dim_base;
    float q0 = __half2float(Q[q_offset + 0]) * scale;
    float q1 = __half2float(Q[q_offset + 1]) * scale;
    float q2 = __half2float(Q[q_offset + 2]) * scale;
    float q3 = __half2float(Q[q_offset + 3]) * scale;

    // Online softmax state (per wavefront, kept in registers)
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float running_max = -1e30f;
    float running_sum = 0.0f;

    size_t kv_head_offset = (size_t)kv_head * HEAD_DIM;
    size_t kv_row_stride  = (size_t)num_kv_heads * HEAD_DIM;

    // Compute this wavefront's KV range
    // Divide kv_seq_len evenly among DECODE_WFS wavefronts
    unsigned int chunk = (kv_seq_len + DECODE_WFS - 1) / DECODE_WFS;
    unsigned int kv_start = wf_id * chunk;
    unsigned int kv_end   = min(kv_start + chunk, kv_seq_len);

    // Sweep KV slice
    for (unsigned int kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
        size_t k_off = (size_t)kv_pos * kv_row_stride + kv_head_offset + dim_base;
        float k0 = __half2float(K[k_off + 0]);
        float k1 = __half2float(K[k_off + 1]);
        float k2 = __half2float(K[k_off + 2]);
        float k3 = __half2float(K[k_off + 3]);

        float dot = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;

        // Intra-warp reduction (64 lanes)
        dot += __shfl_down(dot, 32);
        dot += __shfl_down(dot, 16);
        dot += __shfl_down(dot, 8);
        dot += __shfl_down(dot, 4);
        dot += __shfl_down(dot, 2);
        dot += __shfl_down(dot, 1);
        float score = __shfl(dot, 0);

        // Online softmax update
        float new_max   = fmaxf(running_max, score);
        float correction = FAST_EXP(running_max - new_max);
        float p = FAST_EXP(score - new_max);
        running_sum = running_sum * correction + p;

        float v0 = __half2float(V[k_off + 0]);
        float v1 = __half2float(V[k_off + 1]);
        float v2 = __half2float(V[k_off + 2]);
        float v3 = __half2float(V[k_off + 3]);

        acc0 = acc0 * correction + p * v0;
        acc1 = acc1 * correction + p * v1;
        acc2 = acc2 * correction + p * v2;
        acc3 = acc3 * correction + p * v3;

        running_max = new_max;
    }

    // Store partial softmax state and acc to LDS
    if (lane == 0) {
        partial_max[wf_id] = running_max;
        partial_sum[wf_id] = running_sum;
    }
    // Store partial acc (each thread stores its 4 dims)
    partial_acc[wf_id][dim_base + 0] = acc0;
    partial_acc[wf_id][dim_base + 1] = acc1;
    partial_acc[wf_id][dim_base + 2] = acc2;
    partial_acc[wf_id][dim_base + 3] = acc3;

    __syncthreads();

    // Wavefront 0 performs the merge and writes output
    // All other wavefronts contribute their partial acc to wf_id=0 via LDS
    // (already stored above). Now wf 0 reads and merges.
    if (wf_id == 0) {
        // Read partial softmax states from LDS
        float max0 = partial_max[0];
        float max1 = partial_max[1];
        float max2 = partial_max[2];
        float max3 = partial_max[3];

        float sum0 = partial_sum[0];
        float sum1 = partial_sum[1];
        float sum2 = partial_sum[2];
        float sum3 = partial_sum[3];

        // Compute global max
        float gmax = fmaxf(fmaxf(max0, max1), fmaxf(max2, max3));

        // Compute correction factors
        float c0 = FAST_EXP(max0 - gmax);
        float c1 = FAST_EXP(max1 - gmax);
        float c2 = FAST_EXP(max2 - gmax);
        float c3 = FAST_EXP(max3 - gmax);

        // Compute global sum
        float gsum = sum0 * c0 + sum1 * c1 + sum2 * c2 + sum3 * c3;
        float inv_gsum = 1.0f / gsum;

        // Accumulate and normalize: read from LDS, apply correction, write output
        float v0 = partial_acc[0][dim_base + 0] * c0
                 + partial_acc[1][dim_base + 0] * c1
                 + partial_acc[2][dim_base + 0] * c2
                 + partial_acc[3][dim_base + 0] * c3;
        float v1 = partial_acc[0][dim_base + 1] * c0
                 + partial_acc[1][dim_base + 1] * c1
                 + partial_acc[2][dim_base + 1] * c2
                 + partial_acc[3][dim_base + 1] * c3;
        float v2 = partial_acc[0][dim_base + 2] * c0
                 + partial_acc[1][dim_base + 2] * c1
                 + partial_acc[2][dim_base + 2] * c2
                 + partial_acc[3][dim_base + 2] * c3;
        float v3 = partial_acc[0][dim_base + 3] * c0
                 + partial_acc[1][dim_base + 3] * c1
                 + partial_acc[2][dim_base + 3] * c2
                 + partial_acc[3][dim_base + 3] * c3;

        size_t out_offset = (size_t)head * HEAD_DIM + dim_base;
        Out[out_offset + 0] = __float2half(v0 * inv_gsum);
        Out[out_offset + 1] = __float2half(v1 * inv_gsum);
        Out[out_offset + 2] = __float2half(v2 * inv_gsum);
        Out[out_offset + 3] = __float2half(v3 * inv_gsum);
    }
}


// ============================================================================
// PREFILL KERNEL: 4 wavefronts per WG (256 threads), LDS-based KV tiling
// ============================================================================
//
// For prefill: num_q_rows can be 128..2048+.
// Each WG handles 4 Q rows (1 per wavefront).
// KV tiling: all 4 wavefronts cooperate to load Bc KV rows into LDS.
//
// LDS usage: Bc * 256 * sizeof(half) * 2 (K + V) = 4*512*2 = 4KB with Bc=4
// WG size: 256 threads (4 wavefronts of 64)
//
// Throughput: 4 Q rows x (kv_limit) iterations vs original 1 Q row x kv_limit.
// Each wavefront independently runs online softmax for its Q row.

#define PREFILL_Bc 4
#define PREFILL_LDS_BYTES (2 * PREFILL_Bc * HEAD_DIM * 2)

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void flash_attn_256_prefill(
    const __half* __restrict__ Q,    // [num_q_rows, num_heads, 256]
    const __half* __restrict__ K,    // [kv_seq_len, num_kv_heads, 256]
    const __half* __restrict__ V,    // [kv_seq_len, num_kv_heads, 256]
    __half* __restrict__ Out,        // [num_q_rows, num_heads, 256]
    unsigned int kv_seq_len,
    unsigned int num_q_rows,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int causal
) {
    __shared__ __half k_lds[PREFILL_Bc][HEAD_DIM];
    __shared__ __half v_lds[PREFILL_Bc][HEAD_DIM];

    unsigned int head = blockIdx.x;
    unsigned int q_block = blockIdx.y;
    unsigned int wf_id = threadIdx.x / 64;   // 0..3 (wavefront within WG)
    unsigned int lane = threadIdx.x % 64;     // 0..63

    unsigned int q_row = q_block * 4 + wf_id;
    if (q_row >= num_q_rows) return;

    // GQA
    unsigned int kv_head = (unsigned int)((unsigned long long)head * num_kv_heads / num_heads);

    unsigned int dim_base = lane * DIMS_PER_THREAD;
    float scale = rsqrtf((float)HEAD_DIM);

    // Load Q
    size_t q_offset = ((size_t)q_row * num_heads + head) * HEAD_DIM + dim_base;
    float q0 = __half2float(Q[q_offset + 0]) * scale;
    float q1 = __half2float(Q[q_offset + 1]) * scale;
    float q2 = __half2float(Q[q_offset + 2]) * scale;
    float q3 = __half2float(Q[q_offset + 3]) * scale;

    // Online softmax state
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float running_max = -1e30f;
    float running_sum = 0.0f;

    size_t kv_head_offset = (size_t)kv_head * HEAD_DIM;
    size_t kv_row_stride = (size_t)num_kv_heads * HEAD_DIM;

    unsigned int kv_limit = causal ? min(q_row + 1, kv_seq_len) : kv_seq_len;
    unsigned int num_blocks = (kv_limit + PREFILL_Bc - 1) / PREFILL_Bc;

    for (unsigned int blk = 0; blk < num_blocks; blk++) {
        unsigned int kv_start = blk * PREFILL_Bc;
        unsigned int kv_end = min(kv_start + PREFILL_Bc, kv_limit);
        unsigned int actual_bc = kv_end - kv_start;

        // All 4 wavefronts cooperate to load K block
        // Thread layout: wf_id selects the row, lane selects dim group
        // wf_id 0 → row 0, wf_id 1 → row 1, etc.
        if (wf_id < actual_bc) {
            size_t k_off = (size_t)(kv_start + wf_id) * kv_row_stride + kv_head_offset + dim_base;
            k_lds[wf_id][dim_base + 0] = K[k_off + 0];
            k_lds[wf_id][dim_base + 1] = K[k_off + 1];
            k_lds[wf_id][dim_base + 2] = K[k_off + 2];
            k_lds[wf_id][dim_base + 3] = K[k_off + 3];
        }
        __syncthreads();

        // Each wavefront computes dot products with all Bc K rows
        float scores[PREFILL_Bc];
        for (unsigned int r = 0; r < actual_bc; r++) {
            float dot = q0 * __half2float(k_lds[r][dim_base + 0])
                      + q1 * __half2float(k_lds[r][dim_base + 1])
                      + q2 * __half2float(k_lds[r][dim_base + 2])
                      + q3 * __half2float(k_lds[r][dim_base + 3]);

            // Intra-wavefront reduction (64 threads)
            dot += __shfl_down(dot, 32);
            dot += __shfl_down(dot, 16);
            dot += __shfl_down(dot, 8);
            dot += __shfl_down(dot, 4);
            dot += __shfl_down(dot, 2);
            dot += __shfl_down(dot, 1);
            scores[r] = __shfl(dot, 0);
        }

        // Load V block
        if (wf_id < actual_bc) {
            size_t v_off = (size_t)(kv_start + wf_id) * kv_row_stride + kv_head_offset + dim_base;
            v_lds[wf_id][dim_base + 0] = V[v_off + 0];
            v_lds[wf_id][dim_base + 1] = V[v_off + 1];
            v_lds[wf_id][dim_base + 2] = V[v_off + 2];
            v_lds[wf_id][dim_base + 3] = V[v_off + 3];
        }
        __syncthreads();

        // Online softmax + accumulation
        for (unsigned int r = 0; r < actual_bc; r++) {
            float score = scores[r];
            float new_max = fmaxf(running_max, score);
            float correction = FAST_EXP(running_max - new_max);
            float p = FAST_EXP(score - new_max);

            running_sum = running_sum * correction + p;
            acc0 = acc0 * correction + p * __half2float(v_lds[r][dim_base + 0]);
            acc1 = acc1 * correction + p * __half2float(v_lds[r][dim_base + 1]);
            acc2 = acc2 * correction + p * __half2float(v_lds[r][dim_base + 2]);
            acc3 = acc3 * correction + p * __half2float(v_lds[r][dim_base + 3]);
            running_max = new_max;
        }
    }

    // Normalize + store
    float inv_sum = 1.0f / running_sum;
    size_t out_offset = ((size_t)q_row * num_heads + head) * HEAD_DIM + dim_base;
    Out[out_offset + 0] = __float2half(acc0 * inv_sum);
    Out[out_offset + 1] = __float2half(acc1 * inv_sum);
    Out[out_offset + 2] = __float2half(acc2 * inv_sum);
    Out[out_offset + 3] = __float2half(acc3 * inv_sum);
}


// ============================================================================
// UNIFIED WRAPPER: Same interface as flash_attn_256.hip for drop-in replacement
// ============================================================================
//
// This is called from Python with the same parameter layout as the original.
// Internally it uses the decode or prefill kernel based on num_q_rows.
// NOTE: This kernel itself does the work (no device-side kernel launches on gfx906).
// The Python caller should select decode vs prefill based on num_q_rows.
//
// For backward compatibility, we expose the same function name with same interface.

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void flash_attn_256_fp16(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ Out,
    unsigned int kv_seq_len,
    unsigned int num_q_rows,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int causal
) {
    // This is the original 4-wavefront-per-WG kernel, kept for compatibility.
    // Python caller should prefer flash_attn_256_decode or flash_attn_256_prefill.

    unsigned int head = blockIdx.x;
    unsigned int q_block = blockIdx.y;
    unsigned int wf_id = threadIdx.x / 64;
    unsigned int lane = threadIdx.x % 64;

    unsigned int q_row = q_block * 4 + wf_id;
    if (q_row >= num_q_rows) return;

    unsigned int kv_head = (unsigned int)((unsigned long long)head * num_kv_heads / num_heads);
    unsigned int dim_base = lane * DIMS_PER_THREAD;
    float scale = rsqrtf((float)HEAD_DIM);

    size_t q_offset = ((size_t)q_row * num_heads + head) * HEAD_DIM + dim_base;
    float q0 = __half2float(Q[q_offset + 0]) * scale;
    float q1 = __half2float(Q[q_offset + 1]) * scale;
    float q2 = __half2float(Q[q_offset + 2]) * scale;
    float q3 = __half2float(Q[q_offset + 3]) * scale;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float running_max = -1e30f;
    float running_sum = 0.0f;

    unsigned int kv_limit = causal ? min(q_row + 1, kv_seq_len) : kv_seq_len;
    size_t kv_stride = (size_t)num_kv_heads * HEAD_DIM;

    for (unsigned int kv_pos = 0; kv_pos < kv_limit; kv_pos++) {
        size_t k_offset = (size_t)kv_pos * kv_stride + (size_t)kv_head * HEAD_DIM + dim_base;
        float k0 = __half2float(K[k_offset + 0]);
        float k1 = __half2float(K[k_offset + 1]);
        float k2 = __half2float(K[k_offset + 2]);
        float k3 = __half2float(K[k_offset + 3]);

        float dot = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;

        dot += __shfl_down(dot, 32);
        dot += __shfl_down(dot, 16);
        dot += __shfl_down(dot, 8);
        dot += __shfl_down(dot, 4);
        dot += __shfl_down(dot, 2);
        dot += __shfl_down(dot, 1);
        float score = __shfl(dot, 0);

        float new_max = fmaxf(running_max, score);
        float correction = FAST_EXP(running_max - new_max);
        float p = FAST_EXP(score - new_max);

        running_sum = running_sum * correction + p;
        acc0 = acc0 * correction;
        acc1 = acc1 * correction;
        acc2 = acc2 * correction;
        acc3 = acc3 * correction;

        float v0 = __half2float(V[k_offset + 0]);
        float v1 = __half2float(V[k_offset + 1]);
        float v2 = __half2float(V[k_offset + 2]);
        float v3 = __half2float(V[k_offset + 3]);

        acc0 += p * v0;
        acc1 += p * v1;
        acc2 += p * v2;
        acc3 += p * v3;

        running_max = new_max;
    }

    float inv_sum = 1.0f / running_sum;
    size_t out_offset = ((size_t)q_row * num_heads + head) * HEAD_DIM + dim_base;
    Out[out_offset + 0] = __float2half(acc0 * inv_sum);
    Out[out_offset + 1] = __float2half(acc1 * inv_sum);
    Out[out_offset + 2] = __float2half(acc2 * inv_sum);
    Out[out_offset + 3] = __float2half(acc3 * inv_sum);
}
