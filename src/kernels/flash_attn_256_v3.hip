/**
 * flash_attn_256_v3.hip - Block-tiled FlashAttention-2 for gfx906 (head_dim=256).
 *
 * Improvements over flash_attn_256_tuned.hip (prefill):
 *   - Bc: 4 → 16  (4× larger KV tile → better amortization of global load latency)
 *   - Score compute: scalar FMA → __builtin_amdgcn_fdot2 (v_dot2_f32_f16)
 *                    2× arithmetic throughput per instruction.
 *   - Q rows per WG: 4 (1/wf) → 16 (4/wf)  (4× more work per WG)
 *   - Within each wavefront: 4 Q rows processed with 16-thread partial reduction per score.
 *
 * Thread layout (prefill v3, 256 threads = 4 wavefronts × 64 lanes):
 *   wf_id    = tid / 64            (0..3)
 *   lane     = tid % 64            (0..63)
 *   q_in_wf  = lane / 16           (0..3):  Q row index within wavefront
 *   part     = lane % 16           (0..15): which 16-dim chunk of head dim
 *   dim_base = part * 16           (0, 16, 32, ..., 240)
 *   q_row    = wf_id * 4 + q_in_wf (0..15 per WG)
 *   q_row_global = blockIdx.y * BLOCK_M + q_row
 *
 * Score computation for K row r, Q row q:
 *   - 16 threads (parts 0..15) each hold 16 Q dims → partial FP32 sum via 8 fdot2 calls.
 *   - 4-step shfl_down butterfly within the 16-thread group → final score at part=0.
 *   - Broadcast to all 16 threads via shfl.
 *
 * LDS usage:
 *   k_lds: [BLOCK_N=16][HEAD_DIM=256] FP16 = 8 KB
 *   v_lds: [BLOCK_N=16][HEAD_DIM=256] FP16 = 8 KB
 *   Total: 16 KB per WG → supports 2 WGs/CU (32 KB LDS / WG).
 *
 * VGPR estimate per thread (~50 VGPRs → 4+ waves/SIMD occupancy):
 *   qreg[16]: 16 floats (Q dims for this thread's row)
 *   acc[16]:  16 floats (output accumulator)
 *   scores[16]: 16 floats (one per Bc K-row, held during softmax update)
 *   running_max, running_sum: 2 floats
 *   Misc (addresses, temporaries): ~10 floats
 *   Total: ~60 VGPRs (well within 64-80 target)
 *
 * Tile sizes (final, documented):
 *   BLOCK_M = 16  (16 Q rows per WG = 4 per wavefront × 4 wavefronts)
 *     Note: Requested Br=64 would require q[64 dims] registers per thread (too many VGPRs).
 *     Br=16 with 4 Q rows per wavefront uses 16 Q dim registers per thread = correct budget.
 *   BLOCK_N = 16  (16 KV positions per tile, 16 KB LDS for K+V)
 *
 * GQA: kv_head = head * num_kv_heads / num_heads (integer division).
 *
 * Causal masking: per-element and per-block (skip blocks entirely beyond causal boundary).
 *
 * Interface (same signature as flash_attn_256_tuned.hip prefill):
 *   flash_attn_256_decode:       Grid=(num_heads, 1, 1), Block=(256, 1, 1)
 *   flash_attn_256_v3_prefill:   Grid=(num_heads, ceil(num_q_rows/BLOCK_M), 1), Block=(256,1,1)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ============================================================================
// Constants
// ============================================================================

#define HEAD_DIM      256
#define BLOCK_M       16    // Q rows per WG (4 per wavefront × 4 wavefronts)
#define BLOCK_N       16    // KV rows per tile (Bc)
#define WF_SIZE       64    // wavefront width on gfx906
#define NUM_WF        4     // wavefronts per WG
#define QR_PER_WF     4     // Q rows per wavefront = BLOCK_M / NUM_WF
#define PARTS_PER_ROW 16    // threads per Q row for score reduction (HEAD_DIM/DIMS_PER_PART)
#define DIMS_PER_PART 16    // head dims per thread = HEAD_DIM / PARTS_PER_ROW

#define FAST_EXP __expf

// ============================================================================
// DECODE KERNEL: unchanged from flash_attn_256_tuned.hip
// 256 threads (4 wavefronts), KV range split across 4 wfs, merge via LDS.
// Grid=(num_heads, 1, 1), Block=(256, 1, 1)
// ============================================================================

#define DECODE_WFS           4
#define DECODE_DIMS_PER_LANE 4   // 256 head dims / 64 lanes per wf

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void flash_attn_256_decode(
    const __half* __restrict__ Q,     // [1, num_heads, 256]
    const __half* __restrict__ K,     // [kv_seq_len, num_kv_heads, 256]
    const __half* __restrict__ V,     // [kv_seq_len, num_kv_heads, 256]
    __half* __restrict__ Out,         // [1, num_heads, 256]
    unsigned int kv_seq_len,
    unsigned int num_heads,
    unsigned int num_kv_heads
) {
    __shared__ float partial_max[DECODE_WFS];
    __shared__ float partial_sum[DECODE_WFS];
    __shared__ float partial_acc[DECODE_WFS][HEAD_DIM];

    const unsigned int head  = blockIdx.x;
    const unsigned int wf_id = threadIdx.x / WF_SIZE;
    const unsigned int lane  = threadIdx.x % WF_SIZE;

    const unsigned int kv_head = (unsigned int)((unsigned long long)head * num_kv_heads / num_heads);
    const unsigned int dim_base = lane * DECODE_DIMS_PER_LANE;
    const float scale = rsqrtf((float)HEAD_DIM);

    // Load Q (same for all wavefronts)
    const size_t q_off = (size_t)head * HEAD_DIM + dim_base;
    float q0 = __half2float(Q[q_off + 0]) * scale;
    float q1 = __half2float(Q[q_off + 1]) * scale;
    float q2 = __half2float(Q[q_off + 2]) * scale;
    float q3 = __half2float(Q[q_off + 3]) * scale;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float running_max = -1e30f;
    float running_sum = 0.0f;

    const size_t kv_head_offset = (size_t)kv_head * HEAD_DIM;
    const size_t kv_row_stride  = (size_t)num_kv_heads * HEAD_DIM;

    // Split KV range across 4 wavefronts
    const unsigned int chunk    = (kv_seq_len + DECODE_WFS - 1) / DECODE_WFS;
    const unsigned int kv_start = wf_id * chunk;
    const unsigned int kv_end   = min(kv_start + chunk, kv_seq_len);

    for (unsigned int kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
        const size_t k_off = (size_t)kv_pos * kv_row_stride + kv_head_offset + dim_base;
        float dot = q0 * __half2float(K[k_off + 0])
                  + q1 * __half2float(K[k_off + 1])
                  + q2 * __half2float(K[k_off + 2])
                  + q3 * __half2float(K[k_off + 3]);

        // 64-lane butterfly reduction (intra-wavefront)
        dot += __shfl_down(dot, 32);
        dot += __shfl_down(dot, 16);
        dot += __shfl_down(dot, 8);
        dot += __shfl_down(dot, 4);
        dot += __shfl_down(dot, 2);
        dot += __shfl_down(dot, 1);
        const float score = __shfl(dot, 0);

        const float new_max  = fmaxf(running_max, score);
        const float corr     = FAST_EXP(running_max - new_max);
        const float p        = FAST_EXP(score - new_max);
        running_sum = running_sum * corr + p;

        acc0 = acc0 * corr + p * __half2float(V[k_off + 0]);
        acc1 = acc1 * corr + p * __half2float(V[k_off + 1]);
        acc2 = acc2 * corr + p * __half2float(V[k_off + 2]);
        acc3 = acc3 * corr + p * __half2float(V[k_off + 3]);
        running_max = new_max;
    }

    // Store partial softmax state + acc to LDS
    if (lane == 0) {
        partial_max[wf_id] = running_max;
        partial_sum[wf_id] = running_sum;
    }
    partial_acc[wf_id][dim_base + 0] = acc0;
    partial_acc[wf_id][dim_base + 1] = acc1;
    partial_acc[wf_id][dim_base + 2] = acc2;
    partial_acc[wf_id][dim_base + 3] = acc3;
    __syncthreads();

    // Wavefront 0 merges all partial results and writes output
    if (wf_id == 0) {
        const float gmax = fmaxf(fmaxf(partial_max[0], partial_max[1]),
                                  fmaxf(partial_max[2], partial_max[3]));
        const float c0 = FAST_EXP(partial_max[0] - gmax);
        const float c1 = FAST_EXP(partial_max[1] - gmax);
        const float c2 = FAST_EXP(partial_max[2] - gmax);
        const float c3 = FAST_EXP(partial_max[3] - gmax);
        const float gsum = partial_sum[0]*c0 + partial_sum[1]*c1
                         + partial_sum[2]*c2 + partial_sum[3]*c3;
        const float inv = 1.0f / gsum;

        const size_t out_off = (size_t)head * HEAD_DIM + dim_base;
        for (int d = 0; d < DECODE_DIMS_PER_LANE; d++) {
            float v = partial_acc[0][dim_base+d]*c0 + partial_acc[1][dim_base+d]*c1
                    + partial_acc[2][dim_base+d]*c2 + partial_acc[3][dim_base+d]*c3;
            Out[out_off + d] = __float2half(v * inv);
        }
    }
}


// ============================================================================
// PREFILL V3: Block-tiled [BLOCK_M=16, BLOCK_N=16] with v_dot2_f32_f16
//
// Grid: (num_heads, ceil(num_q_rows / BLOCK_M), 1), Block: (256, 1, 1)
// ============================================================================

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void flash_attn_256_v3_prefill(
    const __half* __restrict__ Q,      // [num_q_rows, num_heads, HEAD_DIM]
    const __half* __restrict__ K,      // [kv_seq_len, num_kv_heads, HEAD_DIM]
    const __half* __restrict__ V,      // [kv_seq_len, num_kv_heads, HEAD_DIM]
    __half* __restrict__ Out,          // [num_q_rows, num_heads, HEAD_DIM]
    unsigned int kv_seq_len,
    unsigned int num_q_rows,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int causal
) {
    // LDS buffers for K and V blocks
    __shared__ __half k_lds[BLOCK_N * HEAD_DIM];  // 8 KB
    __shared__ __half v_lds[BLOCK_N * HEAD_DIM];  // 8 KB

    // Thread decomposition
    const unsigned int tid      = threadIdx.x;
    const unsigned int wf_id    = tid / WF_SIZE;            // 0..3
    const unsigned int lane     = tid % WF_SIZE;            // 0..63
    const unsigned int q_in_wf  = lane / PARTS_PER_ROW;    // 0..3  (Q row within wf)
    const unsigned int part     = lane % PARTS_PER_ROW;    // 0..15 (dim-chunk index)
    const unsigned int dim_base = part * DIMS_PER_PART;    // 0,16,32,...,240

    const unsigned int head         = blockIdx.x;
    const unsigned int q_blk        = blockIdx.y;
    const unsigned int q_row        = wf_id * QR_PER_WF + q_in_wf;  // 0..15
    const unsigned int q_row_global = q_blk * BLOCK_M + q_row;

    // GQA: map attention head to KV head
    const unsigned int kv_head = (unsigned int)((unsigned long long)head * num_kv_heads / num_heads);
    const float scale = rsqrtf((float)HEAD_DIM);

    // -------------------------------------------------------------------------
    // Load Q tile for this thread's q_row into FP32 registers.
    // qreg[d] = Q[q_row_global][head][dim_base + d] * scale, for d in 0..DIMS_PER_PART-1
    // -------------------------------------------------------------------------
    float qreg[DIMS_PER_PART];
    {
        if (q_row_global < num_q_rows) {
            const size_t q_off = ((size_t)q_row_global * num_heads + head) * HEAD_DIM + dim_base;
            #pragma unroll
            for (int d = 0; d < DIMS_PER_PART; d++) {
                qreg[d] = __half2float(Q[q_off + d]) * scale;
            }
        } else {
            #pragma unroll
            for (int d = 0; d < DIMS_PER_PART; d++) qreg[d] = 0.0f;
        }
    }

    // Online softmax accumulators
    float acc[DIMS_PER_PART];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_PART; d++) acc[d] = 0.0f;
    float running_max = -1e30f;
    float running_sum = 0.0f;

    const size_t kv_row_stride  = (size_t)num_kv_heads * HEAD_DIM;
    const size_t kv_head_offset = (size_t)kv_head * HEAD_DIM;

    // KV limit for causal masking
    const unsigned int kv_limit = causal
        ? min(q_row_global + 1, kv_seq_len)
        : kv_seq_len;
    const unsigned int num_kv_blocks = (kv_limit + BLOCK_N - 1) / BLOCK_N;

    // -------------------------------------------------------------------------
    // Cooperative K/V load assignment:
    //   256 threads load BLOCK_N=16 rows × HEAD_DIM=256 halfs = 4096 halfs total.
    //   Each thread: 4096/256 = 16 halfs.
    //   load_row = tid / PARTS_PER_ROW    → which KV row (0..15)
    //   load_dim = (tid % PARTS_PER_ROW) * DIMS_PER_PART  → which 16-dim segment (0..240)
    // -------------------------------------------------------------------------
    const unsigned int load_row = tid / PARTS_PER_ROW;
    const unsigned int load_dim = (tid % PARTS_PER_ROW) * DIMS_PER_PART;

    // =========================================================================
    // Main KV block loop
    // =========================================================================
    for (unsigned int blk = 0; blk < num_kv_blocks; blk++) {
        const unsigned int kv_blk_start = blk * BLOCK_N;
        const unsigned int kv_blk_end   = min(kv_blk_start + BLOCK_N, kv_limit);
        const unsigned int actual_bc    = kv_blk_end - kv_blk_start;

        // === 1. Load K block into LDS ===
        {
            __half* dst = &k_lds[load_row * HEAD_DIM + load_dim];
            if (load_row < actual_bc) {
                const unsigned int kv_pos = kv_blk_start + load_row;
                const size_t k_off = (size_t)kv_pos * kv_row_stride + kv_head_offset + load_dim;
                // Load 16 halfs = 2 × float4 (vectorized global load)
                *reinterpret_cast<float4*>(dst)     = *reinterpret_cast<const float4*>(K + k_off);
                *reinterpret_cast<float4*>(dst + 8) = *reinterpret_cast<const float4*>(K + k_off + 8);
            } else {
                // Zero-pad boundary (so padding doesn't corrupt scores)
                *reinterpret_cast<float4*>(dst)     = make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<float4*>(dst + 8) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        __syncthreads();

        // === 2. Compute QK^T scores for all BLOCK_N K rows ===
        //
        // For K row r:
        //   partial = sum_{d=0}^{DIMS_PER_PART-1} qreg[d] * K_lds[r][dim_base+d]
        //           = 8 calls to fdot2 (2 dims per call)
        //   Then reduce across PARTS_PER_ROW=16 threads within the same q_in_wf group.
        //   Reduction pattern (16-way within wavefront of 64):
        //     step 1: x += shfl_down(x, 8)  → groups of 8
        //     step 2: x += shfl_down(x, 4)  → groups of 4
        //     step 3: x += shfl_down(x, 2)  → groups of 2
        //     step 4: x += shfl_down(x, 1)  → final in part=0 of each group
        //   Broadcast: score = shfl(partial, q_in_wf * PARTS_PER_ROW) for all 16 in group.
        //
        // This uses gfx906 __shfl_down/__shfl which are intra-wavefront (64 lanes). ✓
        // The 4 groups (q_in_wf=0..3) occupy lanes 0..15, 16..31, 32..47, 48..63.
        // Within each group, shfl_down(x,8/4/2/1) correctly reduces all 16 to lane 0.

        float scores[BLOCK_N];
        #pragma unroll
        for (int r = 0; r < BLOCK_N; r++) {
            const __half* k_row = &k_lds[r * HEAD_DIM + dim_base];

            // Partial dot product: 8 fdot2 calls for 16 dims
            float partial = 0.0f;
            #pragma unroll
            for (int d = 0; d < DIMS_PER_PART; d += 2) {
                const __half2 qv = __half2(__float2half(qreg[d]), __float2half(qreg[d + 1]));
                const __half2 kv = *reinterpret_cast<const __half2*>(k_row + d);
                partial = __builtin_amdgcn_fdot2(
                    *reinterpret_cast<const _Float16_2*>(&qv),
                    *reinterpret_cast<const _Float16_2*>(&kv),
                    partial, false);
            }

            // 16-way intra-wavefront reduction within each q_in_wf group
            partial += __shfl_down(partial, 8);
            partial += __shfl_down(partial, 4);
            partial += __shfl_down(partial, 2);
            partial += __shfl_down(partial, 1);
            // Broadcast final score to all 16 threads in this q_in_wf group.
            // Source lane = first lane of this group = q_in_wf * PARTS_PER_ROW.
            scores[r] = __shfl(partial, (int)(q_in_wf * PARTS_PER_ROW));

            // Apply causal mask per element
            if (causal) {
                const unsigned int kv_abs = kv_blk_start + (unsigned int)r;
                if (kv_abs > q_row_global) scores[r] = -1e30f;
            }
        }

        // === 3. Online softmax update ===
        // Process all Bc scores and update running_max, running_sum, acc.
        // Scale acc by correction factor before adding new term.
        // Note: correction is applied once per KV block, scanning from score 0..Bc-1.
        #pragma unroll
        for (int r = 0; r < BLOCK_N; r++) {
            const float s = scores[r];
            const float new_max = fmaxf(running_max, s);
            const float corr    = FAST_EXP(running_max - new_max);
            const float p       = FAST_EXP(s - new_max);

            // Rescale accumulator
            #pragma unroll
            for (int d = 0; d < DIMS_PER_PART; d++) acc[d] *= corr;
            running_sum = running_sum * corr + p;
            running_max = new_max;

            // Store softmax weight back (will use in PV step)
            scores[r] = p;
        }

        // === 4. Load V block into LDS ===
        {
            __half* dst = &v_lds[load_row * HEAD_DIM + load_dim];
            if (load_row < actual_bc) {
                const unsigned int kv_pos = kv_blk_start + load_row;
                const size_t v_off = (size_t)kv_pos * kv_row_stride + kv_head_offset + load_dim;
                *reinterpret_cast<float4*>(dst)     = *reinterpret_cast<const float4*>(V + v_off);
                *reinterpret_cast<float4*>(dst + 8) = *reinterpret_cast<const float4*>(V + v_off + 8);
            } else {
                *reinterpret_cast<float4*>(dst)     = make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<float4*>(dst + 8) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        __syncthreads();

        // === 5. Accumulate PV ===
        // acc[d] += scores[r] * V_lds[r][dim_base + d], for r=0..Bc-1, d=0..DIMS_PER_PART-1
        #pragma unroll
        for (int r = 0; r < BLOCK_N; r++) {
            const float p = scores[r];
            const __half* v_row = &v_lds[r * HEAD_DIM + dim_base];
            #pragma unroll
            for (int d = 0; d < DIMS_PER_PART; d++) {
                acc[d] += p * __half2float(v_row[d]);
            }
        }

        __syncthreads();
    }  // end KV block loop

    // =========================================================================
    // Normalize and write output
    // =========================================================================
    if (q_row_global < num_q_rows && running_sum > 0.0f) {
        const float inv_sum = 1.0f / running_sum;
        const size_t out_off = ((size_t)q_row_global * num_heads + head) * HEAD_DIM + dim_base;
        #pragma unroll
        for (int d = 0; d < DIMS_PER_PART; d++) {
            Out[out_off + d] = __float2half(acc[d] * inv_sum);
        }
    }
}
