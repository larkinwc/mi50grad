/**
 * Fused INT4 QKV GEMV kernel for attention projections.
 *
 * Fuses three separate GEMV launches (Q, K, V) into one:
 *   - gemv_q_fused:  [hidden, q_dim]  -> Q buffer (10 heads × 128 head_dim = 1280)
 *   - gemv_k_only:   [hidden, kv_dim] -> K buffer (2 KV heads × 128 = 256)
 *   - gemv_v_cache:  [hidden, kv_dim] -> V cache position (2 KV heads × 128 = 256)
 *
 * This kernel processes concatenated weight matrix [hidden, q_dim + 2*kv_dim]
 * and writes outputs to three separate buffers.
 *
 * Target: gfx906 (MI50), 256 VGPRs available.
 * Uses same INT4 dequantization and FP32 accumulation as gemv_int4_v8.
 *
 * Output mapping (N_total = q_dim + 2*kv_dim = 1280 + 256 + 256 = 1792 per GPU for TP=4):
 *   - col 0..q_dim-1 (0..1279):         writes to Q buffer
 *   - col q_dim..q_dim+kv_dim-1 (1280..1535):    writes to K buffer
 *   - col q_dim+kv_dim..N_total-1 (1536..1791): writes to V buffer (or cache position)
 *
 * Direct KV write mode:
 *   When v_cache_dst != nullptr, V outputs are written directly to KV cache position.
 *   Otherwise, V outputs are written to d_v working buffer.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__device__ __forceinline__ float dequant_dot_fp32(
    unsigned int packed,
    float scale,
    float zero,
    const __half* __restrict__ A,
    unsigned int a_base
) {
    float a0 = __half2float(A[a_base]);
    float a1 = __half2float(A[a_base + 1]);
    float a2 = __half2float(A[a_base + 2]);
    float a3 = __half2float(A[a_base + 3]);
    float a4 = __half2float(A[a_base + 4]);
    float a5 = __half2float(A[a_base + 5]);
    float a6 = __half2float(A[a_base + 6]);
    float a7 = __half2float(A[a_base + 7]);

    float w0 = ((float)__builtin_amdgcn_ubfe(packed,  0, 4) - zero) * scale;
    float w1 = ((float)__builtin_amdgcn_ubfe(packed,  4, 4) - zero) * scale;
    float w2 = ((float)__builtin_amdgcn_ubfe(packed,  8, 4) - zero) * scale;
    float w3 = ((float)__builtin_amdgcn_ubfe(packed, 12, 4) - zero) * scale;
    float w4 = ((float)__builtin_amdgcn_ubfe(packed, 16, 4) - zero) * scale;
    float w5 = ((float)__builtin_amdgcn_ubfe(packed, 20, 4) - zero) * scale;
    float w6 = ((float)__builtin_amdgcn_ubfe(packed, 24, 4) - zero) * scale;
    float w7 = ((float)__builtin_amdgcn_ubfe(packed, 28, 4) - zero) * scale;

    return w0*a0 + w1*a1 + w2*a2 + w3*a3 + w4*a4 + w5*a5 + w6*a6 + w7*a7;
}

template <int THREADS_PER_COL>
__device__ __forceinline__ void gemv_qkv_fused_cooperative(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ Q_out,
    __half* __restrict__ K_out,
    __half* __restrict__ V_out,
    unsigned int K,
    unsigned int q_dim,
    unsigned int kv_dim,
    unsigned int group_size
) {
    constexpr unsigned int COLS_PER_WG = 256 / THREADS_PER_COL;
    constexpr unsigned int NUM_WF = 4;
    constexpr unsigned int TPC_PER_WF = THREADS_PER_COL / NUM_WF;

    unsigned int col_in_wg = threadIdx.x % COLS_PER_WG;
    unsigned int k_split_id = threadIdx.x / COLS_PER_WG;
    unsigned int col = blockIdx.x * COLS_PER_WG + col_in_wg;
    
    unsigned int N_total = q_dim + 2 * kv_dim;
    if (col >= N_total) return;

    unsigned int num_k_groups = K >> 3;
    unsigned int groups_per_scale = group_size >> 3;
    unsigned int gps_log2 = 31u - __builtin_clz(groups_per_scale);

    unsigned int kg_start = (num_k_groups * k_split_id) / THREADS_PER_COL;
    unsigned int kg_end   = (num_k_groups * (k_split_id + 1)) / THREADS_PER_COL;

    if (kg_start >= kg_end) return;

    // Two accumulators to break dependency chains
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    float cur_scale = 0.0f, cur_zero = 0.0f;
    unsigned int current_group = 0xFFFFFFFF;

    // 4x register blocking: process 4 weight words per iteration
    unsigned int kg = kg_start;
    unsigned int kg_end_4 = kg_start + ((kg_end - kg_start) & ~3u);

    for (; kg < kg_end_4; kg += 4) {
        // Preload 4 weight words (memory coalesced across threads for same kg)
        unsigned int packed0 = B_q4[kg * N_total + col];
        unsigned int packed1 = B_q4[(kg + 1) * N_total + col];
        unsigned int packed2 = B_q4[(kg + 2) * N_total + col];
        unsigned int packed3 = B_q4[(kg + 3) * N_total + col];

        // Process word 0
        unsigned int g0 = kg >> gps_log2;
        if (g0 != current_group) {
            cur_scale = __half2float(scales[g0 * N_total + col]);
            cur_zero  = __half2float(zeros[g0 * N_total + col]);
            current_group = g0;
        }
        acc0 += dequant_dot_fp32(packed0, cur_scale, cur_zero, A, kg << 3);

        // Process word 1
        unsigned int g1 = (kg + 1) >> gps_log2;
        if (g1 != current_group) {
            cur_scale = __half2float(scales[g1 * N_total + col]);
            cur_zero  = __half2float(zeros[g1 * N_total + col]);
            current_group = g1;
        }
        acc1 += dequant_dot_fp32(packed1, cur_scale, cur_zero, A, (kg + 1) << 3);

        // Process word 2
        unsigned int g2 = (kg + 2) >> gps_log2;
        if (g2 != current_group) {
            cur_scale = __half2float(scales[g2 * N_total + col]);
            cur_zero  = __half2float(zeros[g2 * N_total + col]);
            current_group = g2;
        }
        acc0 += dequant_dot_fp32(packed2, cur_scale, cur_zero, A, (kg + 2) << 3);

        // Process word 3
        unsigned int g3 = (kg + 3) >> gps_log2;
        if (g3 != current_group) {
            cur_scale = __half2float(scales[g3 * N_total + col]);
            cur_zero  = __half2float(zeros[g3 * N_total + col]);
            current_group = g3;
        }
        acc1 += dequant_dot_fp32(packed3, cur_scale, cur_zero, A, (kg + 3) << 3);
    }

    // Merge accumulators
    float acc = acc0 + acc1;

    // Handle remaining 1-3 iterations
    for (; kg < kg_end; kg++) {
        unsigned int packed = B_q4[kg * N_total + col];
        unsigned int g = kg >> gps_log2;
        if (g != current_group) {
            cur_scale = __half2float(scales[g * N_total + col]);
            cur_zero  = __half2float(zeros[g * N_total + col]);
            current_group = g;
        }
        acc += dequant_dot_fp32(packed, cur_scale, cur_zero, A, kg << 3);
    }

    // Intra-wavefront DPP reduction
    if (TPC_PER_WF >= 2) acc += __shfl_down(acc, COLS_PER_WG);
    if (TPC_PER_WF >= 4) acc += __shfl_down(acc, COLS_PER_WG * 2);

    // LDS cross-wavefront reduction
    __shared__ float s_reduce[NUM_WF * COLS_PER_WG];

    if ((k_split_id & (TPC_PER_WF - 1)) == 0) {
        unsigned int wf_id = k_split_id / TPC_PER_WF;
        s_reduce[wf_id * COLS_PER_WG + col_in_wg] = acc;
    }
    __syncthreads();

    if (k_split_id == 0) {
        float sum = s_reduce[0 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[1 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[2 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[3 * COLS_PER_WG + col_in_wg];
        
        // Route output to correct buffer based on column index
        if (col < q_dim) {
            // Q output (0..q_dim-1)
            Q_out[col] = __float2half(sum);
        } else if (col < q_dim + kv_dim) {
            // K output (q_dim..q_dim+kv_dim-1)
            K_out[col - q_dim] = __float2half(sum);
        } else {
            // V output (q_dim+kv_dim..N_total-1)
            V_out[col - q_dim - kv_dim] = __float2half(sum);
        }
    }
}

/**
 * Fused QKV GEMV kernel with direct KV cache write support.
 *
 * @param A              Input activation [K] FP16 (normalized hidden state)
 * @param B_q4           Concatenated QKV weights [K/8, N_total] INT4 packed
 * @param scales         Per-group scales [K/group_size, N_total] FP16
 * @param zeros          Per-group zeros [K/group_size, N_total] FP16
 * @param Q_out          Q output buffer [q_dim] FP16
 * @param K_out          K output buffer [kv_dim] FP16
 * @param V_out          V output buffer [kv_dim] FP16 OR KV cache position
 * @param K              Input dimension (hidden_size = 5120)
 * @param q_dim          Q output dimension (1280 for TP=4)
 * @param kv_dim         K/V output dimension (256 for TP=4)
 * @param group_size     Quantization group size (128)
 * @param v_cache_dst    If non-null, V output is written to this KV cache position
 */
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_qkv_fused_t16(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ Q_out,
    __half* __restrict__ K_out,
    __half* __restrict__ V_out,
    unsigned int K,
    unsigned int q_dim,
    unsigned int kv_dim,
    unsigned int group_size,
    __half* __restrict__ v_cache_dst  // optional: if non-null, V writes here
) {
    const unsigned int N_total = q_dim + 2 * kv_dim;
    constexpr unsigned int THREADS_PER_COL = 16;
    constexpr unsigned int COLS_PER_WG = 256 / THREADS_PER_COL;
    constexpr unsigned int NUM_WF = 4;
    constexpr unsigned int TPC_PER_WF = THREADS_PER_COL / NUM_WF;

    unsigned int col_in_wg = threadIdx.x % COLS_PER_WG;
    unsigned int k_split_id = threadIdx.x / COLS_PER_WG;
    unsigned int col = blockIdx.x * COLS_PER_WG + col_in_wg;
    
    if (col >= N_total) return;

    unsigned int num_k_groups = K >> 3;
    unsigned int groups_per_scale = group_size >> 3;
    unsigned int gps_log2 = 31u - __builtin_clz(groups_per_scale);

    unsigned int kg_start = (num_k_groups * k_split_id) / THREADS_PER_COL;
    unsigned int kg_end   = (num_k_groups * (k_split_id + 1)) / THREADS_PER_COL;

    if (kg_start >= kg_end) return;

    // Two accumulators to break dependency chains
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    float cur_scale = 0.0f, cur_zero = 0.0f;
    unsigned int current_group = 0xFFFFFFFF;

    // 4x register blocking: process 4 weight words per iteration
    unsigned int kg = kg_start;
    unsigned int kg_end_4 = kg_start + ((kg_end - kg_start) & ~3u);

    for (; kg < kg_end_4; kg += 4) {
        // Preload 4 weight words (memory coalesced across threads for same kg)
        unsigned int packed0 = B_q4[kg * N_total + col];
        unsigned int packed1 = B_q4[(kg + 1) * N_total + col];
        unsigned int packed2 = B_q4[(kg + 2) * N_total + col];
        unsigned int packed3 = B_q4[(kg + 3) * N_total + col];

        // Process word 0
        unsigned int g0 = kg >> gps_log2;
        if (g0 != current_group) {
            cur_scale = __half2float(scales[g0 * N_total + col]);
            cur_zero  = __half2float(zeros[g0 * N_total + col]);
            current_group = g0;
        }
        acc0 += dequant_dot_fp32(packed0, cur_scale, cur_zero, A, kg << 3);

        // Process word 1
        unsigned int g1 = (kg + 1) >> gps_log2;
        if (g1 != current_group) {
            cur_scale = __half2float(scales[g1 * N_total + col]);
            cur_zero  = __half2float(zeros[g1 * N_total + col]);
            current_group = g1;
        }
        acc1 += dequant_dot_fp32(packed1, cur_scale, cur_zero, A, (kg + 1) << 3);

        // Process word 2
        unsigned int g2 = (kg + 2) >> gps_log2;
        if (g2 != current_group) {
            cur_scale = __half2float(scales[g2 * N_total + col]);
            cur_zero  = __half2float(zeros[g2 * N_total + col]);
            current_group = g2;
        }
        acc0 += dequant_dot_fp32(packed2, cur_scale, cur_zero, A, (kg + 2) << 3);

        // Process word 3
        unsigned int g3 = (kg + 3) >> gps_log2;
        if (g3 != current_group) {
            cur_scale = __half2float(scales[g3 * N_total + col]);
            cur_zero  = __half2float(zeros[g3 * N_total + col]);
            current_group = g3;
        }
        acc1 += dequant_dot_fp32(packed3, cur_scale, cur_zero, A, (kg + 3) << 3);
    }

    // Merge accumulators
    float acc = acc0 + acc1;

    // Handle remaining 1-3 iterations
    for (; kg < kg_end; kg++) {
        unsigned int packed = B_q4[kg * N_total + col];
        unsigned int g = kg >> gps_log2;
        if (g != current_group) {
            cur_scale = __half2float(scales[g * N_total + col]);
            cur_zero  = __half2float(zeros[g * N_total + col]);
            current_group = g;
        }
        acc += dequant_dot_fp32(packed, cur_scale, cur_zero, A, kg << 3);
    }

    // Intra-wavefront DPP reduction
    if (TPC_PER_WF >= 2) acc += __shfl_down(acc, COLS_PER_WG);
    if (TPC_PER_WF >= 4) acc += __shfl_down(acc, COLS_PER_WG * 2);

    // LDS cross-wavefront reduction
    __shared__ float s_reduce[NUM_WF * COLS_PER_WG];

    if ((k_split_id & (TPC_PER_WF - 1)) == 0) {
        unsigned int wf_id = k_split_id / TPC_PER_WF;
        s_reduce[wf_id * COLS_PER_WG + col_in_wg] = acc;
    }
    __syncthreads();

    if (k_split_id == 0) {
        float sum = s_reduce[0 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[1 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[2 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[3 * COLS_PER_WG + col_in_wg];
        
        // Route output to correct buffer based on column index
        if (col < q_dim) {
            // Q output (0..q_dim-1)
            Q_out[col] = __float2half(sum);
        } else if (col < q_dim + kv_dim) {
            // K output (q_dim..q_dim+kv_dim-1)
            K_out[col - q_dim] = __float2half(sum);
        } else {
            // V output (q_dim+kv_dim..N_total-1)
            // If v_cache_dst is provided, write to cache; otherwise write to V_out
            __half* v_dst = (v_cache_dst != nullptr) ? v_cache_dst : V_out;
            v_dst[col - q_dim - kv_dim] = __float2half(sum);
        }
    }
}

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_qkv_fused_t8(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ Q_out,
    __half* __restrict__ K_out,
    __half* __restrict__ V_out,
    unsigned int K,
    unsigned int q_dim,
    unsigned int kv_dim,
    unsigned int group_size,
    __half* __restrict__ v_cache_dst
) {
    gemv_qkv_fused_cooperative<8>(A, B_q4, scales, zeros, Q_out, K_out, V_out, K, q_dim, kv_dim, group_size);
}

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_qkv_fused_t4(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ Q_out,
    __half* __restrict__ K_out,
    __half* __restrict__ V_out,
    unsigned int K,
    unsigned int q_dim,
    unsigned int kv_dim,
    unsigned int group_size,
    __half* __restrict__ v_cache_dst
) {
    gemv_qkv_fused_cooperative<4>(A, B_q4, scales, zeros, Q_out, K_out, V_out, K, q_dim, kv_dim, group_size);
}
