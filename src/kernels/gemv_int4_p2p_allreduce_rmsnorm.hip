/**
 * Fused INT4 GEMV + P2P Allreduce + RMSNorm kernel for TP inference.
 *
 * Milestone: fused-gemv-ar
 *
 * This kernel eliminates intermediate buffer round-trips by fusing:
 *   1. INT4 GEMV: Dequantizes INT4 weights, computes FP16 GEMV with FP32 accumulation
 *   2. P2P Allreduce: Each GPU reads peer partials via BAR1 P2P pointers
 *   3. FP32 Accumulation: Reduces all partials with FP32 precision
 *   4. RMSNorm: Applies RMSNorm with learned weights
 *   5. Write Result: Single write to output buffer
 *
 * Compared to separate kernels (gemv_int4_v6 + kernel_p2p_allreduce + rmsnorm_v3):
 *   - Eliminates 2 intermediate buffer writes per token
 *   - Eliminates 2 kernel launches per layer per token
 *   - Reduces kernel launch count from 192 to 64 per token (for 64-layer model)
 *
 * Each GPU runs the fused kernel simultaneously:
 *   - A: FP16 activation buffer (local, read-only)
 *   - B_q4: INT4 weights in GPTQ format (local, read-only)
 *   - scales, zeros: Per-group quantization params (local, read-only)
 *   - partial_peer0/1/2: Peer GPU partial buffers (P2P BAR1 pointers, read-only)
 *   - weight: RMSNorm weight buffer (local, read-only)
 *   - output: Output buffer (local, writes fused result)
 *
 * Thread layout (same as gemv_int4_v6_t16):
 *   - 16 threads per output column
 *   - 256/16 = 16 columns per workgroup
 *   - Cooperative reduction via DPP + LDS
 *
 * Target: gfx906 (MI50/MI60), compiled as SHARED LIBRARY with:
 *   hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC
 *
 * Exposes host-callable C functions for ctypes-based dispatch.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/* ===================================================================
 * Device helper functions
 * =================================================================== */

/**
 * dequant_fdot2: Process one uint32 (8 INT4 weights) via fdot2.
 * Extracts nibbles, applies scale/zero dequantization, uses v_dot2_f32_f16.
 */
__device__ __forceinline__ float dequant_fdot2(
    unsigned int packed,
    float cur_scale,
    float cur_zero,
    const __half* __restrict__ A,
    unsigned int a_base
) {
    __half2 a01 = *reinterpret_cast<const __half2*>(A + a_base);
    __half2 a23 = *reinterpret_cast<const __half2*>(A + a_base + 2);
    __half2 a45 = *reinterpret_cast<const __half2*>(A + a_base + 4);
    __half2 a67 = *reinterpret_cast<const __half2*>(A + a_base + 6);

    float scale = cur_scale;
    float zero  = cur_zero;

    float w0 = ((float)__builtin_amdgcn_ubfe(packed,  0, 4) - zero) * scale;
    float w1 = ((float)__builtin_amdgcn_ubfe(packed,  4, 4) - zero) * scale;
    float w2 = ((float)__builtin_amdgcn_ubfe(packed,  8, 4) - zero) * scale;
    float w3 = ((float)__builtin_amdgcn_ubfe(packed, 12, 4) - zero) * scale;
    float w4 = ((float)__builtin_amdgcn_ubfe(packed, 16, 4) - zero) * scale;
    float w5 = ((float)__builtin_amdgcn_ubfe(packed, 20, 4) - zero) * scale;
    float w6 = ((float)__builtin_amdgcn_ubfe(packed, 24, 4) - zero) * scale;
    float w7 = ((float)__builtin_amdgcn_ubfe(packed, 28, 4) - zero) * scale;

    __half2 w01 = __halves2half2(__float2half(w0), __float2half(w1));
    __half2 w23 = __halves2half2(__float2half(w2), __float2half(w3));
    __half2 w45 = __halves2half2(__float2half(w4), __float2half(w5));
    __half2 w67 = __halves2half2(__float2half(w6), __float2half(w7));

    float acc = 0.0f;
    acc = __builtin_amdgcn_fdot2(w01, a01, acc, false);
    acc = __builtin_amdgcn_fdot2(w23, a23, acc, false);
    acc = __builtin_amdgcn_fdot2(w45, a45, acc, false);
    acc = __builtin_amdgcn_fdot2(w67, a67, acc, false);

    return acc;
}

/* ===================================================================
 * Fused GEMV + P2P Allreduce + RMSNorm kernel (TP=4)
 * =================================================================== */

/**
 * gemv_int4_p2p_allreduce_rmsnorm_tp4_kernel:
 *
 *   output[i] = rmsnorm(
 *     gemv_int4(A, B_q4, scales, zeros)[i] 
 *     + partial_local[i] + partial_peer0[i] + partial_peer1[i] + partial_peer2[i]
 *   ) * weight[i]
 *
 * Runs on EACH of the 4 GPUs simultaneously. Each GPU:
 *   - Computes its partial GEMV result for assigned columns (N/4 columns per GPU)
 *   - Reads 3 peer GPU partial buffers via P2P BAR1 pointers
 *   - Reduces all 4 partials with FP32 accumulation
 *   - Applies RMSNorm with sum-of-squares reduction
 *   - Writes final normalized result to output buffer
 *
 * Tensor Parallelism: Each GPU handles N/TP output columns.
 *   - GPU 0: columns [0, N/4)
 *   - GPU 1: columns [N/4, N/2)
 *   - GPU 2: columns [N/2, 3N/4)
 *   - GPU 3: columns [3N/4, N)
 * The partial buffers are full-size [N], but each GPU only accesses its partition.
 *
 * Grid: (ceil((N/TP)/16), 1, 1), Block: (256, 1, 1)
 * Each workgroup processes 16 output columns within this GPU's partition.
 * Within each WG: 16 threads/col cooperate via DPP + LDS.
 *
 * @param output        Output buffer (fused result) [N/TP] for this GPU's partition
 * @param A             FP16 activation buffer [K]
 * @param B_q4          INT4 weights in GPTQ format [K/8, N/TP] (partitioned)
 * @param scales        Per-group scales [K/group_size, N/TP] (partitioned)
 * @param zeros         Per-group zeros [K/group_size, N/TP] (partitioned)
 * @param partial_local This GPU's partial buffer [N] (full size, P2P accessible)
 * @param partial_peer0 First peer GPU's partial buffer [N] (P2P BAR1 pointer)
 * @param partial_peer1 Second peer GPU's partial buffer [N] (P2P BAR1 pointer)
 * @param partial_peer2 Third peer GPU's partial buffer [N] (P2P BAR1 pointer)
 * @param weight        RMSNorm weight buffer [N] (full size, replicated)
 * @param K             Input dimension (activation size)
 * @param N             Total output dimension (all GPUs combined)
 * @param dim           Hidden dimension for RMSNorm (typically = N)
 * @param group_size    Quantization group size (e.g., 128)
 * @param eps           RMSNorm epsilon (default: 1e-6)
 * @param tp_rank       This GPU's rank in TP group (0-3)
 * @param tp_size       Tensor parallel size (4)
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void gemv_int4_p2p_allreduce_rmsnorm_tp4_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    const __half* __restrict__ partial_local,
    const __half* __restrict__ partial_peer0,
    const __half* __restrict__ partial_peer1,
    const __half* __restrict__ partial_peer2,
    const __half* __restrict__ weight,
    unsigned int K,
    unsigned int N,
    unsigned int dim,
    unsigned int group_size,
    float eps,
    unsigned int tp_rank,
    unsigned int tp_size
) {
    constexpr unsigned int THREADS_PER_COL = 16;
    constexpr unsigned int COLS_PER_WG     = 256 / THREADS_PER_COL;  // 16
    constexpr unsigned int NUM_WF          = 4;
    constexpr unsigned int TPC_PER_WF      = THREADS_PER_COL / NUM_WF;  // 4

    unsigned int col_in_wg = threadIdx.x % COLS_PER_WG;
    unsigned int k_split_id = threadIdx.x / COLS_PER_WG;
    
    // Compute global column index for this GPU's partition
    unsigned int cols_per_gpu = (N + tp_size - 1) / tp_size;  // ceil(N/tp_size)
    unsigned int col_start = tp_rank * cols_per_gpu;
    unsigned int local_col = blockIdx.x * COLS_PER_WG + col_in_wg;
    unsigned int col = col_start + local_col;
    
    // Check bounds for this GPU's partition
    if (local_col >= cols_per_gpu || col >= N) return;

    unsigned int wf_id = threadIdx.x / 64;
    unsigned int lane   = threadIdx.x % 64;

    // Compute K-split range for this thread
    unsigned int num_k_groups = K >> 3;  // K/8 (8 weights per uint32)
    unsigned int groups_per_scale = group_size >> 3;
    unsigned int gps_log2 = 31u - __builtin_clz(groups_per_scale);

    unsigned int kg_start = (num_k_groups * k_split_id) / THREADS_PER_COL;
    unsigned int kg_end   = (num_k_groups * (k_split_id + 1)) / THREADS_PER_COL;

    // Compute column index within this GPU's weight partition
    unsigned int col_in_partition = local_col;

    // =================================================================
    // Phase 1: INT4 GEMV with register-cached scale/zero + weight prefetch
    // Computes partial GEMV result for this column
    // =================================================================
    float gemv_acc = 0.0f;
    
    // Register-cached scale/zero
    float cur_scale = 0.0f, cur_zero = 0.0f;
    unsigned int current_group = 0xFFFFFFFF;
    
    // Weight prefetch with 2-register double buffering
    unsigned int packed = 0;
    if (kg_start < kg_end) {
        packed = B_q4[kg_start * cols_per_gpu + col_in_partition];
    }
    
    for (unsigned int kg = kg_start; kg < kg_end; kg++) {
        // Check if we're crossing a quantization group boundary
        unsigned int next_group = kg >> gps_log2;
        if (next_group != current_group) {
            cur_scale = __half2float(scales[next_group * cols_per_gpu + col_in_partition]);
            cur_zero  = __half2float(zeros[next_group * cols_per_gpu + col_in_partition]);
            current_group = next_group;
        }
        
        // Compute with current weight
        gemv_acc += dequant_fdot2(packed, cur_scale, cur_zero, A, kg << 3);
        
        // Prefetch next weight
        if (kg + 1 < kg_end) {
            packed = B_q4[(kg + 1) * cols_per_gpu + col_in_partition];
        }
    }

    // Phase 1b: Intra-wavefront DPP reduction
    if (TPC_PER_WF >= 2) gemv_acc += __shfl_down(gemv_acc, COLS_PER_WG);
    if (TPC_PER_WF >= 4) gemv_acc += __shfl_down(gemv_acc, COLS_PER_WG * 2);

    // Phase 1c: LDS cross-wavefront reduction
    __shared__ float s_reduce[NUM_WF * COLS_PER_WG];

    if ((k_split_id & (TPC_PER_WF - 1)) == 0) {
        unsigned int wf_id_local = k_split_id / TPC_PER_WF;
        s_reduce[wf_id_local * COLS_PER_WG + col_in_wg] = gemv_acc;
    }
    __syncthreads();

    float partial_result = 0.0f;
    if (k_split_id == 0) {
        float sum = s_reduce[0 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[1 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[2 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[3 * COLS_PER_WG + col_in_wg];
        partial_result = sum;
    }
    __syncthreads();

    // =================================================================
    // Phase 2: P2P Allreduce - Read peer partials and reduce
    // Each GPU reads all 4 partials (local + 3 peers) via P2P BAR1
    // Access uses global column index (col) for P2P buffers
    // =================================================================
    float total = partial_result;
    
    // Add peer partials via P2P BAR1 pointers
    // partial_local and peer buffers are full-size [N], accessed at global col
    if (partial_local != nullptr) {
        total += __half2float(partial_local[col]);
    }
    total += __half2float(partial_peer0[col]);
    total += __half2float(partial_peer1[col]);
    total += __half2float(partial_peer2[col]);

    // =================================================================
    // Phase 3: RMSNorm - Compute sum-of-squares across all columns
    // Each thread contributes to the reduction for its column
    // =================================================================
    float sum_sq = total * total;

    // Phase 3b: Intra-wavefront DPP reduction for sum_sq
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);

    // Phase 3c: Cross-warp reduction via LDS
    __shared__ float s_warp[4];
    if (lane == 0) s_warp[wf_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < 4) {
        float warp_sum = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
        s_warp[threadIdx.x] = warp_sum;
    }
    __syncthreads();
    
    float total_sq = s_warp[0];

    // Compute inverse RMS
    float rms_inv = rsqrtf(total_sq / (float)dim + eps);

    // =================================================================
    // Phase 4: Apply RMSNorm + weight, write final result
    // Only k_split_id == 0 threads write output
    // Weight is accessed at global col, output at local_col
    // =================================================================
    if (k_split_id == 0) {
        float normalized = total * rms_inv;
        float w = __half2float(weight[col]);  // Full-size weight buffer
        output[local_col] = __float2half(normalized * w);  // Output is partition-local
    }
}

/* ===================================================================
 * Fused GEMV + P2P Allreduce + RMSNorm kernel (TP=2)
 * =================================================================== */

/**
 * gemv_int4_p2p_allreduce_rmsnorm_tp2_kernel:
 *
 *   output[i] = rmsnorm(
 *     gemv_int4(A, B_q4, scales, zeros)[i] 
 *     + partial_local[i] + partial_peer0[i]
 *   ) * weight[i]
 *
 * Runs on BOTH GPUs simultaneously (TP=2).
 * Tensor Parallelism: Each GPU handles N/2 output columns.
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void gemv_int4_p2p_allreduce_rmsnorm_tp2_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    const __half* __restrict__ partial_local,
    const __half* __restrict__ partial_peer0,
    const __half* __restrict__ weight,
    unsigned int K,
    unsigned int N,
    unsigned int dim,
    unsigned int group_size,
    float eps,
    unsigned int tp_rank,
    unsigned int tp_size
) {
    constexpr unsigned int THREADS_PER_COL = 16;
    constexpr unsigned int COLS_PER_WG     = 256 / THREADS_PER_COL;
    constexpr unsigned int NUM_WF          = 4;
    constexpr unsigned int TPC_PER_WF      = THREADS_PER_COL / NUM_WF;

    unsigned int col_in_wg = threadIdx.x % COLS_PER_WG;
    unsigned int k_split_id = threadIdx.x / COLS_PER_WG;
    
    // Compute global column index for this GPU's partition
    unsigned int cols_per_gpu = (N + tp_size - 1) / tp_size;  // ceil(N/tp_size)
    unsigned int col_start = tp_rank * cols_per_gpu;
    unsigned int local_col = blockIdx.x * COLS_PER_WG + col_in_wg;
    unsigned int col = col_start + local_col;
    
    // Check bounds for this GPU's partition
    if (local_col >= cols_per_gpu || col >= N) return;

    unsigned int wf_id = threadIdx.x / 64;
    unsigned int lane   = threadIdx.x % 64;

    unsigned int num_k_groups = K >> 3;
    unsigned int groups_per_scale = group_size >> 3;
    unsigned int gps_log2 = 31u - __builtin_clz(groups_per_scale);

    unsigned int kg_start = (num_k_groups * k_split_id) / THREADS_PER_COL;
    unsigned int kg_end   = (num_k_groups * (k_split_id + 1)) / THREADS_PER_COL;

    unsigned int col_in_partition = local_col;

    // Phase 1: INT4 GEMV
    float gemv_acc = 0.0f;
    float cur_scale = 0.0f, cur_zero = 0.0f;
    unsigned int current_group = 0xFFFFFFFF;
    unsigned int packed = 0;
    
    if (kg_start < kg_end) {
        packed = B_q4[kg_start * cols_per_gpu + col_in_partition];
    }
    
    for (unsigned int kg = kg_start; kg < kg_end; kg++) {
        unsigned int next_group = kg >> gps_log2;
        if (next_group != current_group) {
            cur_scale = __half2float(scales[next_group * cols_per_gpu + col_in_partition]);
            cur_zero  = __half2float(zeros[next_group * cols_per_gpu + col_in_partition]);
            current_group = next_group;
        }
        
        gemv_acc += dequant_fdot2(packed, cur_scale, cur_zero, A, kg << 3);
        
        if (kg + 1 < kg_end) {
            packed = B_q4[(kg + 1) * cols_per_gpu + col_in_partition];
        }
    }

    // Phase 1b: DPP reduction
    if (TPC_PER_WF >= 2) gemv_acc += __shfl_down(gemv_acc, COLS_PER_WG);
    if (TPC_PER_WF >= 4) gemv_acc += __shfl_down(gemv_acc, COLS_PER_WG * 2);

    // Phase 1c: LDS reduction
    __shared__ float s_reduce[NUM_WF * COLS_PER_WG];

    if ((k_split_id & (TPC_PER_WF - 1)) == 0) {
        unsigned int wf_id_local = k_split_id / TPC_PER_WF;
        s_reduce[wf_id_local * COLS_PER_WG + col_in_wg] = gemv_acc;
    }
    __syncthreads();

    float partial_result = 0.0f;
    if (k_split_id == 0) {
        float sum = s_reduce[0 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[1 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[2 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[3 * COLS_PER_WG + col_in_wg];
        partial_result = sum;
    }
    __syncthreads();

    // Phase 2: P2P Allreduce (TP=2)
    float total = partial_result;
    if (partial_local != nullptr) {
        total += __half2float(partial_local[col]);
    }
    total += __half2float(partial_peer0[col]);

    // Phase 3: RMSNorm
    float sum_sq = total * total;
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);

    __shared__ float s_warp[4];
    if (lane == 0) s_warp[wf_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < 4) {
        float warp_sum = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
        s_warp[threadIdx.x] = warp_sum;
    }
    __syncthreads();
    
    float total_sq = s_warp[0];
    float rms_inv = rsqrtf(total_sq / (float)dim + eps);

    // Phase 4: Write output
    if (k_split_id == 0) {
        float normalized = total * rms_inv;
        float w = __half2float(weight[col]);
        output[local_col] = __float2half(normalized * w);
    }
}

/* ===================================================================
 * Host-callable C wrappers for Python ctypes dispatch
 * =================================================================== */

/**
 * gemv_int4_p2p_allreduce_rmsnorm_tp4:
 *
 * Launch the TP=4 fused GEMV + P2P allreduce + RMSNorm kernel.
 *
 * Args:
 *   output:        Output buffer (fused result) [N/4] for this GPU's partition
 *   A:             FP16 activation buffer [K]
 *   B_q4:          INT4 weights [K/8, N/4] (partitioned)
 *   scales:        Per-group scales [K/group_size, N/4] (partitioned)
 *   zeros:         Per-group zeros [K/group_size, N/4] (partitioned)
 *   partial_local: This GPU's partial buffer [N] (or NULL)
 *   partial_peer0: First peer's partial buffer [N] (P2P pointer)
 *   partial_peer1: Second peer's partial buffer [N] (P2P pointer)
 *   partial_peer2: Third peer's partial buffer [N] (P2P pointer)
 *   weight:        RMSNorm weight buffer [N] (full size)
 *   K:             Input dimension
 *   N:             Total output dimension (all GPUs combined)
 *   dim:           Hidden dimension for RMSNorm
 *   group_size:    Quantization group size
 *   eps:           RMSNorm epsilon
 *   tp_rank:       This GPU's rank (0-3)
 *   tp_size:       Tensor parallel size (4)
 *   stream:        HIP stream
 */
extern "C" int gemv_int4_p2p_allreduce_rmsnorm_tp4(
    void* output,
    const void* A,
    const unsigned int* B_q4,
    const void* scales,
    const void* zeros,
    const void* partial_local,
    const void* partial_peer0,
    const void* partial_peer1,
    const void* partial_peer2,
    const void* weight,
    unsigned int K,
    unsigned int N,
    unsigned int dim,
    unsigned int group_size,
    float eps,
    unsigned int tp_rank,
    unsigned int tp_size,
    hipStream_t stream)
{
    (void)hipGetLastError();  // Clear stale error
    unsigned int cols_per_gpu = (N + tp_size - 1) / tp_size;
    unsigned int blocks = (cols_per_gpu + 16 - 1) / 16;  // ceil(cols_per_gpu/16)
    
    hipLaunchKernelGGL(gemv_int4_p2p_allreduce_rmsnorm_tp4_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)output,
                       (const __half*)A,
                       (const unsigned int*)B_q4,
                       (const __half*)scales,
                       (const __half*)zeros,
                       (const __half*)partial_local,
                       (const __half*)partial_peer0,
                       (const __half*)partial_peer1,
                       (const __half*)partial_peer2,
                       (const __half*)weight,
                       K, N, dim, group_size, eps,
                       tp_rank, tp_size);
    
    return (int)hipGetLastError();
}

/**
 * gemv_int4_p2p_allreduce_rmsnorm_tp2:
 *
 * Launch the TP=2 fused GEMV + P2P allreduce + RMSNorm kernel.
 */
extern "C" int gemv_int4_p2p_allreduce_rmsnorm_tp2(
    void* output,
    const void* A,
    const unsigned int* B_q4,
    const void* scales,
    const void* zeros,
    const void* partial_local,
    const void* partial_peer0,
    const void* weight,
    unsigned int K,
    unsigned int N,
    unsigned int dim,
    unsigned int group_size,
    float eps,
    unsigned int tp_rank,
    unsigned int tp_size,
    hipStream_t stream)
{
    (void)hipGetLastError();  // Clear stale error
    unsigned int cols_per_gpu = (N + tp_size - 1) / tp_size;
    unsigned int blocks = (cols_per_gpu + 16 - 1) / 16;  // ceil(cols_per_gpu/16)
    
    hipLaunchKernelGGL(gemv_int4_p2p_allreduce_rmsnorm_tp2_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)output,
                       (const __half*)A,
                       (const unsigned int*)B_q4,
                       (const __half*)scales,
                       (const __half*)zeros,
                       (const __half*)partial_local,
                       (const __half*)partial_peer0,
                       (const __half*)weight,
                       K, N, dim, group_size, eps,
                       tp_rank, tp_size);
    
    return (int)hipGetLastError();
}
