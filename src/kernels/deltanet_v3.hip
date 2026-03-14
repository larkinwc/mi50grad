/**
 * DeltaNet v3: Optimized Gated DeltaNet decode kernel.
 *
 * Improvements over v2:
 * 1. Parallel kq_dot (was serial 128-iter loop on thread 0)
 * 2. All 256 threads active in Pass 1 (was 128)
 * 3. Fast math (__expf instead of expf)
 * 4. Loop unrolling for state traversal ILP
 *
 * Grid: (48, 1, 1) = NUM_V_HEADS, Block: (256, 1, 1)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Allow compile-time override for tensor parallelism
#ifndef NUM_V_HEADS
#define NUM_V_HEADS 48
#endif
#ifndef NUM_K_HEADS
#define NUM_K_HEADS 16
#endif
#define K_HEAD_DIM 128
#define V_HEAD_DIM 128
#define HEADS_PER_Q (NUM_V_HEADS / NUM_K_HEADS)
#define CONV_DIM (NUM_K_HEADS * K_HEAD_DIM * 2 + NUM_V_HEADS * V_HEAD_DIM)
#define Q_DIM (NUM_K_HEADS * K_HEAD_DIM)
#define K_DIM Q_DIM
#define V_DIM (NUM_V_HEADS * V_HEAD_DIM)
#define CONV_KERNEL 4
#define BLOCK_SIZE 256

__device__ __forceinline__ float silu_fast(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float sigmoid_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down(val, 32);
    val += __shfl_down(val, 16);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 1);
    return val;
}

/**
 * Conv state shift (same as v2, unchanged).
 */
extern "C" __global__ void deltanet_conv_shift_v3(
    const __half* __restrict__ qkv_proj,
    __half* __restrict__ conv_state,
    unsigned int conv_dim
) {
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;
    unsigned int base = ch * 3;
    conv_state[base + 0] = conv_state[base + 1];
    conv_state[base + 1] = conv_state[base + 2];
    conv_state[base + 2] = qkv_proj[ch];
}

/**
 * DeltaNet v3 decode kernel.
 */
extern "C" __global__ void deltanet_decode_v3(
    const __half* __restrict__ qkv_proj,
    const __half* __restrict__ a_proj,
    const __half* __restrict__ b_proj,
    const __half* __restrict__ z_proj,
    const __half* __restrict__ conv_state,
    const float* __restrict__ conv_weight,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ norm_weight,
    float* __restrict__ state,
    __half* __restrict__ output
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 64;
    const int lane = tid % 64;

    __shared__ float s_q[K_HEAD_DIM];
    __shared__ float s_k[K_HEAD_DIM];
    __shared__ float s_v[V_HEAD_DIM];
    __shared__ float s_out[V_HEAD_DIM];
    __shared__ float s_delta[V_HEAD_DIM];
    __shared__ float s_scalar[4];  // [decay, beta, kq_dot, unused]
    __shared__ float s_warp[4];
    // Partial sums for 256-thread Pass 1
    __shared__ float s_partial_kv[BLOCK_SIZE];
    __shared__ float s_partial_sq[BLOCK_SIZE];

    int kh = head / HEADS_PER_Q;

    // === Step 1: Conv1d + SiLU ===
    int v_base = Q_DIM + K_DIM + head * V_HEAD_DIM;
    if (tid < V_HEAD_DIM) {
        int ch = v_base + tid;
        unsigned int cb = ch * 3;
        float cs0 = __half2float(conv_state[cb + 0]);
        float cs1 = __half2float(conv_state[cb + 1]);
        float cs2 = __half2float(conv_state[cb + 2]);
        float cur = __half2float(qkv_proj[ch]);

        float conv_out = cs0 * conv_weight[ch * CONV_KERNEL + 0]
                        + cs1 * conv_weight[ch * CONV_KERNEL + 1]
                        + cs2 * conv_weight[ch * CONV_KERNEL + 2]
                        + cur * conv_weight[ch * CONV_KERNEL + 3];
        s_v[tid] = silu_fast(conv_out);
    }

    int q_base = kh * K_HEAD_DIM;
    int k_base = Q_DIM + kh * K_HEAD_DIM;
    if (tid < K_HEAD_DIM) {
        int ch_q = q_base + tid;
        unsigned int cb_q = ch_q * 3;
        float qconv = __half2float(conv_state[cb_q + 0]) * conv_weight[ch_q * CONV_KERNEL + 0]
                     + __half2float(conv_state[cb_q + 1]) * conv_weight[ch_q * CONV_KERNEL + 1]
                     + __half2float(conv_state[cb_q + 2]) * conv_weight[ch_q * CONV_KERNEL + 2]
                     + __half2float(qkv_proj[ch_q]) * conv_weight[ch_q * CONV_KERNEL + 3];
        s_q[tid] = silu_fast(qconv);

        int ch_k = k_base + tid;
        unsigned int cb_k = ch_k * 3;
        float kconv = __half2float(conv_state[cb_k + 0]) * conv_weight[ch_k * CONV_KERNEL + 0]
                     + __half2float(conv_state[cb_k + 1]) * conv_weight[ch_k * CONV_KERNEL + 1]
                     + __half2float(conv_state[cb_k + 2]) * conv_weight[ch_k * CONV_KERNEL + 2]
                     + __half2float(qkv_proj[ch_k]) * conv_weight[ch_k * CONV_KERNEL + 3];
        s_k[tid] = silu_fast(kconv);
    }
    __syncthreads();

    // === Step 2: L2 normalize Q and K ===
    float q_sq = (tid < K_HEAD_DIM) ? s_q[tid] * s_q[tid] : 0.0f;
    float k_sq = (tid < K_HEAD_DIM) ? s_k[tid] * s_k[tid] : 0.0f;

    float wq = warp_reduce_sum(q_sq);
    float wk = warp_reduce_sum(k_sq);

    if (lane == 0) s_warp[warp_id] = wq;
    __syncthreads();
    float q_norm_sq = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
    float q_scale = rsqrtf(q_norm_sq + 1e-6f) / sqrtf((float)K_HEAD_DIM);

    if (lane == 0) s_warp[warp_id] = wk;
    __syncthreads();
    float k_norm_sq = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
    float k_norm_inv = rsqrtf(k_norm_sq + 1e-6f);

    if (tid < K_HEAD_DIM) {
        s_q[tid] *= q_scale;
        s_k[tid] *= k_norm_inv;
    }
    __syncthreads();

    // === Step 3: Compute decay, beta ===
    if (tid == 0) {
        float a_val = __half2float(a_proj[head]);
        float b_val = __half2float(b_proj[head]);
        float A = __expf(A_log[head]);
        float softplus_a = log1pf(__expf(a_val + dt_bias[head]));
        float g = fmaxf(-A * softplus_a, -20.0f);
        s_scalar[0] = __expf(g);            // decay
        s_scalar[1] = sigmoid_fast(b_val);   // beta
    }

    // Parallel kq_dot: all K_HEAD_DIM threads compute partial products
    float kq_partial = (tid < K_HEAD_DIM) ? s_k[tid] * s_q[tid] : 0.0f;
    float kq_warp = warp_reduce_sum(kq_partial);
    if (lane == 0) s_warp[warp_id] = kq_warp;
    __syncthreads();
    float kq_dot = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];

    float decay = s_scalar[0];
    float beta_val = s_scalar[1];

    // === Step 4: Fused DeltaNet recurrence ===
    float* head_state = state + (size_t)head * K_HEAD_DIM * V_HEAD_DIM;

    // --- Pass 1 (read-only): ALL 256 threads participate ---
    // Thread tid handles v_idx = tid % 128, iterates over K half
    // Threads 0-127: k = 0..63, Threads 128-255: k = 64..127
    {
        int v_idx = tid % V_HEAD_DIM;      // 0-127
        int k_half = tid / V_HEAD_DIM;     // 0 or 1
        int k_start = k_half * (K_HEAD_DIM / 2);
        int k_end = k_start + (K_HEAD_DIM / 2);

        float kv_acc = 0.0f;
        float sq_acc = 0.0f;
        #pragma unroll 8
        for (int k = k_start; k < k_end; k++) {
            float sv = head_state[k * V_HEAD_DIM + v_idx];
            kv_acc += sv * s_k[k];
            sq_acc += sv * s_q[k];
        }
        s_partial_kv[tid] = kv_acc;
        s_partial_sq[tid] = sq_acc;
    }
    __syncthreads();

    // Combine halves and compute delta + output
    if (tid < V_HEAD_DIM) {
        float kv_mem = (s_partial_kv[tid] + s_partial_kv[tid + V_HEAD_DIM]) * decay;
        float sq = (s_partial_sq[tid] + s_partial_sq[tid + V_HEAD_DIM]) * decay;

        float delta = (s_v[tid] - kv_mem) * beta_val;
        s_delta[tid] = delta;
        s_out[tid] = sq + delta * kq_dot;
    }
    __syncthreads();

    // --- Pass 2 (read+write): Update state ---
    const int ELEMS = K_HEAD_DIM * V_HEAD_DIM;  // 16384
    #pragma unroll 4
    for (int e = 0; e < ELEMS / BLOCK_SIZE; e++) {
        int idx = tid + e * BLOCK_SIZE;
        int k_idx = idx / V_HEAD_DIM;
        int v_idx = idx % V_HEAD_DIM;
        head_state[idx] = head_state[idx] * decay + s_k[k_idx] * s_delta[v_idx];
    }

    // === Step 5: Gated RMSNorm ===
    float out_sq = (tid < V_HEAD_DIM) ? s_out[tid] * s_out[tid] : 0.0f;
    float wsum = warp_reduce_sum(out_sq);
    if (lane == 0) s_warp[warp_id] = wsum;
    __syncthreads();

    float rms_sum = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
    float rms_inv = rsqrtf(rms_sum / V_HEAD_DIM + 1e-6f);

    if (tid < V_HEAD_DIM) {
        float normed = s_out[tid] * rms_inv * norm_weight[tid];
        float z_val = __half2float(z_proj[head * V_HEAD_DIM + tid]);
        output[head * V_HEAD_DIM + tid] = __float2half(normed * silu_fast(z_val));
    }
}
