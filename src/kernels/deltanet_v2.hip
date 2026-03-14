/**
 * Optimized Gated DeltaNet decode kernel for Qwen 3.5 linear attention.
 *
 * Key optimizations over v1:
 * 1. Fused 2-pass state traversal (was 4 passes):
 *    - Pass 1 (read): accumulate kv_mem and sq from state
 *    - Output via algebraic reformulation: out = sq + delta * dot(k,q)
 *    - Pass 2 (read+write): update state = S*decay + k*delta
 *    Halves global memory traffic (192KB vs 384KB per head).
 * 2. Warp shuffle reductions (eliminates 16+ barrier syncs from tree reductions)
 * 3. Race-free conv state: main kernel reads conv_state read-only,
 *    separate deltanet_conv_shift kernel updates it afterward.
 *
 * Launch order per layer:
 *   1. deltanet_decode_v2  (reads conv_state, does full recurrence)
 *   2. deltanet_conv_shift (shifts conv_state for next token)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define NUM_V_HEADS 48
#define NUM_K_HEADS 16
#define K_HEAD_DIM 128
#define V_HEAD_DIM 128
#define HEADS_PER_Q (NUM_V_HEADS / NUM_K_HEADS)
#define CONV_DIM (NUM_K_HEADS * K_HEAD_DIM * 2 + NUM_V_HEADS * V_HEAD_DIM)  // 10240
#define Q_DIM (NUM_K_HEADS * K_HEAD_DIM)   // 2048
#define K_DIM Q_DIM                         // 2048
#define V_DIM (NUM_V_HEADS * V_HEAD_DIM)   // 6144
#define CONV_KERNEL 4
#define BLOCK_SIZE 256

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
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
 * Shift conv state after main kernel completes.
 * conv_state: [CONV_DIM, 3] stores [h[-3], h[-2], h[-1]]
 * After shift: [h[-2], h[-1], current_input]
 * Grid: (ceil(conv_dim/256), 1, 1), Block: (256, 1, 1)
 */
extern "C" __global__ void deltanet_conv_shift(
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
 * Optimized DeltaNet decode kernel.
 *
 * Reads conv_state as [h[-3], h[-2], h[-1]] (read-only).
 * Uses qkv_proj as current input for conv.
 * Grid: (48, 1, 1), Block: (256, 1, 1)
 */
extern "C" __global__ void deltanet_decode_v2(
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

    __shared__ float s_q[K_HEAD_DIM];
    __shared__ float s_k[K_HEAD_DIM];
    __shared__ float s_v[V_HEAD_DIM];
    __shared__ float s_kv_mem[V_HEAD_DIM];
    __shared__ float s_out[V_HEAD_DIM];
    __shared__ float s_delta[V_HEAD_DIM];
    __shared__ float s_scalar[4];  // [decay, beta, kq_dot, unused]
    __shared__ float s_warp[4];    // warp reduction scratch

    int kh = head / HEADS_PER_Q;

    // === Step 1: Conv1d + SiLU ===
    // Conv state layout: [ch, 3] = [h[-3], h[-2], h[-1]]
    // Conv: h[-3]*w[0] + h[-2]*w[1] + h[-1]*w[2] + current*w[3]

    // V channels (unique per value head, no race)
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
        s_v[tid] = silu_f(conv_out);
    }

    // Q/K channels (shared across 3 value heads, read-only - no race)
    int q_base = kh * K_HEAD_DIM;
    int k_base = Q_DIM + kh * K_HEAD_DIM;
    if (tid < K_HEAD_DIM) {
        // Q conv
        int ch_q = q_base + tid;
        unsigned int cb_q = ch_q * 3;
        float qs0 = __half2float(conv_state[cb_q + 0]);
        float qs1 = __half2float(conv_state[cb_q + 1]);
        float qs2 = __half2float(conv_state[cb_q + 2]);
        float qcur = __half2float(qkv_proj[ch_q]);

        float qconv = qs0 * conv_weight[ch_q * CONV_KERNEL + 0]
                     + qs1 * conv_weight[ch_q * CONV_KERNEL + 1]
                     + qs2 * conv_weight[ch_q * CONV_KERNEL + 2]
                     + qcur * conv_weight[ch_q * CONV_KERNEL + 3];
        s_q[tid] = silu_f(qconv);

        // K conv
        int ch_k = k_base + tid;
        unsigned int cb_k = ch_k * 3;
        float ks0 = __half2float(conv_state[cb_k + 0]);
        float ks1 = __half2float(conv_state[cb_k + 1]);
        float ks2 = __half2float(conv_state[cb_k + 2]);
        float kcur = __half2float(qkv_proj[ch_k]);

        float kconv = ks0 * conv_weight[ch_k * CONV_KERNEL + 0]
                     + ks1 * conv_weight[ch_k * CONV_KERNEL + 1]
                     + ks2 * conv_weight[ch_k * CONV_KERNEL + 2]
                     + kcur * conv_weight[ch_k * CONV_KERNEL + 3];
        s_k[tid] = silu_f(kconv);
    }
    __syncthreads();

    // === Step 2: L2 normalize Q and K ===
    float q_sq = (tid < K_HEAD_DIM) ? s_q[tid] * s_q[tid] : 0.0f;
    float k_sq = (tid < K_HEAD_DIM) ? s_k[tid] * s_k[tid] : 0.0f;

    // Warp-level reduction + cross-warp combine
    float wq = warp_reduce_sum(q_sq);
    float wk = warp_reduce_sum(k_sq);
    int warp_id = tid / 64;
    int lane = tid % 64;

    // Store warp partial sums (reuse s_warp for q, then k)
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

    // === Step 3: Compute decay, beta, and dot(k,q) ===
    if (tid == 0) {
        float a_val = __half2float(a_proj[head]);
        float b_val = __half2float(b_proj[head]);
        float A = expf(A_log[head]);
        float softplus_a = log1pf(expf(a_val + dt_bias[head]));
        float g = fmaxf(-A * softplus_a, -20.0f);
        s_scalar[0] = expf(g);          // decay
        s_scalar[1] = sigmoid_f(b_val); // beta

        // Precompute dot(k, q) for output formula
        float kq = 0.0f;
        for (int i = 0; i < K_HEAD_DIM; i++) {
            kq += s_k[i] * s_q[i];
        }
        s_scalar[2] = kq;               // kq_dot
    }
    __syncthreads();

    float decay = s_scalar[0];
    float beta_val = s_scalar[1];
    float kq_dot = s_scalar[2];

    // === Step 4: Fused DeltaNet recurrence ===
    float* head_state = state + (size_t)head * K_HEAD_DIM * V_HEAD_DIM;

    // --- Pass 1 (read-only): Accumulate kv_mem and sq ---
    // kv_mem[v] = sum_k(S[k,v] * decay * k[k])
    // sq[v]     = sum_k(S[k,v] * decay * q[k])
    if (tid < V_HEAD_DIM) {
        float kv_acc = 0.0f;
        float sq_acc = 0.0f;
        for (int k = 0; k < K_HEAD_DIM; k++) {
            float sv = head_state[k * V_HEAD_DIM + tid];
            kv_acc += sv * s_k[k];
            sq_acc += sv * s_q[k];
        }
        // Multiply by decay once (factored out of inner loop)
        s_kv_mem[tid] = kv_acc * decay;
        s_out[tid] = sq_acc * decay;
    }
    __syncthreads();

    // Compute delta and final output
    // delta[v] = (v[v] - kv_mem[v]) * beta
    // out[v] = sq[v] + delta[v] * dot(k, q)  (algebraic reformulation)
    if (tid < V_HEAD_DIM) {
        float delta = (s_v[tid] - s_kv_mem[tid]) * beta_val;
        s_delta[tid] = delta;
        s_out[tid] = s_out[tid] + delta * kq_dot;
    }
    __syncthreads();

    // --- Pass 2 (read+write): Update state ---
    // S_new[k,v] = S[k,v] * decay + k[k] * delta[v]
    const int ELEMS = K_HEAD_DIM * V_HEAD_DIM;  // 16384
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
        output[head * V_HEAD_DIM + tid] = __float2half(normed * silu_f(z_val));
    }
}
