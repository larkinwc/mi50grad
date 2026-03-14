/**
 * Gated DeltaNet decode kernel for Qwen 3.5 linear attention layers.
 *
 * Performs the full DeltaNet decode step on GPU:
 * 1. Causal conv1d (depthwise, kernel=4, SiLU activation)
 * 2. Split Q/K/V, L2 normalize, head expansion
 * 3. Compute decay and beta from a, b inputs
 * 4. DeltaNet recurrence: decay state, read, delta, write, output
 * 5. Gated RMSNorm: RMSNorm(output) * SiLU(z)
 *
 * One workgroup per value head (48 workgroups).
 * 256 threads per workgroup for matrix-vector products.
 *
 * Config: 48 value heads, 16 key heads (3:1 expansion),
 *         key_head_dim=128, value_head_dim=128, conv_kernel=4
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Constants for Qwen 3.5
#define NUM_V_HEADS 48
#define NUM_K_HEADS 16
#define K_HEAD_DIM 128
#define V_HEAD_DIM 128
#define HEADS_PER_Q (NUM_V_HEADS / NUM_K_HEADS)  // 3
#define CONV_DIM (NUM_K_HEADS * K_HEAD_DIM * 2 + NUM_V_HEADS * V_HEAD_DIM)  // 10240
#define Q_DIM (NUM_K_HEADS * K_HEAD_DIM)   // 2048
#define K_DIM Q_DIM                         // 2048
#define V_DIM (NUM_V_HEADS * V_HEAD_DIM)   // 6144
#define CONV_KERNEL 4
#define BLOCK_SIZE 256

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * DeltaNet decode kernel.
 *
 * Args:
 *   qkv_proj:   [CONV_DIM] FP16 - output of in_proj_qkv GEMV
 *   a_proj:     [NUM_V_HEADS] FP16 - output of in_proj_a GEMV
 *   b_proj:     [NUM_V_HEADS] FP16 - output of in_proj_b GEMV
 *   z_proj:     [V_DIM] FP16 - output of in_proj_z GEMV
 *   conv_state: [CONV_DIM, CONV_KERNEL-1] FP16 - causal conv state
 *   conv_weight:[CONV_DIM, CONV_KERNEL] FP32 - conv1d weights
 *   A_log:      [NUM_V_HEADS] FP32 - log decay params
 *   dt_bias:    [NUM_V_HEADS] FP32 - dt bias params
 *   norm_weight:[V_HEAD_DIM] FP32 - gated RMSNorm weight
 *   state:      [NUM_V_HEADS, K_HEAD_DIM, V_HEAD_DIM] FP32 - recurrent state
 *   output:     [V_DIM] FP16 - output buffer
 */
extern "C" __global__ void deltanet_decode(
    const __half* __restrict__ qkv_proj,
    const __half* __restrict__ a_proj,
    const __half* __restrict__ b_proj,
    const __half* __restrict__ z_proj,
    __half* __restrict__ conv_state,
    const float* __restrict__ conv_weight,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ norm_weight,
    float* __restrict__ state,
    __half* __restrict__ output
) {
    const int head = blockIdx.x;  // 0..47 value head
    const int tid = threadIdx.x;  // 0..255

    // Shared memory for intermediate results
    __shared__ float s_q[K_HEAD_DIM];   // 128
    __shared__ float s_k[K_HEAD_DIM];   // 128
    __shared__ float s_v[V_HEAD_DIM];   // 128
    __shared__ float s_out[V_HEAD_DIM]; // 128
    __shared__ float s_reduce[BLOCK_SIZE]; // for reductions

    // === Step 1: Conv1d + SiLU for this head's Q/K/V channels ===
    // Each value head h maps to key head h/3
    // Q channels: [kh*K_HEAD_DIM .. kh*K_HEAD_DIM + K_HEAD_DIM) for key head kh = head/3
    // K channels: [Q_DIM + kh*K_HEAD_DIM .. Q_DIM + kh*K_HEAD_DIM + K_HEAD_DIM)
    // V channels: [Q_DIM + K_DIM + head*V_HEAD_DIM .. Q_DIM + K_DIM + head*V_HEAD_DIM + V_HEAD_DIM)

    int kh = head / HEADS_PER_Q;  // key head index

    // Process V channels (128 channels per head, perfectly fits 256 threads with 2 iterations)
    // Actually with 256 threads and 128 channels, each thread handles ~0.5 channels
    // Let's have first 128 threads handle V, threads 128-255 idle for this part

    // Conv1d for V channels: offset = Q_DIM + K_DIM + head * V_HEAD_DIM
    int v_base = Q_DIM + K_DIM + head * V_HEAD_DIM;
    if (tid < V_HEAD_DIM) {
        int ch = v_base + tid;
        // conv_state[ch, 0..2] are the 3 previous inputs
        // qkv_proj[ch] is the current input
        float cs0 = __half2float(conv_state[ch * (CONV_KERNEL - 1) + 0]);
        float cs1 = __half2float(conv_state[ch * (CONV_KERNEL - 1) + 1]);
        float cs2 = __half2float(conv_state[ch * (CONV_KERNEL - 1) + 2]);
        float cur = __half2float(qkv_proj[ch]);

        // Update conv state: shift left, append current
        conv_state[ch * (CONV_KERNEL - 1) + 0] = conv_state[ch * (CONV_KERNEL - 1) + 1];
        conv_state[ch * (CONV_KERNEL - 1) + 1] = conv_state[ch * (CONV_KERNEL - 1) + 2];
        conv_state[ch * (CONV_KERNEL - 1) + 2] = __float2half(cur);

        // Depthwise conv: sum(input[k] * weight[k])
        float conv_out = cs0 * conv_weight[ch * CONV_KERNEL + 0]
                        + cs1 * conv_weight[ch * CONV_KERNEL + 1]
                        + cs2 * conv_weight[ch * CONV_KERNEL + 2]
                        + cur * conv_weight[ch * CONV_KERNEL + 3];

        // SiLU
        s_v[tid] = silu(conv_out);
    }

    // Conv1d for Q channels (only for heads where head % 3 == 0 to avoid redundant work)
    // Actually, each value head needs its own Q/K from the corresponding key head
    // Multiple value heads share the same key head, but the conv result is the same
    // So we can compute Q/K conv in any of the 3 value heads and broadcast
    // For simplicity, let every head compute its own Q/K (redundant but simple)
    int q_base = kh * K_HEAD_DIM;
    int k_base = Q_DIM + kh * K_HEAD_DIM;
    if (tid < K_HEAD_DIM) {
        // Q conv
        int ch_q = q_base + tid;
        float cs0 = __half2float(conv_state[ch_q * (CONV_KERNEL - 1) + 0]);
        float cs1 = __half2float(conv_state[ch_q * (CONV_KERNEL - 1) + 1]);
        float cs2 = __half2float(conv_state[ch_q * (CONV_KERNEL - 1) + 2]);
        float cur = __half2float(qkv_proj[ch_q]);

        // Update conv state for Q (only do this once per key head)
        if (head % HEADS_PER_Q == 0) {
            conv_state[ch_q * (CONV_KERNEL - 1) + 0] = conv_state[ch_q * (CONV_KERNEL - 1) + 1];
            conv_state[ch_q * (CONV_KERNEL - 1) + 1] = conv_state[ch_q * (CONV_KERNEL - 1) + 2];
            conv_state[ch_q * (CONV_KERNEL - 1) + 2] = __float2half(cur);
        }

        float conv_out = cs0 * conv_weight[ch_q * CONV_KERNEL + 0]
                        + cs1 * conv_weight[ch_q * CONV_KERNEL + 1]
                        + cs2 * conv_weight[ch_q * CONV_KERNEL + 2]
                        + cur * conv_weight[ch_q * CONV_KERNEL + 3];
        s_q[tid] = silu(conv_out);

        // K conv
        int ch_k = k_base + tid;
        cs0 = __half2float(conv_state[ch_k * (CONV_KERNEL - 1) + 0]);
        cs1 = __half2float(conv_state[ch_k * (CONV_KERNEL - 1) + 1]);
        cs2 = __half2float(conv_state[ch_k * (CONV_KERNEL - 1) + 2]);
        cur = __half2float(qkv_proj[ch_k]);

        if (head % HEADS_PER_Q == 0) {
            conv_state[ch_k * (CONV_KERNEL - 1) + 0] = conv_state[ch_k * (CONV_KERNEL - 1) + 1];
            conv_state[ch_k * (CONV_KERNEL - 1) + 1] = conv_state[ch_k * (CONV_KERNEL - 1) + 2];
            conv_state[ch_k * (CONV_KERNEL - 1) + 2] = __float2half(cur);
        }

        conv_out = cs0 * conv_weight[ch_k * CONV_KERNEL + 0]
                 + cs1 * conv_weight[ch_k * CONV_KERNEL + 1]
                 + cs2 * conv_weight[ch_k * CONV_KERNEL + 2]
                 + cur * conv_weight[ch_k * CONV_KERNEL + 3];
        s_k[tid] = silu(conv_out);
    }
    __syncthreads();

    // === Step 2: L2 normalize Q and K ===
    // Compute ||q||^2 and ||k||^2 using parallel reduction
    float q_sq_sum = 0.0f, k_sq_sum = 0.0f;
    if (tid < K_HEAD_DIM) {
        q_sq_sum = s_q[tid] * s_q[tid];
        k_sq_sum = s_k[tid] * s_k[tid];
    }

    // Reduce q_sq_sum
    s_reduce[tid] = q_sq_sum;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }
    float q_norm_inv = rsqrtf(s_reduce[0] + 1e-6f);
    float q_scale = q_norm_inv / sqrtf((float)K_HEAD_DIM);  // L2norm / sqrt(d_k)

    // Reduce k_sq_sum
    s_reduce[tid] = k_sq_sum;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }
    float k_norm_inv = rsqrtf(s_reduce[0] + 1e-6f);

    // Normalize
    if (tid < K_HEAD_DIM) {
        s_q[tid] *= q_scale;
        s_k[tid] *= k_norm_inv;
    }
    __syncthreads();

    // === Step 3: Compute decay and beta ===
    float decay, beta_val;
    if (tid == 0) {
        float a_val = __half2float(a_proj[head]);
        float b_val = __half2float(b_proj[head]);
        float A = expf(A_log[head]);
        float softplus_a = log1pf(expf(a_val + dt_bias[head]));
        float g = -A * softplus_a;
        g = fmaxf(g, -20.0f);
        decay = expf(g);
        beta_val = sigmoid(b_val);
    }
    // Broadcast decay and beta to all threads
    // Use shared memory
    __shared__ float s_decay, s_beta;
    if (tid == 0) {
        s_decay = decay;
        s_beta = beta_val;
    }
    __syncthreads();
    decay = s_decay;
    beta_val = s_beta;

    // === Step 4: DeltaNet recurrence ===
    // state[head]: [K_HEAD_DIM, V_HEAD_DIM] = [128, 128]
    // Total state size per head: 16K floats = 64KB
    // With 256 threads, each thread handles 128*128/256 = 64 elements

    float* head_state = state + head * K_HEAD_DIM * V_HEAD_DIM;

    // 4a. Decay state: state *= decay
    int elems_per_thread = (K_HEAD_DIM * V_HEAD_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int e = 0; e < elems_per_thread; e++) {
        int idx = tid + e * BLOCK_SIZE;
        if (idx < K_HEAD_DIM * V_HEAD_DIM) {
            head_state[idx] *= decay;
        }
    }
    __syncthreads();

    // 4b. Read memory: kv_mem[v] = sum_k(state[k,v] * k[k])
    // state: [128, 128], k: [128] → kv_mem: [128]
    // Each thread computes one element of kv_mem (if tid < V_HEAD_DIM)
    __shared__ float s_kv_mem[V_HEAD_DIM];
    if (tid < V_HEAD_DIM) {
        float acc = 0.0f;
        for (int k_idx = 0; k_idx < K_HEAD_DIM; k_idx++) {
            acc += head_state[k_idx * V_HEAD_DIM + tid] * s_k[k_idx];
        }
        s_kv_mem[tid] = acc;
    }
    __syncthreads();

    // 4c. Delta: delta[v] = (v[v] - kv_mem[v]) * beta
    __shared__ float s_delta[V_HEAD_DIM];
    if (tid < V_HEAD_DIM) {
        s_delta[tid] = (s_v[tid] - s_kv_mem[tid]) * beta_val;
    }
    __syncthreads();

    // 4d. Write: state[k,v] += k[k] * delta[v]
    for (int e = 0; e < elems_per_thread; e++) {
        int idx = tid + e * BLOCK_SIZE;
        if (idx < K_HEAD_DIM * V_HEAD_DIM) {
            int k_idx = idx / V_HEAD_DIM;
            int v_idx = idx % V_HEAD_DIM;
            head_state[idx] += s_k[k_idx] * s_delta[v_idx];
        }
    }
    __syncthreads();

    // 4e. Read output: out[v] = sum_k(state[k,v] * q[k])
    if (tid < V_HEAD_DIM) {
        float acc = 0.0f;
        for (int k_idx = 0; k_idx < K_HEAD_DIM; k_idx++) {
            acc += head_state[k_idx * V_HEAD_DIM + tid] * s_q[k_idx];
        }
        s_out[tid] = acc;
    }
    __syncthreads();

    // === Step 5: Gated RMSNorm: out = (out / rms) * weight * SiLU(z) ===
    if (tid < V_HEAD_DIM) {
        // Compute RMS
        s_reduce[tid] = s_out[tid] * s_out[tid];
    } else {
        s_reduce[tid] = 0.0f;
    }
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }
    float rms_inv = rsqrtf(s_reduce[0] / V_HEAD_DIM + 1e-6f);

    if (tid < V_HEAD_DIM) {
        float normed = s_out[tid] * rms_inv * norm_weight[tid];
        float z_val = __half2float(z_proj[head * V_HEAD_DIM + tid]);
        float z_silu_val = silu(z_val);
        output[head * V_HEAD_DIM + tid] = __float2half(normed * z_silu_val);
    }
}
