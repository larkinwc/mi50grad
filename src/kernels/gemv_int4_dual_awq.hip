/**
 * Fused gate+up INT4 GEMV for FFN decode (M=1) on gfx906 - AWQ variant.
 *
 * AWQ dequantization: w = q * scale (no zero-point subtraction)
 * GPTQ dequantization: w = (q - zero) * scale
 *
 * This AWQ variant saves:
 * - Zero-point subtraction arithmetic (8 v_sub_f32 per uint32)
 * - No zeros tensor loads (reduced memory traffic)
 * - Same persistent buffer + atomic counter protocol as GPTQ dual
 *
 * Both gate and up projections read the same input x[K].
 * Fusing them into one kernel saves:
 * - 1 kernel launch (splitk)
 * - 1 memset (FP32 accumulator)
 * - 1 fp32_to_fp16 conversion pass
 * - 1 redundant x[K] read from L2/HBM
 *
 * Split-K with atomicAdd to dual FP32 buffers, then convert+SiLU fused.
 *
 * Grid: (ceil(N/256), K_SPLITS, 1), Block: (256, 1, 1)
 * N = local_intermediate_size (same for gate and up)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---------------------------------------------------------------------------
// dequant_dot8_awq: Process one packed uint32 (8 INT4 weights) — AWQ variant.
//
// Extracts 8 nibbles, dequantizes to q * scale (no zero-point),
// accumulates via direct FP32 multiplication.
//
// Saves 8 v_sub_f32 instructions + no zeros load vs the GPTQ version.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dequant_dot8_awq(
    unsigned int packed,
    float cur_scale,
    const __half* __restrict__ A,
    unsigned int a_base
) {
    // Load 8 FP16 activations
    float a0 = __half2float(A[a_base + 0]);
    float a1 = __half2float(A[a_base + 1]);
    float a2 = __half2float(A[a_base + 2]);
    float a3 = __half2float(A[a_base + 3]);
    float a4 = __half2float(A[a_base + 4]);
    float a5 = __half2float(A[a_base + 5]);
    float a6 = __half2float(A[a_base + 6]);
    float a7 = __half2float(A[a_base + 7]);

    // AWQ: w = q * scale (no zero subtraction)
    float w0 = (float)((packed >>  0) & 0xF) * cur_scale;
    float w1 = (float)((packed >>  4) & 0xF) * cur_scale;
    float w2 = (float)((packed >>  8) & 0xF) * cur_scale;
    float w3 = (float)((packed >> 12) & 0xF) * cur_scale;
    float w4 = (float)((packed >> 16) & 0xF) * cur_scale;
    float w5 = (float)((packed >> 20) & 0xF) * cur_scale;
    float w6 = (float)((packed >> 24) & 0xF) * cur_scale;
    float w7 = (float)((packed >> 28) & 0xF) * cur_scale;

    // Direct FP32 accumulation
    return w0 * a0 + w1 * a1 + w2 * a2 + w3 * a3 +
           w4 * a4 + w5 * a5 + w6 * a6 + w7 * a7;
}

// ---------------------------------------------------------------------------
// AWQ dual split-K GEMV (no zeros).
// Both gate and up weight matrices are [K/8, N] packed INT4.
// C_gate_fp32 and C_up_fp32 must be zeroed before launch.
// ---------------------------------------------------------------------------
extern "C" __global__ void gemv_int4_dual_awq_splitk(
    const __half* __restrict__ A,           // input x[K]
    const unsigned int* __restrict__ B_gate, // gate qweight [K/8, N]
    const __half* __restrict__ gate_scales,  // gate scales
    const unsigned int* __restrict__ B_up,   // up qweight [K/8, N]
    const __half* __restrict__ up_scales,    // up scales
    float* __restrict__ C_gate_fp32,         // gate accumulator [N]
    float* __restrict__ C_up_fp32,           // up accumulator [N]
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    unsigned int num_k_groups = K >> 3;
    unsigned int groups_per_scale = group_size >> 3;
    unsigned int k_splits = gridDim.y;
    unsigned int k_split_id = blockIdx.y;
    unsigned int kg_start = (num_k_groups * k_split_id) / k_splits;
    unsigned int kg_end = (num_k_groups * (k_split_id + 1)) / k_splits;

    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    float gate_s = 0.0f;
    float up_s = 0.0f;
    unsigned int last_sg = 0xFFFFFFFF;

    for (unsigned int kg = kg_start; kg < kg_end; kg++) {
        unsigned int sg = kg / groups_per_scale;
        if (sg != last_sg) {
            gate_s = __half2float(gate_scales[sg * N + col]);
            up_s   = __half2float(up_scales[sg * N + col]);
            last_sg = sg;
        }
        unsigned int gate_packed = B_gate[kg * N + col];
        unsigned int up_packed   = B_up[kg * N + col];
        acc_gate += dequant_dot8_awq(gate_packed, gate_s, A, kg << 3);
        acc_up   += dequant_dot8_awq(up_packed,   up_s,   A, kg << 3);
    }

    atomicAdd(&C_gate_fp32[col], acc_gate);
    atomicAdd(&C_up_fp32[col],   acc_up);
}

// ---------------------------------------------------------------------------
// Convert FP32 accumulators to FP16 with fused SiLU: out = silu(gate) * up
// ---------------------------------------------------------------------------
extern "C" __global__ void dual_awq_fp32_to_silu_fp16(
    const float* __restrict__ gate_fp32,
    const float* __restrict__ up_fp32,
    __half* __restrict__ out,
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float g = gate_fp32[idx];
        float u = up_fp32[idx];
        float sig = 1.0f / (1.0f + __expf(-g));
        out[idx] = __float2half(g * sig * u);
    }
}

/**
 * Fully fused gate+up dual split-K INT4 GEMV - AWQ variant: single kernel launch,
 * no separate memset or convert+SiLU.
 *
 * AWQ dequantization: w = q * scale (no zero-point subtraction)
 *
 * Protocol (race-free using atomic counter per column):
 *   - All k_splits tiles atomicAdd their partial FP32 sums into C_gate_fp32[col]
 *     and C_up_fp32[col]. Both buffers are PERSISTENT (zeroed once at engine init).
 *   - Each tile atomicAdd(&done[col], 1). The tile that sees old == k_splits-1 is
 *     the LAST tile for that column — it:
 *       (a) reads C_gate_fp32[col] and C_up_fp32[col] (complete sums)
 *       (b) applies SiLU: out[col] = silu(gate) * up
 *       (c) resets C_gate_fp32[col] = 0, C_up_fp32[col] = 0, done[col] = 0
 *
 * Persistent buffers (C_gate_fp32, C_up_fp32, done) must be zero-initialized
 * ONCE at engine startup. Each call leaves them zeroed for the next call.
 *
 * Parameters:
 *   A            - input activations [K] FP16
 *   B_gate       - packed gate INT4 weights [K/8, N] uint32
 *   gate_scales  - gate quantization scales [num_groups, N] FP16
 *   B_up         - packed up INT4 weights [K/8, N] uint32
 *   up_scales    - up quantization scales [num_groups, N] FP16
 *   C_gate_fp32  - persistent gate FP32 accumulator [N] float (zeroed at init)
 *   C_up_fp32    - persistent up FP32 accumulator [N] float (zeroed at init)
 *   done         - persistent atomic counter [N] uint32 (zeroed at init)
 *   out          - FP16 output [N] half (silu(gate) * up)
 *   K, N, group_size - GEMV dimensions
 *   k_splits     - number of K-dimension splits (== gridDim.y)
 *
 * Grid:  (ceil(N/256), k_splits, 1)
 * Block: (256, 1, 1)
 */
extern "C" __global__ void gemv_int4_dual_awq_fused(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_gate,
    const __half* __restrict__ gate_scales,
    const unsigned int* __restrict__ B_up,
    const __half* __restrict__ up_scales,
    float* __restrict__ C_gate_fp32,
    float* __restrict__ C_up_fp32,
    unsigned int* __restrict__ done,
    __half* __restrict__ out,
    unsigned int K,
    unsigned int N,
    unsigned int group_size,
    unsigned int k_splits
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    unsigned int num_k_groups = K >> 3;
    unsigned int groups_per_scale = group_size >> 3;
    unsigned int k_split_id = blockIdx.y;
    unsigned int kg_start = (num_k_groups * k_split_id) / k_splits;
    unsigned int kg_end = (num_k_groups * (k_split_id + 1)) / k_splits;

    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    float gate_s = 0.0f;
    float up_s = 0.0f;
    unsigned int last_sg = 0xFFFFFFFF;

    for (unsigned int kg = kg_start; kg < kg_end; kg++) {
        unsigned int sg = kg / groups_per_scale;
        if (sg != last_sg) {
            gate_s = __half2float(gate_scales[sg * N + col]);
            up_s   = __half2float(up_scales[sg * N + col]);
            last_sg = sg;
        }
        unsigned int gate_packed = B_gate[kg * N + col];
        unsigned int up_packed   = B_up[kg * N + col];
        acc_gate += dequant_dot8_awq(gate_packed, gate_s, A, kg << 3);
        acc_up   += dequant_dot8_awq(up_packed,   up_s,   A, kg << 3);
    }

    // Accumulate into persistent FP32 buffers
    atomicAdd(&C_gate_fp32[col], acc_gate);
    atomicAdd(&C_up_fp32[col],   acc_up);

    // Fence to ensure our atomicAdds are globally visible before incrementing done
    __threadfence();

    // Increment completion counter. The last tile writes FP16 output and resets.
    unsigned int old_done = atomicAdd(&done[col], 1U);
    if (old_done == k_splits - 1U) {
        // We are the last tile to complete for this column.
        // Both accumulators now hold the complete sum.
        float g = C_gate_fp32[col];
        float u = C_up_fp32[col];

        // Apply SiLU: silu(g) * u
        float sig = 1.0f / (1.0f + __expf(-g));
        out[col] = __float2half(g * sig * u);

        // Reset persistent buffers for the next kernel call
        C_gate_fp32[col] = 0.0f;
        C_up_fp32[col]   = 0.0f;
        done[col]        = 0U;
    }
}
