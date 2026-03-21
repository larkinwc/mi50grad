/**
 * Fused gate+up INT4 GEMV for FFN decode (M=1) on gfx906.
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

// Inline dequant+dot for 8 INT4 values against 8 FP16 activations
__device__ __forceinline__ float dequant_dot8(
    unsigned int packed, float cur_scale, float cur_zero,
    const __half* __restrict__ A, unsigned int a_base
) {
    __half2 a01 = *reinterpret_cast<const __half2*>(A + a_base);
    __half2 a23 = *reinterpret_cast<const __half2*>(A + a_base + 2);
    __half2 a45 = *reinterpret_cast<const __half2*>(A + a_base + 4);
    __half2 a67 = *reinterpret_cast<const __half2*>(A + a_base + 6);

    float a0 = __half2float(__low2half(a01));
    float a1 = __half2float(__high2half(a01));
    float a2 = __half2float(__low2half(a23));
    float a3 = __half2float(__high2half(a23));
    float a4 = __half2float(__low2half(a45));
    float a5 = __half2float(__high2half(a45));
    float a6 = __half2float(__low2half(a67));
    float a7 = __half2float(__high2half(a67));

    float zs = cur_zero * cur_scale;
    float a_sum = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;

    float acc = 0.0f;
    acc += (float)(packed        & 0xF) * a0;
    acc += (float)((packed >> 4)  & 0xF) * a1;
    acc += (float)((packed >> 8)  & 0xF) * a2;
    acc += (float)((packed >> 12) & 0xF) * a3;
    acc += (float)((packed >> 16) & 0xF) * a4;
    acc += (float)((packed >> 20) & 0xF) * a5;
    acc += (float)((packed >> 24) & 0xF) * a6;
    acc += (float)(packed >> 28)         * a7;

    return acc * cur_scale - zs * a_sum;
}

// Fused gate+up split-K GEMV.
// Both gate and up weight matrices are [K/8, N] packed INT4.
// C_gate_fp32 and C_up_fp32 must be zeroed before launch.
extern "C" __global__ void gemv_int4_dual_splitk(
    const __half* __restrict__ A,           // input x[K]
    const unsigned int* __restrict__ B_gate, // gate qweight [K/8, N]
    const __half* __restrict__ gate_scales,  // gate scales
    const __half* __restrict__ gate_zeros,   // gate zeros
    const unsigned int* __restrict__ B_up,   // up qweight [K/8, N]
    const __half* __restrict__ up_scales,    // up scales
    const __half* __restrict__ up_zeros,     // up zeros
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
    float gate_s = 0.0f, gate_z = 0.0f;
    float up_s = 0.0f, up_z = 0.0f;
    unsigned int last_sg = 0xFFFFFFFF;

    for (unsigned int kg = kg_start; kg < kg_end; kg++) {
        unsigned int sg = kg / groups_per_scale;
        if (sg != last_sg) {
            gate_s = __half2float(gate_scales[sg * N + col]);
            gate_z = __half2float(gate_zeros[sg * N + col]);
            up_s   = __half2float(up_scales[sg * N + col]);
            up_z   = __half2float(up_zeros[sg * N + col]);
            last_sg = sg;
        }
        unsigned int gate_packed = B_gate[kg * N + col];
        unsigned int up_packed   = B_up[kg * N + col];
        acc_gate += dequant_dot8(gate_packed, gate_s, gate_z, A, kg << 3);
        acc_up   += dequant_dot8(up_packed,   up_s,   up_z,   A, kg << 3);
    }

    atomicAdd(&C_gate_fp32[col], acc_gate);
    atomicAdd(&C_up_fp32[col],   acc_up);
}

// Convert FP32 accumulators to FP16 with fused SiLU: out = silu(gate) * up
extern "C" __global__ void dual_fp32_to_silu_fp16(
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
 * Fully fused gate+up dual split-K INT4 GEMV: single kernel launch, no separate
 * memset or convert+SiLU.
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
 *   gate_zeros   - gate quantization zeros  [num_groups, N] FP16
 *   B_up         - packed up INT4 weights [K/8, N] uint32
 *   up_scales    - up quantization scales [num_groups, N] FP16
 *   up_zeros     - up quantization zeros  [num_groups, N] FP16
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
extern "C" __global__ void gemv_int4_dual_fused(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_gate,
    const __half* __restrict__ gate_scales,
    const __half* __restrict__ gate_zeros,
    const unsigned int* __restrict__ B_up,
    const __half* __restrict__ up_scales,
    const __half* __restrict__ up_zeros,
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

    // 4x register blocking with dual accumulators to break dependency chains
    float acc_gate0 = 0.0f, acc_gate1 = 0.0f;
    float acc_up0 = 0.0f, acc_up1 = 0.0f;
    float gate_s = 0.0f, gate_z = 0.0f;
    float up_s = 0.0f, up_z = 0.0f;
    unsigned int last_sg = 0xFFFFFFFF;

    unsigned int kg = kg_start;
    unsigned int kg_end_4 = kg_start + ((kg_end - kg_start) & ~3u);

    for (; kg < kg_end_4; kg += 4) {
        unsigned int gp0 = B_gate[kg * N + col];
        unsigned int up0 = B_up[kg * N + col];
        unsigned int gp1 = B_gate[(kg+1) * N + col];
        unsigned int up1 = B_up[(kg+1) * N + col];
        unsigned int gp2 = B_gate[(kg+2) * N + col];
        unsigned int up2 = B_up[(kg+2) * N + col];
        unsigned int gp3 = B_gate[(kg+3) * N + col];
        unsigned int up3 = B_up[(kg+3) * N + col];

        unsigned int sg0 = kg / groups_per_scale;
        if (sg0 != last_sg) {
            gate_s = __half2float(gate_scales[sg0 * N + col]);
            gate_z = __half2float(gate_zeros[sg0 * N + col]);
            up_s   = __half2float(up_scales[sg0 * N + col]);
            up_z   = __half2float(up_zeros[sg0 * N + col]);
            last_sg = sg0;
        }
        acc_gate0 += dequant_dot8(gp0, gate_s, gate_z, A, kg << 3);
        acc_up0   += dequant_dot8(up0, up_s,   up_z,   A, kg << 3);

        unsigned int sg1 = (kg+1) / groups_per_scale;
        if (sg1 != last_sg) {
            gate_s = __half2float(gate_scales[sg1 * N + col]);
            gate_z = __half2float(gate_zeros[sg1 * N + col]);
            up_s   = __half2float(up_scales[sg1 * N + col]);
            up_z   = __half2float(up_zeros[sg1 * N + col]);
            last_sg = sg1;
        }
        acc_gate1 += dequant_dot8(gp1, gate_s, gate_z, A, (kg+1) << 3);
        acc_up1   += dequant_dot8(up1, up_s,   up_z,   A, (kg+1) << 3);

        unsigned int sg2 = (kg+2) / groups_per_scale;
        if (sg2 != last_sg) {
            gate_s = __half2float(gate_scales[sg2 * N + col]);
            gate_z = __half2float(gate_zeros[sg2 * N + col]);
            up_s   = __half2float(up_scales[sg2 * N + col]);
            up_z   = __half2float(up_zeros[sg2 * N + col]);
            last_sg = sg2;
        }
        acc_gate0 += dequant_dot8(gp2, gate_s, gate_z, A, (kg+2) << 3);
        acc_up0   += dequant_dot8(up2, up_s,   up_z,   A, (kg+2) << 3);

        unsigned int sg3 = (kg+3) / groups_per_scale;
        if (sg3 != last_sg) {
            gate_s = __half2float(gate_scales[sg3 * N + col]);
            gate_z = __half2float(gate_zeros[sg3 * N + col]);
            up_s   = __half2float(up_scales[sg3 * N + col]);
            up_z   = __half2float(up_zeros[sg3 * N + col]);
            last_sg = sg3;
        }
        acc_gate1 += dequant_dot8(gp3, gate_s, gate_z, A, (kg+3) << 3);
        acc_up1   += dequant_dot8(up3, up_s,   up_z,   A, (kg+3) << 3);
    }

    for (; kg < kg_end; kg++) {
        unsigned int sg = kg / groups_per_scale;
        if (sg != last_sg) {
            gate_s = __half2float(gate_scales[sg * N + col]);
            gate_z = __half2float(gate_zeros[sg * N + col]);
            up_s   = __half2float(up_scales[sg * N + col]);
            up_z   = __half2float(up_zeros[sg * N + col]);
            last_sg = sg;
        }
        acc_gate0 += dequant_dot8(B_gate[kg * N + col], gate_s, gate_z, A, kg << 3);
        acc_up0   += dequant_dot8(B_up[kg * N + col],   up_s,   up_z,   A, kg << 3);
    }

    float acc_gate = acc_gate0 + acc_gate1;
    float acc_up = acc_up0 + acc_up1;

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
