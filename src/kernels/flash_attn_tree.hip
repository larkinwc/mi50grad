/**
 * Tree Attention Kernel for Sequoia-style Speculative Decoding on gfx906.
 *
 * This kernel implements tree-structured attention for batch verification of
 * draft tokens. Instead of causal masking, each query (tree node) attends to
 * a specific subset of KV positions defined by a tree attention mask.
 *
 * Key differences from standard flash attention:
 * 1. Non-causal masking: Each query has a custom attention pattern
 * 2. Small batch size: M = tree_size (typically 4-16 tokens)
 * 3. Variable KV length: Each query may attend to different KV positions
 * 4. Tree topology: Query i attends to ancestors + self + KV cache
 *
 * Kernel interface:
 *   Grid: (num_heads, tree_size, 1)
 *   Block: (64, 1, 1) - single wavefront per query
 *
 * Each WG (head, query_idx) processes one query head:
 * - Loads Q vector for this query
 * - Iterates over attended KV positions (from mask)
 * - Computes attention scores and accumulates V
 *
 * Memory layout:
 *   Q: [tree_size, num_heads, head_dim]
 *   K: [kv_len, num_kv_heads, head_dim]
 *   V: [kv_len, num_kv_heads, head_dim]
 *   Out: [tree_size, num_heads, head_dim]
 *   mask: [tree_size, kv_len] boolean (flattened to 1D for efficient access)
 *
 * The mask is stored as a bit-packed array for efficiency:
 *   mask_words[i * mask_words_per_row + j / 32] & (1 << (j % 32)) != 0
 *   means query i attends to KV position j.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HEAD_DIM 256
#define DIMS_PER_THREAD 4  // 256 / 64 = 4 dims per thread
#define WARP_SIZE 64

// Fast exp
static __device__ __forceinline__ float fast_expf_approx(float x) {
    // Clamp to safe range
    x = fmaxf(-87.3f, x);
    // Schraudolph's approximation: exp(x) ≈ 2^(x * log2(e))
    union { float f; unsigned int i; } u;
    u.i = (unsigned int)(12102203.1f * x + 1064866805.0f);
    return u.f;
}

// Use __expf for correctness (fast enough for attention)
#define FAST_EXP __expf

/**
 * Tree attention decode kernel.
 *
 * Each WG handles one (head, query) pair and processes all attended KV positions.
 * The mask specifies which KV positions each query attends to.
 *
 * @param Q [tree_size, num_heads, HEAD_DIM] query vectors
 * @param K [kv_len, num_kv_heads, HEAD_DIM] key vectors
 * @param V [kv_len, num_kv_heads, HEAD_DIM] value vectors
 * @param Out [tree_size, num_heads, HEAD_DIM] output
 * @param mask [tree_size, kv_len] boolean mask (row-major, 1 bit per entry)
 * @param kv_len Length of KV sequence
 * @param tree_size Number of queries (tree nodes)
 * @param num_heads Number of attention heads
 * @param num_kv_heads Number of KV heads (for GQA)
 * @param mask_words_per_row Number of 32-bit words per mask row (= ceil(kv_len / 32))
 */
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(64, 64)))
void flash_attn_tree_decode(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ Out,
    const unsigned int* __restrict__ mask,
    unsigned int kv_len,
    unsigned int tree_size,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int mask_words_per_row
) {
    unsigned int head = blockIdx.x;
    unsigned int query_idx = blockIdx.y;
    unsigned int lane = threadIdx.x;  // 0..63

    if (query_idx >= tree_size) return;

    // GQA: map head to kv_head
    unsigned int kv_head = (unsigned int)((unsigned long long)head * num_kv_heads / num_heads);
    unsigned int dim_base = lane * DIMS_PER_THREAD;
    float scale = rsqrtf((float)HEAD_DIM);

    // Load Q vector (4 dimensions per thread)
    size_t q_offset = ((size_t)query_idx * num_heads + head) * HEAD_DIM + dim_base;
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

    // Get mask row for this query
    const unsigned int* mask_row = mask + query_idx * mask_words_per_row;

    // Iterate over all KV positions
    // Only process positions where mask bit is set
    for (unsigned int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
        // Check if this KV position is attended
        unsigned int word_idx = kv_pos / 32;
        unsigned int bit_idx = kv_pos % 32;
        unsigned int mask_word = mask_row[word_idx];
        
        if ((mask_word & (1u << bit_idx)) == 0) {
            // This position is masked out - skip
            continue;
        }

        // Load K vector
        size_t k_off = (size_t)kv_pos * kv_row_stride + kv_head_offset + dim_base;
        float k0 = __half2float(K[k_off + 0]);
        float k1 = __half2float(K[k_off + 1]);
        float k2 = __half2float(K[k_off + 2]);
        float k3 = __half2float(K[k_off + 3]);

        // Dot product: q · k
        float dot = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;

        // Intra-wavefront reduction using shuffle
        dot += __shfl_down(dot, 32);
        dot += __shfl_down(dot, 16);
        dot += __shfl_down(dot, 8);
        dot += __shfl_down(dot, 4);
        dot += __shfl_down(dot, 2);
        dot += __shfl_down(dot, 1);
        float score = __shfl(dot, 0);

        // Online softmax update
        float new_max = fmaxf(running_max, score);
        float correction = FAST_EXP(running_max - new_max);
        float p = FAST_EXP(score - new_max);

        running_sum = running_sum * correction + p;

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
    size_t out_offset = ((size_t)query_idx * num_heads + head) * HEAD_DIM + dim_base;
    Out[out_offset + 0] = __float2half(acc0 * inv_sum);
    Out[out_offset + 1] = __float2half(acc1 * inv_sum);
    Out[out_offset + 2] = __float2half(acc2 * inv_sum);
    Out[out_offset + 3] = __float2half(acc3 * inv_sum);
}


/**
 * Tree attention kernel with optimized sparse iteration.
 *
 * Instead of iterating all KV positions and checking mask, this variant
 * takes a list of attended positions for each query. This is more efficient
 * when the mask is sparse (few attended positions).
 *
 * @param attended_indices Flat array of attended KV indices
 * @param attended_offsets [tree_size + 1] offsets into attended_indices
 *                         Query i attends to indices[offsets[i]:offsets[i+1]]
 *
 * Other parameters same as flash_attn_tree_decode.
 */
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(64, 64)))
void flash_attn_tree_decode_sparse(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ Out,
    const unsigned int* __restrict__ attended_indices,
    const unsigned int* __restrict__ attended_offsets,
    unsigned int kv_len,
    unsigned int tree_size,
    unsigned int num_heads,
    unsigned int num_kv_heads
) {
    unsigned int head = blockIdx.x;
    unsigned int query_idx = blockIdx.y;
    unsigned int lane = threadIdx.x;

    if (query_idx >= tree_size) return;

    unsigned int kv_head = (unsigned int)((unsigned long long)head * num_kv_heads / num_heads);
    unsigned int dim_base = lane * DIMS_PER_THREAD;
    float scale = rsqrtf((float)HEAD_DIM);

    // Load Q
    size_t q_offset = ((size_t)query_idx * num_heads + head) * HEAD_DIM + dim_base;
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

    // Get attended positions for this query
    unsigned int start_offset = attended_offsets[query_idx];
    unsigned int end_offset = attended_offsets[query_idx + 1];
    unsigned int num_attended = end_offset - start_offset;

    // Iterate only over attended positions
    for (unsigned int i = 0; i < num_attended; i++) {
        unsigned int kv_pos = attended_indices[start_offset + i];

        // Load K
        size_t k_off = (size_t)kv_pos * kv_row_stride + kv_head_offset + dim_base;
        float k0 = __half2float(K[k_off + 0]);
        float k1 = __half2float(K[k_off + 1]);
        float k2 = __half2float(K[k_off + 2]);
        float k3 = __half2float(K[k_off + 3]);

        // Dot product
        float dot = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;

        // Reduction
        dot += __shfl_down(dot, 32);
        dot += __shfl_down(dot, 16);
        dot += __shfl_down(dot, 8);
        dot += __shfl_down(dot, 4);
        dot += __shfl_down(dot, 2);
        dot += __shfl_down(dot, 1);
        float score = __shfl(dot, 0);

        // Online softmax
        float new_max = fmaxf(running_max, score);
        float correction = FAST_EXP(running_max - new_max);
        float p = FAST_EXP(score - new_max);

        running_sum = running_sum * correction + p;

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

    // Normalize and store
    float inv_sum = 1.0f / running_sum;
    size_t out_offset = ((size_t)query_idx * num_heads + head) * HEAD_DIM + dim_base;
    Out[out_offset + 0] = __float2half(acc0 * inv_sum);
    Out[out_offset + 1] = __float2half(acc1 * inv_sum);
    Out[out_offset + 2] = __float2half(acc2 * inv_sum);
    Out[out_offset + 3] = __float2half(acc3 * inv_sum);
}


// ============================================================================
// Host helper functions (for Python binding reference)
// ============================================================================
//
// These are comments documenting how to call the kernels from Python/HIP C API.
//
// Example launch configuration for flash_attn_tree_decode:
//
//   dim3 grid(num_heads, tree_size, 1);
//   dim3 block(64, 1, 1);
//
//   hipLaunchKernelGGL(
//       flash_attn_tree_decode,
//       grid, block, 0, stream,
//       Q, K, V, Out,
//       mask,
//       kv_len,
//       tree_size,
//       num_heads,
//       num_kv_heads,
//       mask_words_per_row
//   );
//
// Memory requirements:
//   - Q: tree_size * num_heads * HEAD_DIM * 2 bytes
//   - K: kv_len * num_kv_heads * HEAD_DIM * 2 bytes
//   - V: kv_len * num_kv_heads * HEAD_DIM * 2 bytes
//   - Out: tree_size * num_heads * HEAD_DIM * 2 bytes
//   - mask: tree_size * mask_words_per_row * 4 bytes (packed bits)
//
// where mask_words_per_row = (kv_len + 31) / 32
//
// For the sparse variant:
//   - attended_indices: total_attended_positions * 4 bytes
//   - attended_offsets: (tree_size + 1) * 4 bytes
//
// The sparse variant is faster when attended_positions << tree_size * kv_len.
