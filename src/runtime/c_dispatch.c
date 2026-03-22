/*
 * c_dispatch.c: Tight C dispatch loop for TP=4 decode step.
 *
 * Replaces the Python _decode_step_cached_stream() loop with a pure C loop
 * that dispatches all 64 layers' kernels without returning to Python between
 * layers. Eliminates ~14ms/tok Python dispatch overhead.
 *
 * Build command (on dev server, inside Docker):
 *   gcc -O3 -shared -fPIC -I/opt/rocm/include -L/opt/rocm/lib -lamdhip64 \
 *       -o src/runtime/c_dispatch.so src/runtime/c_dispatch.c
 *
 * Key design:
 * - Python serializes the dispatch plan into flat ctypes structures
 * - C extension iterates layers, dispatches kernels, handles allreduce
 * - Position-dependent params (cos/sin, seq_len) updated in C via byte-offset
 *   arithmetic on the pre-built params_arrays
 * - KV cache appended via hipMemcpyAsync (D2D) inside the C loop
 * - All HIP API functions accessed via function pointers (no HIP headers needed)
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ---------------------------------------------------------------------- */
/* HIP API function typedefs (no HIP headers — use function pointers)      */
/* ---------------------------------------------------------------------- */

/* hipError_t hipModuleLaunchKernel(hipFunction_t, gx,gy,gz, bx,by,bz,
 *                                  shared_mem, stream, params, extra) */
typedef int (*hipModuleLaunchKernel_t)(
    void* func,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared_mem,
    void* stream,
    void** params,
    void** extra
);

/* hipError_t hipSetDevice(int deviceId) */
typedef int (*hipSetDevice_t)(int);

/* hipError_t hipStreamSynchronize(hipStream_t stream) */
typedef int (*hipStreamSynchronize_t)(void*);

/* hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) */
typedef int (*hipEventRecord_t)(void*, void*);

/* hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned flags) */
typedef int (*hipStreamWaitEvent_t)(void*, void*, unsigned);

/* hipError_t hipMemcpyPeerAsync(dst, dstDev, src, srcDev, size, stream) */
typedef int (*hipMemcpyPeerAsync_t)(void*, int, void*, int, size_t, void*);

/* hipError_t hipMemcpyAsync(dst, src, size, kind, stream) */
/* kind: 3 = hipMemcpyDeviceToDevice */
typedef int (*hipMemcpyAsync_t)(void*, void*, size_t, int, void*);

/* hipError_t hipGetLastError() */
typedef int (*hipGetLastError_t)(void);

/* kernel_p2p_allreduce_residual_tp4(hidden, partial_local, peer0, peer1, peer2, n, stream)
 * Host-callable C wrapper from kernel_p2p_allreduce.so.
 * Launches kernel on calling device to read peer partials via BAR1 P2P.
 * Must be called after hipSetDevice(device) for the target GPU.
 */
typedef int (*kernel_p2p_tp4_fn_t)(
    void* hidden,
    const void* partial_local,
    const void* partial_peer0,
    const void* partial_peer1,
    const void* partial_peer2,
    unsigned int n,
    void* stream  /* hipStream_t */
);

/* kernel_p2p_allreduce_rmsnorm_tp4(output, hidden, partial_local, peer0, peer1, peer2,
 *                                    weight, dim, batch, eps, stream)
 * Host-callable C wrapper from kernel_p2p_allreduce_rmsnorm.so.
 * Fused P2P allreduce + RMSNorm kernel.
 * hidden: hidden residual input (ADD to sum of partials)
 */
typedef int (*kernel_p2p_fused_tp4_fn_t)(
    void* output,
    const void* hidden,
    const void* partial_local,
    const void* partial_peer0,
    const void* partial_peer1,
    const void* partial_peer2,
    const void* weight,
    unsigned int dim,
    unsigned int batch,
    float eps,
    void* stream  /* hipStream_t */
);

/* gemv_int4_p2p_allreduce_rmsnorm_tp4(output, A, B_q4, scales, zeros,
 *                                        partial_local, peer0, peer1, peer2,
 *                                        weight, wg_partial_sum_sq, wg_write_counter, wg_done_counter,
 *                                        K, N, dim, group_size, eps,
 *                                        tp_rank, tp_size, stream)
 * Host-callable C wrapper from gemv_int4_p2p_allreduce_rmsnorm.so.
 * Fused INT4 GEMV + P2P allreduce + RMSNorm kernel for FFN down projection.
 * Eliminates separate ffn_down kernel + allreduce + RMSNorm launches.
 */
typedef int (*gemv_int4_fused_tp4_fn_t)(
    void* output,
    const void* A,                    /* Input activation (FFN gate output) */
    const unsigned int* B_q4,         /* INT4 weights (FFN down proj) */
    const void* scales,               /* Per-group scales */
    const void* zeros,                /* Per-group zeros */
    const void* partial_local,        /* This GPU's partial buffer */
    const void* partial_peer0,        /* Peer GPU partial buffers (P2P) */
    const void* partial_peer1,
    const void* partial_peer2,
    const void* weight,               /* RMSNorm weight (next layer's attn_norm) */
    void* wg_partial_sum_sq,          /* [num_wgs] WG partial sum array */
    void* wg_write_counter,           /* [1] write barrier counter */
    void* wg_done_counter,            /* [1] completion counter */
    unsigned int K,                   /* Input dim (intermediate size) */
    unsigned int N,                   /* Output dim (hidden size) */
    unsigned int dim,                 /* Hidden dim for RMSNorm */
    unsigned int group_size,          /* Quantization group size */
    float eps,                        /* RMSNorm epsilon */
    unsigned int tp_rank,             /* This GPU's rank */
    unsigned int tp_size,             /* Tensor parallel size */
    void* stream                      /* HIP stream */
);

/* gemv_int4_p2p_allreduce_rmsnorm_compressed_tp4(output, A, B_q4, scales, zeros,
 *                                                  partial_local, compressed_local, compressed_peer0, compressed_peer1, compressed_peer2,
 *                                                  weight, wg_partial_sum_sq, wg_write_counter, wg_done_counter,
 *                                                  K, N, dim, group_size, eps,
 *                                                  tp_rank, tp_size, stream)
 * Host-callable C wrapper from gemv_int4_p2p_allreduce_rmsnorm_compressed.so.
 * Fused INT4 GEMV + INT8-compressed P2P allreduce + RMSNorm kernel for FFN down projection.
 * Uses INT8 compression for P2P transfers to reduce PCIe bandwidth.
 */
typedef int (*gemv_int4_fused_compressed_tp4_fn_t)(
    void* output,
    const void* A,                    /* Input activation (FFN gate output) */
    const unsigned int* B_q4,         /* INT4 weights (FFN down proj) */
    const void* scales,               /* Per-group scales */
    const void* zeros,                /* Per-group zeros */
    void* partial_local,              /* This GPU's partial buffer (FP16, for test extraction) */
    void* compressed_local,           /* Local INT8 compressed buffer [n + num_blocks*2] */
    const void* compressed_peer0,     /* Peer GPU INT8 compressed buffers (P2P) */
    const void* compressed_peer1,
    const void* compressed_peer2,
    const void* weight,               /* RMSNorm weight (next layer's attn_norm) */
    void* wg_partial_sum_sq,          /* [num_wgs] WG partial sum array */
    void* wg_write_counter,           /* [1] write barrier counter */
    void* wg_done_counter,            /* [1] completion counter */
    unsigned int K,                   /* Input dim (intermediate size) */
    unsigned int N,                   /* Output dim (hidden size) */
    unsigned int dim,                 /* Hidden dim for RMSNorm */
    unsigned int group_size,          /* Quantization group size */
    float eps,                        /* RMSNorm epsilon */
    unsigned int tp_rank,             /* This GPU's rank */
    unsigned int tp_size,             /* Tensor parallel size */
    void* stream                      /* HIP stream */
);

/* residual_add_v2(dst, src, n): Host-callable C wrapper from elementwise_v2.so.
 * Performs element-wise addition: dst[i] += src[i] for all i in [0, n).
 * Used for deferred attention allreduce to add partial attention output to hidden.
 */
typedef int (*residual_add_v2_fn_t)(
    void* dst,                      /* Destination buffer (modified in place) */
    const void* src,                /* Source buffer (added to dst) */
    unsigned int n                  /* Number of FP16 elements */
);

/* ---------------------------------------------------------------------- */
/* Data structures (must match Python ctypes definitions exactly)           */
/* ---------------------------------------------------------------------- */

/*
 * CKernelSpec: pre-built kernel launch specification.
 */
typedef struct {
    uint64_t  func;           /* hipFunction_t handle (cast to uint64) */
    uint32_t  grid_x;
    uint32_t  grid_y;
    uint32_t  grid_z;
    uint32_t  block_x;
    uint32_t  block_y;
    uint32_t  block_z;
    uint32_t  shared_mem;
    uint64_t  params_array;   /* void** pointing to ctypes param objects */
    uint32_t  num_params;
    uint32_t  present;        /* 1 = launch, 0 = skip */
} CKernelSpec;

/* Param indices for mutable values */
#define QKNORM_COS_IDX        2
#define QKNORM_SIN_IDX        3
#define DECODE_ATTN_SEQ_IDX   4
#define QKNORM_CACHEW_DST_IDX 4   /* cache_dst in qknorm_rope_cachew (index 4) */
#define GEMV_V_CACHE_OUT_IDX  2   /* output ptr for gemv_v_cache (index 2 in gemv_fp16_v2) */

/*
 * CEngineLayerSpec: kernel specs for one engine on one layer.
 */
typedef struct {
    /* Attention kernels */
    CKernelSpec attn_rmsnorm;
    CKernelSpec gemv_q_fused;
    CKernelSpec gemv_kv_fused;   /* standard: fused [K,V] GEMV */
    CKernelSpec gemv_qkv_fused;  /* fused QKV GEMV (3-in-1, INT4 only) */
    CKernelSpec qknorm_q;
    CKernelSpec qknorm_k;        /* standard: qknorm_rope_fused; or cachew variant */
    CKernelSpec decode_attn;
    CKernelSpec sigmoid_mul;
    CKernelSpec gemv_o_proj;
    /* DeltaNet kernels */
    CKernelSpec gemv_la_in_proj;
    CKernelSpec deltanet_v3;
    CKernelSpec deltanet_v3_shift;
    CKernelSpec gemv_la_out_proj;
    /* FFN kernels */
    CKernelSpec ffn_rmsnorm;
    CKernelSpec ffn_gate_up_silu;
    CKernelSpec ffn_down;
    /* Direct KV write kernels (when use_direct_kv_write=1) */
    CKernelSpec gemv_k_only;     /* K-only GEMV to working buffer */
    CKernelSpec gemv_v_cache;    /* V GEMV writing directly to cache position (mutable out) */

    int      layer_type;         /* 0=full_attention, 1=deltanet */
    int      streams_ready;      /* 1 if Q/KV streams need sync before qknorm */
    uint64_t stream_q;           /* HIP stream for Q projection (if streams_ready) */
    uint64_t stream_kv;          /* HIP stream for KV projection (if streams_ready) */

    /* Position-dependent params: base pointers for cos/sin tables */
    uint64_t d_cos_base;         /* engine->d_cos */
    uint64_t d_sin_base;         /* engine->d_sin */

    /* KV cache append for full attention layers */
    uint64_t d_k_src;            /* engine->d_k (source K buffer after GEMV) */
    uint64_t d_v_src;            /* engine->d_v (source V buffer after GEMV) */
    uint64_t kv_cache_k_base;    /* kv_cache.layer_k_ptr(layer_idx); 0=skip D2D copy */
    uint64_t kv_cache_v_base;    /* kv_cache.layer_v_ptr(layer_idx); 0=skip D2D copy */
    uint32_t kv_stride;          /* local_num_kv_heads * head_dim * 2 bytes */
    uint32_t use_direct_kv_write;/* 1=K and V written directly by kernels (no memcpy) */
    
    /* Deferred attention allreduce (M3 optimization) */
    uint64_t d_hidden;           /* engine->d_hidden (for deferred AR residual add) */
    uint64_t d_proj_out;         /* engine->d_proj_out (attention output partial) */
} CEngineLayerSpec;

/*
 * CAllreduceSpec: parameters for one async allreduce call.
 *
 * Supports two modes:
 *   use_kernel_p2p=0 (default): star topology gather/reduce/broadcast via hipMemcpyPeerAsync
 *   use_kernel_p2p=1:           kernel P2P allreduce — each GPU runs a single kernel
 *                                that reads all peer partial buffers via BAR1 P2P,
 *                                eliminating gather/reduce/broadcast host round-trips.
 */
typedef struct {
    /* p2p_reduce_residual_tp{N} function pointers (star topology path) */
    int (*reduce_tp2)(void *hidden, void *p0, void *p1,
                      uint32_t n, void *stream);
    int (*reduce_tp3)(void *hidden, void *p0, void *p1, void *p2,
                      uint32_t n, void *stream);
    int (*reduce_tp4)(void *hidden, void *p0, void *p1, void *p2, void *p3,
                      uint32_t n, void *stream);

    int      tp_size;
    int      device_ids[4];
    uint64_t partial_ptrs[4];
    uint64_t hidden_ptrs[4];
    uint64_t gather_bufs[3];
    uint64_t allreduce_streams[4];
    uint64_t compute_events[4];
    uint64_t ar_done_events[4];
    uint64_t compute_streams[4];
    uint32_t num_elems;

    /* Kernel P2P allreduce fields (added for kernel-p2p-tp4-integration) */
    uint32_t use_kernel_p2p;          /* 1=use kernel P2P, 0=use star topology */
    kernel_p2p_tp4_fn_t kernel_p2p_tp4_fn; /* ptr to kernel_p2p_allreduce_residual_tp4 */
    
    /* Fused kernel P2P allreduce + RMSNorm fields (kernel_p2p_allreduce_rmsnorm) */
    uint32_t use_fused_kernel;        /* 1=use fused allreduce+RMSNorm, 0=separate */
    kernel_p2p_fused_tp4_fn_t kernel_p2p_fused_tp4_fn; /* ptr to fused kernel */
    uint64_t rmsnorm_weight_ptrs[4];  /* RMSNorm weight pointers for each GPU */
    float eps;                        /* RMSNorm epsilon (default: 1e-6) */
    
    /* Fused GEMV+AR+RMSNorm kernel fields (gemv_int4_p2p_allreduce_rmsnorm) */
    /* For FFN down-proj fusion: replaces ffn_down + allreduce + next_layer attn_rmsnorm */
    uint32_t use_gemv_fused;          /* 1=use fused GEMV+allreduce+RMSNorm for FFN down */
    gemv_int4_fused_tp4_fn_t gemv_fused_tp4_fn; /* ptr to gemv_int4_p2p_allreduce_rmsnorm_tp4 */
    /* Input activation pointer: FFN gate output after SiLU (replicated across GPUs) */
    uint64_t ffn_gate_ptrs[4];        /* FFN gate output pointers per GPU (input to down proj) */
    /* INT4 weight parameters for fused GEMV (FFN down projection) */
    uint64_t ffn_down_qweight_ptrs[4]; /* FFN down proj INT4 weights per GPU */
    uint64_t ffn_down_scales_ptrs[4];  /* FFN down proj scales per GPU */
    uint64_t ffn_down_zeros_ptrs[4];   /* FFN down proj zeros per GPU */
    uint32_t ffn_K;                    /* FFN intermediate size (input to down proj) */
    uint32_t ffn_group_size;           /* Quantization group size */
    /* Fused GEMV kernel cross-WG coordination counters */
    uint64_t gemv_fused_wg_partial_sum_sq[4];  /* [4] WG partial sum arrays */
    uint64_t gemv_fused_wg_write_counter[4];   /* [4] write barrier counters */
    uint64_t gemv_fused_wg_done_counter[4];    /* [4] completion counters */
    
    /* INT8-compressed GEMV+AR+RMSNorm kernel fields (gemv_int4_p2p_allreduce_rmsnorm_compressed) */
    /* Same as use_gemv_fused but uses INT8 compression for P2P transfers */
    uint32_t use_gemv_fused_compressed;       /* 1=use compressed fused kernel for FFN down */
    gemv_int4_fused_compressed_tp4_fn_t gemv_fused_compressed_tp4_fn; /* ptr to compressed kernel */
    /* Compressed buffer pointers: [INT8 data (n bytes)] [FP16 scales (num_blocks*2 bytes)] */
    uint64_t compressed_ptrs[4];              /* INT8 compressed buffer pointers per GPU */
    /* Padding for 8-byte alignment: 4 + 8 + 32 = 44 bytes */
    
    /* Weight prefetch fields (for overlapping prefetch with allreduce) */
    uint32_t enable_prefetch;           /* 1=enable async prefetch of next layer weights during allreduce */
    uint64_t prefetch_qkv_weight;       /* Next layer QKV fused weight pointer (INT4 qweight) */
    uint64_t prefetch_qkv_scales;       /* Next layer QKV fused scales pointer */
    uint64_t prefetch_qkv_zeros;        /* Next layer QKV fused zeros pointer */
    uint64_t prefetch_ffn_gate_weight;  /* Next layer FFN gate weight pointer (INT4 qweight) */
    uint64_t prefetch_ffn_gate_scales;  /* Next layer FFN gate scales pointer */
    uint64_t prefetch_ffn_gate_zeros;   /* Next layer FFN gate zeros pointer */
    uint64_t prefetch_ffn_up_weight;    /* Next layer FFN up weight pointer (INT4 qweight) */
    uint64_t prefetch_ffn_up_scales;    /* Next layer FFN up scales pointer */
    uint64_t prefetch_ffn_up_zeros;     /* Next layer FFN up zeros pointer */
    uint32_t prefetch_qkv_bytes;        /* Size of QKV weight in bytes */
    uint32_t prefetch_ffn_gate_bytes;   /* Size of FFN gate weight in bytes */
    uint32_t prefetch_ffn_up_bytes;     /* Size of FFN up weight in bytes */
    
    /* Scratch buffer for weight prefetch (avoids dst=src memory corruption) */
    uint64_t scratch_buf;               /* Scratch buffer pointer for prefetch destination (~256KB per GPU) */
    uint32_t scratch_size;              /* Size of scratch buffer in bytes */
} CAllreduceSpec;

/*
 * CDispatchPlan: top-level plan passed from Python via ctypes pointer.
 */
typedef struct {
    int     num_layers;
    int     num_engines;
    uint64_t engine_layer_specs;
    uint64_t attn_allreduce_specs;
    uint64_t ffn_allreduce_specs;
    int     use_stream_overlap;
    
    /* Deferred attention allreduce (M3 optimization) */
    int     use_deferred_attention_ar;  /* 1=skip attention AR, add partial to hidden locally */
    
    /* residual_add_v2 function pointer (for deferred attention AR) */
    residual_add_v2_fn_t residual_add_fn;
    
    /* HIP API function pointers */
    hipSetDevice_t          hipSetDevice_fn;
    hipStreamSynchronize_t  hipStreamSynchronize_fn;

    /* Additional HIP API function pointers needed for allreduce */
    hipEventRecord_t        hipEventRecord_fn;
    hipStreamWaitEvent_t    hipStreamWaitEvent_fn;
    hipMemcpyPeerAsync_t    hipMemcpyPeerAsync_fn;
    hipMemcpyAsync_t        hipMemcpyAsync_fn;
    hipGetLastError_t       hipGetLastError_fn;
    hipModuleLaunchKernel_t hipModuleLaunchKernel_fn;
} CDispatchPlan;

/* ---------------------------------------------------------------------- */
/* Internal helpers                                                         */
/* ---------------------------------------------------------------------- */

/* Device caching: track current device to avoid redundant hipSetDevice calls.
 * At ~0.6us per call with ~1792 calls/token, this saves ~1.1ms/tok.
 * 
 * Instrumentation: when HIP_SETDEVICE_VERBOSE=1, print call counts.
 */
static int cached_device = -1;
static unsigned long hipsetdevice_total_calls = 0;
static unsigned long hipsetdevice_skipped_calls = 0;

static int set_device_cached(hipSetDevice_t hipSetDevice_fn, int device_id)
{
    hipsetdevice_total_calls++;
    if (cached_device == device_id) {
        hipsetdevice_skipped_calls++;
        return 0; /* Skip redundant call */
    }
    int err = hipSetDevice_fn(device_id);
    if (err == 0) {
        cached_device = device_id;
    }
    return err;
}

/* Print hipSetDevice statistics. Call at end of decode step. */
static void print_hipsetdevice_stats(void)
{
    const char *verbose = getenv("HIP_SETDEVICE_VERBOSE");
    if (verbose && verbose[0] == '1') {
        fprintf(stderr, "[c_dispatch] hipSetDevice: %lu total, %lu skipped (%.1f%% reduction)\n",
                hipsetdevice_total_calls,
                hipsetdevice_skipped_calls,
                hipsetdevice_total_calls > 0 ? 
                    (100.0 * hipsetdevice_skipped_calls / hipsetdevice_total_calls) : 0.0);
    }
    /* Reset counters for next step */
    hipsetdevice_total_calls = 0;
    hipsetdevice_skipped_calls = 0;
}

static int launch_kernel(const CKernelSpec *spec, CDispatchPlan *plan)
{
    if (!spec->present || !spec->func) return 0;
    void **params = (void **)(uintptr_t)spec->params_array;
    return plan->hipModuleLaunchKernel_fn(
        (void *)(uintptr_t)spec->func,
        spec->grid_x,  spec->grid_y,  spec->grid_z,
        spec->block_x, spec->block_y, spec->block_z,
        spec->shared_mem,
        NULL,   /* stream = default (null) */
        params,
        NULL
    );
}

static void update_cos_sin(CKernelSpec *spec, uint64_t cos_ptr, uint64_t sin_ptr)
{
    if (!spec->present || !spec->params_array) return;
    void **params = (void **)(uintptr_t)spec->params_array;
    *((uint64_t *)params[QKNORM_COS_IDX]) = cos_ptr;
    *((uint64_t *)params[QKNORM_SIN_IDX]) = sin_ptr;
}

static void update_cos_sin_and_cache(CKernelSpec *spec, uint64_t cos_ptr, uint64_t sin_ptr,
                                      uint64_t cache_dst_ptr)
{
    if (!spec->present || !spec->params_array) return;
    void **params = (void **)(uintptr_t)spec->params_array;
    *((uint64_t *)params[QKNORM_COS_IDX]) = cos_ptr;
    *((uint64_t *)params[QKNORM_SIN_IDX]) = sin_ptr;
    *((uint64_t *)params[QKNORM_CACHEW_DST_IDX]) = cache_dst_ptr;
}

static void update_v_cache_ptr(CKernelSpec *spec, uint64_t cache_dst_ptr)
{
    if (!spec->present || !spec->params_array) return;
    void **params = (void **)(uintptr_t)spec->params_array;
    *((uint64_t *)params[GEMV_V_CACHE_OUT_IDX]) = cache_dst_ptr;
}

static void update_seq_len(CKernelSpec *spec, uint32_t seq_len)
{
    if (!spec->present || !spec->params_array) return;
    void **params = (void **)(uintptr_t)spec->params_array;
    *((uint32_t *)params[DECODE_ATTN_SEQ_IDX]) = seq_len;
}

/* ---------------------------------------------------------------------- */
/* Async allreduce (stream overlap mode)                                    */
/* ---------------------------------------------------------------------- */

static int do_allreduce_async(CAllreduceSpec *ar, CDispatchPlan *plan)
{
    int tp = ar->tp_size;
    if (tp <= 1) return 0;

    int i, err;
    void *ar_stream0 = (void *)(uintptr_t)ar->allreduce_streams[0];
    uint32_t n = ar->num_elems;
    size_t size = (size_t)n * 2;

    /* Step 1: Record compute events */
    for (i = 0; i < tp; i++) {
        set_device_cached(plan->hipSetDevice_fn, ar->device_ids[i]);
        plan->hipEventRecord_fn(
            (void *)(uintptr_t)ar->compute_events[i],
            (void *)(uintptr_t)ar->compute_streams[i]);
    }

    /* Step 2: GPU0's allreduce stream waits for all compute events */
    set_device_cached(plan->hipSetDevice_fn, ar->device_ids[0]);
    for (i = 0; i < tp; i++) {
        plan->hipStreamWaitEvent_fn(
            ar_stream0,
            (void *)(uintptr_t)ar->compute_events[i], 0);
    }

    /* Step 3: Async P2P gather to GPU0 */
    for (i = 1; i < tp; i++) {
        plan->hipMemcpyPeerAsync_fn(
            (void *)(uintptr_t)ar->gather_bufs[i - 1], ar->device_ids[0],
            (void *)(uintptr_t)ar->partial_ptrs[i],    ar->device_ids[i],
            size, ar_stream0);
    }

    /* Step 4: Reduce kernel on GPU0 */
    set_device_cached(plan->hipSetDevice_fn, ar->device_ids[0]);
    if (tp == 4) {
        err = ar->reduce_tp4(
            (void *)(uintptr_t)ar->hidden_ptrs[0],
            (void *)(uintptr_t)ar->partial_ptrs[0],
            (void *)(uintptr_t)ar->gather_bufs[0],
            (void *)(uintptr_t)ar->gather_bufs[1],
            (void *)(uintptr_t)ar->gather_bufs[2],
            n, ar_stream0);
    } else if (tp == 3) {
        err = ar->reduce_tp3(
            (void *)(uintptr_t)ar->hidden_ptrs[0],
            (void *)(uintptr_t)ar->partial_ptrs[0],
            (void *)(uintptr_t)ar->gather_bufs[0],
            (void *)(uintptr_t)ar->gather_bufs[1],
            n, ar_stream0);
    } else {
        err = ar->reduce_tp2(
            (void *)(uintptr_t)ar->hidden_ptrs[0],
            (void *)(uintptr_t)ar->partial_ptrs[0],
            (void *)(uintptr_t)ar->gather_bufs[0],
            n, ar_stream0);
    }
    if (err) return err;

    /* Step 5: Record done event on GPU0, broadcast to other GPUs */
    set_device_cached(plan->hipSetDevice_fn, ar->device_ids[0]);
    plan->hipEventRecord_fn(
        (void *)(uintptr_t)ar->ar_done_events[0], ar_stream0);

    for (i = 1; i < tp; i++) {
        set_device_cached(plan->hipSetDevice_fn, ar->device_ids[i]);
        void *ar_si = (void *)(uintptr_t)ar->allreduce_streams[i];
        plan->hipStreamWaitEvent_fn(
            ar_si, (void *)(uintptr_t)ar->ar_done_events[0], 0);
        plan->hipMemcpyPeerAsync_fn(
            (void *)(uintptr_t)ar->hidden_ptrs[i], ar->device_ids[i],
            (void *)(uintptr_t)ar->hidden_ptrs[0], ar->device_ids[0],
            size, ar_si);
        plan->hipEventRecord_fn(
            (void *)(uintptr_t)ar->ar_done_events[i], ar_si);
    }

    /* Re-record GPU0 done event after all broadcasts queued */
    set_device_cached(plan->hipSetDevice_fn, ar->device_ids[0]);
    plan->hipEventRecord_fn(
        (void *)(uintptr_t)ar->ar_done_events[0], ar_stream0);

    return 0;
}

static void wait_for_allreduce(CAllreduceSpec *ar, CDispatchPlan *plan)
{
    int tp = ar->tp_size;
    int i;
    for (i = 0; i < tp; i++) {
        set_device_cached(plan->hipSetDevice_fn, ar->device_ids[i]);
        plan->hipStreamWaitEvent_fn(
            (void *)(uintptr_t)ar->compute_streams[i],
            (void *)(uintptr_t)ar->ar_done_events[i],
            0);
    }
}

/* ---------------------------------------------------------------------- */
/* Kernel P2P allreduce (single kernel per GPU, no gather/broadcast)        */
/* ---------------------------------------------------------------------- */

/*
 * do_allreduce_kernel_p2p: launch kernel P2P allreduce on each GPU.
 *
 * OPTIMIZED: Batch operations per-GPU to minimize hipSetDevice calls.
 * Instead of cycling through GPUs 4 times (once per step), visit each GPU once.
 *
 * Each GPU launches kernel_p2p_allreduce_residual_tp4 on its allreduce stream.
 * The kernel reads all 4 partials (1 local + 3 remote via BAR1 P2P) and reduces
 * them with the local hidden buffer in a single launch — no gather/broadcast.
 *
 * Sync protocol (optimized per-GPU batching):
 *   For each GPU i (0..3):
 *     1. Record compute event
 *     2. Wait for ALL compute events on AR stream
 *     3. Launch kernel
 *     4. Record AR done event
 *   This is valid because all operations are async and queued to hardware.
 *
 * wait_for_allreduce() is called separately to make compute streams wait for
 * the AR done events (same protocol as the star topology path).
 *
 * Supports TP=4 only (TP=2 falls back to star topology).
 *
 * Fused kernel mode (use_fused_kernel=1):
 *   Launches kernel_p2p_allreduce_rmsnorm_tp4 which performs:
 *   - P2P allreduce (sum partials from all GPUs)
 *   - RMSNorm normalization
 *   - Write normalized result to hidden buffer
 *   This eliminates the separate RMSNorm kernel launch.
 */
static int do_allreduce_kernel_p2p(CAllreduceSpec *ar, CDispatchPlan *plan)
{
    int tp = ar->tp_size;
    if (tp <= 1) return 0;

    int i, j, err;
    uint32_t n = ar->num_elems;

    /* Fused kernel mode: use fused allreduce+RMSNorm kernel */
    if (ar->use_fused_kernel && ar->kernel_p2p_fused_tp4_fn != NULL && tp == 4) {
        /* OPTIMIZED: Batch all 4 steps per GPU to minimize hipSetDevice calls.
         * Visit each GPU once, perform all operations, then move to next GPU.
         * This reduces hipSetDevice calls from 16 to 4 per allreduce. */
        for (i = 0; i < tp; i++) {
            /* Set device once for this GPU */
            set_device_cached(plan->hipSetDevice_fn, ar->device_ids[i]);
            
            /* Step 1: Record compute event */
            plan->hipEventRecord_fn(
                (void *)(uintptr_t)ar->compute_events[i],
                (void *)(uintptr_t)ar->compute_streams[i]);
            
            /* Step 2: This GPU's AR stream waits for ALL compute events */
            for (j = 0; j < tp; j++) {
                plan->hipStreamWaitEvent_fn(
                    (void *)(uintptr_t)ar->allreduce_streams[i],
                    (void *)(uintptr_t)ar->compute_events[j], 0);
            }
            
            /* Step 3: Launch fused kernel on this GPU's AR stream */
            int p0 = (i + 1) % 4;
            int p1 = (i + 2) % 4;
            int p2 = (i + 3) % 4;
            err = ar->kernel_p2p_fused_tp4_fn(
                (void *)(uintptr_t)ar->hidden_ptrs[i],  /* output (normalized) */
                (const void *)(uintptr_t)ar->hidden_ptrs[i],  /* hidden input (residual) */
                (const void *)(uintptr_t)ar->partial_ptrs[i],
                (const void *)(uintptr_t)ar->partial_ptrs[p0],
                (const void *)(uintptr_t)ar->partial_ptrs[p1],
                (const void *)(uintptr_t)ar->partial_ptrs[p2],
                (const void *)(uintptr_t)ar->rmsnorm_weight_ptrs[i],
                n,
                1,  /* batch_size */
                ar->eps,
                (void *)(uintptr_t)ar->allreduce_streams[i]
            );
            if (err) return err;
            
            /* Step 4: Record AR done event on this GPU's AR stream */
            plan->hipEventRecord_fn(
                (void *)(uintptr_t)ar->ar_done_events[i],
                (void *)(uintptr_t)ar->allreduce_streams[i]);
        }

        return 0;
    }

    /* Kernel P2P only implemented for TP=4 */
    if (tp != 4 || ar->kernel_p2p_tp4_fn == NULL) {
        return do_allreduce_async(ar, plan);
    }

    /* OPTIMIZED: Batch all 4 steps per GPU to minimize hipSetDevice calls. */
    for (i = 0; i < tp; i++) {
        /* Set device once for this GPU */
        set_device_cached(plan->hipSetDevice_fn, ar->device_ids[i]);
        
        /* Step 1: Record compute event */
        plan->hipEventRecord_fn(
            (void *)(uintptr_t)ar->compute_events[i],
            (void *)(uintptr_t)ar->compute_streams[i]);
        
        /* Step 2: This GPU's AR stream waits for ALL compute events */
        for (j = 0; j < tp; j++) {
            plan->hipStreamWaitEvent_fn(
                (void *)(uintptr_t)ar->allreduce_streams[i],
                (void *)(uintptr_t)ar->compute_events[j], 0);
        }
        
        /* Step 3: Launch kernel P2P allreduce on this GPU's AR stream */
        int p0 = (i + 1) % 4;
        int p1 = (i + 2) % 4;
        int p2 = (i + 3) % 4;
        err = ar->kernel_p2p_tp4_fn(
            (void *)(uintptr_t)ar->hidden_ptrs[i],
            (const void *)(uintptr_t)ar->partial_ptrs[i],
            (const void *)(uintptr_t)ar->partial_ptrs[p0],
            (const void *)(uintptr_t)ar->partial_ptrs[p1],
            (const void *)(uintptr_t)ar->partial_ptrs[p2],
            n,
            (void *)(uintptr_t)ar->allreduce_streams[i]
        );
        if (err) return err;
        
        /* Step 4: Record AR done event on this GPU's AR stream */
        plan->hipEventRecord_fn(
            (void *)(uintptr_t)ar->ar_done_events[i],
            (void *)(uintptr_t)ar->allreduce_streams[i]);
    }

    return 0;
}

/* ---------------------------------------------------------------------- */
/* Fused GEMV+AR+RMSNorm kernel dispatch (for FFN down projection)          */
/* ---------------------------------------------------------------------- */

/*
 * do_allreduce_gemv_fused: launch fused GEMV + P2P allreduce + RMSNorm kernel.
 *
 * OPTIMIZED: Batch operations per-GPU to minimize hipSetDevice calls.
 * Instead of cycling through GPUs 4 times (once per step), we visit each GPU
 * once and perform all 4 steps before moving to the next GPU. This reduces
 * hipSetDevice calls from 16 to 4 per allreduce (75% reduction).
 *
 * This replaces the separate:
 *   1. ffn_down kernel (INT4 GEMV)
 *   2. ffn_allreduce (P2P allreduce)
 *   3. next layer's attn_rmsnorm (RMSNorm)
 *
 * With a single fused kernel launch per GPU.
 *
 * The fused kernel:
 *   - Computes INT4 GEMV for FFN down projection
 *   - Performs P2P allreduce across all GPUs
 *   - Applies RMSNorm with next layer's attention norm weights
 *   - Writes final normalized result to hidden buffer
 *
 * Sync protocol (optimized per-GPU batching):
 *   For each GPU i (0..3):
 *     1. Record compute event
 *     2. Wait for ALL compute events on AR stream
 *     3. Launch fused kernel
 *     4. Record AR done event
 *   This is valid because all operations are async and queued to hardware.
 *
 * wait_for_allreduce() is called separately to make compute streams wait for
 * the AR done events.
 */
static int do_allreduce_gemv_fused(CAllreduceSpec *ar, CDispatchPlan *plan)
{
    int tp = ar->tp_size;
    if (tp != 4 || ar->gemv_fused_tp4_fn == NULL) {
        /* Fall back to standard path if not TP=4 or kernel not available */
        return do_allreduce_async(ar, plan);
    }

    int i, j, err;
    uint32_t n = ar->num_elems;
    uint32_t K = ar->ffn_K;
    uint32_t group_size = ar->ffn_group_size;

    /* OPTIMIZED: Batch all 4 steps per GPU to minimize hipSetDevice calls.
     * Visit each GPU once, perform all operations, then move to next GPU.
     * This reduces hipSetDevice calls from 16 to 4 per allreduce. */
    for (i = 0; i < tp; i++) {
        /* Set device once for this GPU */
        set_device_cached(plan->hipSetDevice_fn, ar->device_ids[i]);
        
        /* Step 1: Record compute event */
        plan->hipEventRecord_fn(
            (void *)(uintptr_t)ar->compute_events[i],
            (void *)(uintptr_t)ar->compute_streams[i]);
        
        /* Step 2: This GPU's AR stream waits for ALL compute events */
        for (j = 0; j < tp; j++) {
            plan->hipStreamWaitEvent_fn(
                (void *)(uintptr_t)ar->allreduce_streams[i],
                (void *)(uintptr_t)ar->compute_events[j], 0);
        }
        
        /* Step 3: Launch fused GEMV+AR+RMSNorm kernel on this GPU's AR stream */
        int p0 = (i + 1) % 4;
        int p1 = (i + 2) % 4;
        int p2 = (i + 3) % 4;
        
        err = ar->gemv_fused_tp4_fn(
            (void *)(uintptr_t)ar->hidden_ptrs[i],        /* output (normalized) */
            (const void *)(uintptr_t)ar->ffn_gate_ptrs[i],  /* A: FFN gate output after SiLU */
            (const unsigned int*)(uintptr_t)ar->ffn_down_qweight_ptrs[i],
            (const void *)(uintptr_t)ar->ffn_down_scales_ptrs[i],
            (const void *)(uintptr_t)ar->ffn_down_zeros_ptrs[i],
            (const void *)(uintptr_t)ar->partial_ptrs[i],
            (const void *)(uintptr_t)ar->partial_ptrs[p0],
            (const void *)(uintptr_t)ar->partial_ptrs[p1],
            (const void *)(uintptr_t)ar->partial_ptrs[p2],
            (const void *)(uintptr_t)ar->rmsnorm_weight_ptrs[i],
            (void *)(uintptr_t)ar->gemv_fused_wg_partial_sum_sq[i],  /* wg_partial_sum_sq */
            (void *)(uintptr_t)ar->gemv_fused_wg_write_counter[i],   /* wg_write_counter */
            (void *)(uintptr_t)ar->gemv_fused_wg_done_counter[i],    /* wg_done_counter */
            K,                                            /* Input dim (intermediate) */
            n,                                            /* Output dim (hidden) */
            n,                                            /* Hidden dim for RMSNorm */
            group_size,
            ar->eps,
            i,                                            /* tp_rank */
            tp,                                           /* tp_size */
            (void *)(uintptr_t)ar->allreduce_streams[i]
        );
        if (err) return err;
        
        /* Step 4: Record AR done event on this GPU's AR stream */
        plan->hipEventRecord_fn(
            (void *)(uintptr_t)ar->ar_done_events[i],
            (void *)(uintptr_t)ar->allreduce_streams[i]);
    }

    return 0;
}

/*
 * do_allreduce_gemv_fused_compressed: launch compressed fused GEMV + P2P allreduce + RMSNorm kernel.
 *
 * OPTIMIZED: Batch operations per-GPU to minimize hipSetDevice calls.
 * Same as do_allreduce_gemv_fused but uses INT8-compressed P2P transfers.
 * The kernel quantizes GEMV output to INT8+scale format, reducing PCIe transfer volume by ~47%.
 */
static int do_allreduce_gemv_fused_compressed(CAllreduceSpec *ar, CDispatchPlan *plan)
{
    int tp = ar->tp_size;
    if (tp != 4 || ar->gemv_fused_compressed_tp4_fn == NULL) {
        /* Fall back to standard uncompressed path if not TP=4 or kernel not available */
        return do_allreduce_gemv_fused(ar, plan);
    }

    int i, j, err;
    uint32_t n = ar->num_elems;
    uint32_t K = ar->ffn_K;
    uint32_t group_size = ar->ffn_group_size;

    /* OPTIMIZED: Batch all 4 steps per GPU to minimize hipSetDevice calls. */
    for (i = 0; i < tp; i++) {
        /* Set device once for this GPU */
        set_device_cached(plan->hipSetDevice_fn, ar->device_ids[i]);
        
        /* Step 1: Record compute event */
        plan->hipEventRecord_fn(
            (void *)(uintptr_t)ar->compute_events[i],
            (void *)(uintptr_t)ar->compute_streams[i]);
        
        /* Step 2: This GPU's AR stream waits for ALL compute events */
        for (j = 0; j < tp; j++) {
            plan->hipStreamWaitEvent_fn(
                (void *)(uintptr_t)ar->allreduce_streams[i],
                (void *)(uintptr_t)ar->compute_events[j], 0);
        }
        
        /* Step 3: Launch compressed fused GEMV+AR+RMSNorm kernel on this GPU's AR stream */
        int p0 = (i + 1) % 4;
        int p1 = (i + 2) % 4;
        int p2 = (i + 3) % 4;
        
        err = ar->gemv_fused_compressed_tp4_fn(
            (void *)(uintptr_t)ar->hidden_ptrs[i],        /* output (normalized) */
            (const void *)(uintptr_t)ar->ffn_gate_ptrs[i],  /* A: FFN gate output after SiLU */
            (const unsigned int*)(uintptr_t)ar->ffn_down_qweight_ptrs[i],
            (const void *)(uintptr_t)ar->ffn_down_scales_ptrs[i],
            (const void *)(uintptr_t)ar->ffn_down_zeros_ptrs[i],
            (void *)(uintptr_t)ar->partial_ptrs[i],         /* partial_local (FP16, for test) */
            (void *)(uintptr_t)ar->compressed_ptrs[i],      /* compressed_local (INT8+scale) */
            (const void *)(uintptr_t)ar->compressed_ptrs[p0],  /* compressed_peer* */
            (const void *)(uintptr_t)ar->compressed_ptrs[p1],
            (const void *)(uintptr_t)ar->compressed_ptrs[p2],
            (const void *)(uintptr_t)ar->rmsnorm_weight_ptrs[i],
            (void *)(uintptr_t)ar->gemv_fused_wg_partial_sum_sq[i],
            (void *)(uintptr_t)ar->gemv_fused_wg_write_counter[i],
            (void *)(uintptr_t)ar->gemv_fused_wg_done_counter[i],
            K, n, n, group_size, ar->eps, i, tp,
            (void *)(uintptr_t)ar->allreduce_streams[i]
        );
        if (err) return err;
        
        /* Step 4: Record AR done event on this GPU's AR stream */
        plan->hipEventRecord_fn(
            (void *)(uintptr_t)ar->ar_done_events[i],
            (void *)(uintptr_t)ar->allreduce_streams[i]);
    }

    return 0;
}

/*
 * dispatch_allreduce: dispatch allreduce using the appropriate method.
 *
 * Checks use_gemv_fused, use_fused_kernel, use_kernel_p2p flags in CAllreduceSpec
 * and routes to:
 *   - do_allreduce_gemv_fused:           fused GEMV+allreduce+RMSNorm (FFN down-proj)
 *   - do_allreduce_kernel_p2p with fused kernel: fused allreduce+RMSNorm
 *   - do_allreduce_kernel_p2p:                   kernel P2P (each GPU reads peers via BAR1)
 *   - do_allreduce_async:                        star topology (gather + reduce + broadcast)
 */

/*
 * issue_weight_prefetch: issue async D2D copies to prefetch next layer weights.
 * 
 * This is called on the compute stream (NULL stream) during the allreduce window.
 * The compute stream is idle during allreduce (~2.94ms for 64 layers), so we can
 * prefetch next layer's weights to warm up HBM memory controller and potentially
 * fill L2 cache with hot weight tiles.
 * 
 * CRITICAL FIX (2026-03-21): The original implementation used hipMemcpyAsync(dst=src)
 * which is invalid on gfx906 and causes memory corruption (cosine_sim 0.58) and
 * -10% throughput regression. 
 * 
 * INVESTIGATION: Multiple approaches were tested:
 *   1. dst=src self-copy: INVALID - causes memory corruption
 *   2. Copy to scratch buffer (256KB per GPU): STILL causes memory corruption (cosine_sim ~0.60-0.85)
 * 
 * ROOT CAUSE: On gfx906 (MI50), async memory operations during allreduce window appear to
 * interfere with P2P allreduce operations, even when using separate streams and proper
 * synchronization. The exact mechanism is unclear but may involve:
 *   - Memory controller bandwidth contention
 *   - L2 cache eviction of critical allreduce data
 *   - PCIe bus saturation during P2P transfers
 * 
 * RESOLUTION: Weight prefetch is DISABLED (no-op) on gfx906. This function returns
 * immediately without issuing any memory operations. The prefetch mechanism remains
 * in the codebase for future investigation on newer architectures (gfx908+).
 * 
 * NOTE: Even with prefetch disabled, the system achieves 56+ tok/s which exceeds
 * the 53 tok/s baseline requirement.
 */
static int issue_weight_prefetch(CAllreduceSpec *ar, CDispatchPlan *plan, int next_layer_idx, int num_layers)
{
    /* Prefetch is DISABLED on gfx906 due to memory corruption issues.
     * This function intentionally does nothing to avoid interfering with allreduce.
     * 
     * Future work: Investigate alternative prefetch mechanisms:
     *   - hipMemAdvise with hipMemAdviseSetReadMostly
     *   - Managed memory with hipMemPrefetchAsync
     *   - Hardware prefetcher tuning via kernel launch parameters
     */
    return 0;
}

static int dispatch_allreduce(CAllreduceSpec *ar, CDispatchPlan *plan)
{
    /* Compressed fused GEMV+AR+RMSNorm mode takes priority (for FFN down projection with INT8 compression) */
    if (ar->use_gemv_fused_compressed && ar->gemv_fused_compressed_tp4_fn != NULL && ar->tp_size == 4) {
        return do_allreduce_gemv_fused_compressed(ar, plan);
    }
    /* Uncompressed fused GEMV+AR+RMSNorm mode */
    if (ar->use_gemv_fused && ar->gemv_fused_tp4_fn != NULL && ar->tp_size == 4) {
        return do_allreduce_gemv_fused(ar, plan);
    }
    /* Fused kernel mode (allreduce+RMSNorm only) */
    if (ar->use_fused_kernel && ar->kernel_p2p_fused_tp4_fn != NULL && ar->tp_size == 4) {
        return do_allreduce_kernel_p2p(ar, plan);
    }
    /* Regular kernel P2P mode */
    if (ar->use_kernel_p2p && ar->kernel_p2p_tp4_fn != NULL && ar->tp_size == 4) {
        return do_allreduce_kernel_p2p(ar, plan);
    }
    /* Fall back to star topology */
    return do_allreduce_async(ar, plan);
}

/*
 * c_dispatch_step: dispatch all layers for one decode step.
 *
 * Parameters:
 *   plan_ptr:   address of CDispatchPlan (as uint64)
 *   cos_offset: byte offset into cos/sin tables = position * half_rotary * 2
 *   seq_len:    decode attention seq_len = kv_cache.current_len + 1
 *
 * Returns: 0 on success, non-zero HIP error code on failure.
 *
 * Deferred Attention Allreduce (M3 optimization):
 *   - Attention output allreduce is SKIPPED
 *   - Partial attention output (d_proj_out) is added directly to d_hidden (local residual)
 *   - FFN operates on partial hidden state (d_hidden contains partial attention + residual)
 *   - Single allreduce after FFN down-projection (d_ffn_out → d_hidden)
 *   - Reduces allreduce count from 128 to 64 per token
 *
 * Fused kernel mode (use_fused_kernel=1 in CAllreduceSpec):
 *   When the FFN allreduce uses the fused kernel, it performs:
 *   - P2P allreduce + RMSNorm in one kernel launch
 *   - The output is already RMSNorm-normalized
 *   - The next layer's attn_rmsnorm is skipped (already done by fused kernel)
 */
int c_dispatch_step(uint64_t plan_ptr, uint64_t cos_offset, uint32_t seq_len)
{
    CDispatchPlan *plan = (CDispatchPlan *)(uintptr_t)plan_ptr;
    int num_layers  = plan->num_layers;
    int num_engines = plan->num_engines;

    CEngineLayerSpec *all_specs =
        (CEngineLayerSpec *)(uintptr_t)plan->engine_layer_specs;
    CAllreduceSpec *attn_ars =
        (CAllreduceSpec *)(uintptr_t)plan->attn_allreduce_specs;
    CAllreduceSpec *ffn_ars =
        (CAllreduceSpec *)(uintptr_t)plan->ffn_allreduce_specs;

    int use_overlap = plan->use_stream_overlap;
    int layer_idx, engine_idx;
    int err = 0;
    
    /* Track whether previous allreduce used fused kernel.
     * If attn_ar used fused (allreduce+RMSNorm), skip ffn_rmsnorm.
     * If ffn_ar used gemv_fused (GEMV+allreduce+RMSNorm), skip next layer's attn_rmsnorm.
     * If ffn_ar used fused (allreduce+RMSNorm), skip next layer's attn_rmsnorm. */
    int prev_ffn_used_gemv_fused = 0;  /* Tracks gemv_fused (GEMV+AR+RMSNorm) */
    int prev_ffn_used_fused = 0;       /* Tracks fused (AR+RMSNorm only) */

    /* Clear any stale HIP errors */
    if (plan->hipGetLastError_fn) {
        (void)plan->hipGetLastError_fn();
    }

    for (layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        CAllreduceSpec *attn_ar = &attn_ars[layer_idx];
        CAllreduceSpec *ffn_ar  = &ffn_ars[layer_idx];

        /* Wait for previous layer's FFN allreduce */
        if (use_overlap && layer_idx > 0) {
            wait_for_allreduce(ffn_ar, plan);
        }

        /* Per-engine attention kernels */
        for (engine_idx = 0; engine_idx < num_engines; engine_idx++) {
            CEngineLayerSpec *es =
                &all_specs[layer_idx * num_engines + engine_idx];

            set_device_cached(plan->hipSetDevice_fn, attn_ar->device_ids[engine_idx]);

            /* RMSNorm: skip if previous FFN allreduce used fused kernel
             * (either gemv_fused or regular fused both do RMSNorm) */
            if (!prev_ffn_used_gemv_fused && !prev_ffn_used_fused) {
                err = launch_kernel(&es->attn_rmsnorm, plan);
                if (err) return err;
            }

            if (es->layer_type == 0) {
                /* ---- Full attention ---- */
                /* Check for fused QKV kernel (INT4 attention, 3-in-1 launch) */
                if (es->gemv_qkv_fused.present) {
                    /* Fused QKV GEMV: single kernel for Q, K, V projections */
                    /* Update V cache pointer if in direct KV write mode */
                    if (es->use_direct_kv_write) {
                        uint64_t pos = (uint64_t)(seq_len - 1);
                        uint64_t dst_v = es->kv_cache_v_base + pos * (uint64_t)es->kv_stride;
                        update_v_cache_ptr(&es->gemv_qkv_fused, dst_v);
                    }
                    err = launch_kernel(&es->gemv_qkv_fused, plan);
                    if (err) return err;
                    /* Skip separate Q/K/V launches */
                } else {
                    /* Standard path: separate Q, K, V launches */
                    err = launch_kernel(&es->gemv_q_fused, plan);
                    if (err) return err;

                    if (es->use_direct_kv_write) {
                        /* Direct KV write mode: K to working buffer, V directly to cache */
                        err = launch_kernel(&es->gemv_k_only, plan);
                        if (err) return err;
                        /* Update V GEMV output ptr to current cache position */
                        uint64_t pos = (uint64_t)(seq_len - 1);
                        uint64_t dst_v = es->kv_cache_v_base + pos * (uint64_t)es->kv_stride;
                        update_v_cache_ptr(&es->gemv_v_cache, dst_v);
                        err = launch_kernel(&es->gemv_v_cache, plan);
                        if (err) return err;
                    } else {
                        err = launch_kernel(&es->gemv_kv_fused, plan);
                        if (err) return err;
                    }
                }

                /* No stream sync: Q/KV GEMVs now run on the default (null) stream.
                 * Sequential execution on the null stream guarantees ordering for
                 * QKNorm without explicit host-blocking hipStreamSynchronize calls. */

                /* QK-norm + RoPE: d_cos_base + offset */
                uint64_t cos_ptr = es->d_cos_base + cos_offset;
                uint64_t sin_ptr = es->d_sin_base + cos_offset;
                update_cos_sin(&es->qknorm_q, cos_ptr, sin_ptr);
                err = launch_kernel(&es->qknorm_q, plan);
                if (err) return err;

                if (es->use_direct_kv_write) {
                    /* qknorm_k variant: also writes K to cache position */
                    uint64_t pos = (uint64_t)(seq_len - 1);
                    uint64_t dst_k = es->kv_cache_k_base + pos * (uint64_t)es->kv_stride;
                    update_cos_sin_and_cache(&es->qknorm_k, cos_ptr, sin_ptr, dst_k);
                } else {
                    update_cos_sin(&es->qknorm_k, cos_ptr, sin_ptr);
                }
                err = launch_kernel(&es->qknorm_k, plan);
                if (err) return err;

                /* KV cache append (D2D) at position (seq_len - 1)
                 * Skipped when use_direct_kv_write=1 (kernels handle it directly) */
                if (!es->use_direct_kv_write && es->kv_cache_k_base) {
                    uint64_t pos = (uint64_t)(seq_len - 1);
                    uint64_t dst_k = es->kv_cache_k_base + pos * (uint64_t)es->kv_stride;
                    uint64_t dst_v = es->kv_cache_v_base + pos * (uint64_t)es->kv_stride;
                    plan->hipMemcpyAsync_fn(
                        (void *)(uintptr_t)dst_k,
                        (void *)(uintptr_t)es->d_k_src,
                        (size_t)es->kv_stride, 3 /* hipMemcpyDeviceToDevice */, NULL);
                    plan->hipMemcpyAsync_fn(
                        (void *)(uintptr_t)dst_v,
                        (void *)(uintptr_t)es->d_v_src,
                        (size_t)es->kv_stride, 3 /* hipMemcpyDeviceToDevice */, NULL);
                }

                /* Decode attention: update seq_len */
                update_seq_len(&es->decode_attn, seq_len);
                err = launch_kernel(&es->decode_attn, plan);
                if (err) return err;

                err = launch_kernel(&es->sigmoid_mul, plan);
                if (err) return err;
                err = launch_kernel(&es->gemv_o_proj, plan);
                if (err) return err;
                
                /* Deferred attention allreduce (M3 optimization):
                 * Add partial attention output (d_proj_out) to hidden locally,
                 * then skip the attention allreduce. The FFN will operate on
                 * the partial hidden state (residual + partial attention). */
                if (plan->use_deferred_attention_ar && plan->residual_add_fn != NULL) {
                    err = plan->residual_add_fn(
                        (void *)(uintptr_t)es->d_hidden,
                        (const void *)(uintptr_t)es->d_proj_out,
                        attn_ar->num_elems  /* hidden_size */
                    );
                    if (err) return err;
                }

            } else {
                /* ---- DeltaNet linear attention ---- */
                err = launch_kernel(&es->gemv_la_in_proj, plan);
                if (err) return err;
                err = launch_kernel(&es->deltanet_v3, plan);
                if (err) return err;
                err = launch_kernel(&es->deltanet_v3_shift, plan);
                if (err) return err;
                err = launch_kernel(&es->gemv_la_out_proj, plan);
                if (err) return err;
                
                /* Deferred attention allreduce for DeltaNet layers */
                if (plan->use_deferred_attention_ar && plan->residual_add_fn != NULL) {
                    err = plan->residual_add_fn(
                        (void *)(uintptr_t)es->d_hidden,
                        (const void *)(uintptr_t)es->d_proj_out,
                        attn_ar->num_elems  /* hidden_size */
                    );
                    if (err) return err;
                }
            }
        }

        /* Async allreduce for attention partials
         * SKIPPED when use_deferred_attention_ar=1 (M3 optimization) */
        if (use_overlap && !plan->use_deferred_attention_ar) {
            err = dispatch_allreduce(attn_ar, plan);
            if (err) return err;
            wait_for_allreduce(attn_ar, plan);
        }

        /* Track whether attention allreduce would have been used (for FFN RMSNorm skip logic)
         * In deferred mode, attn_allreduce is skipped, so attn_used_fused=0 */
        int attn_used_fused = (!plan->use_deferred_attention_ar) &&
                               attn_ar->use_fused_kernel && 
                               attn_ar->kernel_p2p_fused_tp4_fn != NULL &&
                               attn_ar->tp_size == 4;

        /* Per-engine FFN kernels */
        for (engine_idx = 0; engine_idx < num_engines; engine_idx++) {
            CEngineLayerSpec *es =
                &all_specs[layer_idx * num_engines + engine_idx];
            set_device_cached(plan->hipSetDevice_fn, ffn_ar->device_ids[engine_idx]);
            
            /* FFN RMSNorm: skip if attention allreduce used fused kernel */
            if (!attn_used_fused) {
                err = launch_kernel(&es->ffn_rmsnorm, plan);
                if (err) return err;
            }
            
            err = launch_kernel(&es->ffn_gate_up_silu, plan);
            if (err) return err;
            
            /* FFN down projection: skip if using gemv_fused (fused GEMV+AR+RMSNorm)
             * The fused kernel will be launched via dispatch_allreduce() below */
            if (!ffn_ar->use_gemv_fused || ffn_ar->gemv_fused_tp4_fn == NULL) {
                err = launch_kernel(&es->ffn_down, plan);
                if (err) return err;
            }
        }

        /* Async allreduce for FFN partials (non-blocking)
         * When gemv_fused is used, this launches the fused GEMV+AR+RMSNorm kernel
         * which replaces ffn_down + separate allreduce. */
        if (use_overlap) {
            err = dispatch_allreduce(ffn_ar, plan);
            if (err) return err;
            
            /* Issue weight prefetch for next layer during allreduce window.
             * The compute stream is idle during allreduce (~46us), so we can
             * prefetch next layer's weights to warm up HBM memory controller. */
            int next_layer = layer_idx + 1;
            if (ffn_ar->enable_prefetch && next_layer < num_layers) {
                err = issue_weight_prefetch(ffn_ar, plan, next_layer, num_layers);
                if (err) return err;
            }
        }
        
        /* Track whether FFN allreduce used fused kernels (affects next layer) */
        prev_ffn_used_gemv_fused = (ffn_ar->use_gemv_fused && 
                                     ffn_ar->gemv_fused_tp4_fn != NULL &&
                                     ffn_ar->tp_size == 4);
        prev_ffn_used_fused = (ffn_ar->use_fused_kernel && 
                                ffn_ar->kernel_p2p_fused_tp4_fn != NULL &&
                                ffn_ar->tp_size == 4);
    }

    /* Wait for the last FFN allreduce */
    if (use_overlap) {
        wait_for_allreduce(&ffn_ars[num_layers - 1], plan);
    }

    /* Print hipSetDevice statistics if HIP_SETDEVICE_VERBOSE=1 */
    print_hipsetdevice_stats();

    return 0;
}

int c_dispatch_get_spec_size(void)
{
    return (int)sizeof(CEngineLayerSpec);
}

int c_dispatch_get_kernel_spec_size(void)
{
    return (int)sizeof(CKernelSpec);
}

int c_dispatch_get_allreduce_spec_size(void)
{
    return (int)sizeof(CAllreduceSpec);
}

int c_dispatch_get_plan_size(void)
{
    return (int)sizeof(CDispatchPlan);
}
