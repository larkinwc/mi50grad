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
} CEngineLayerSpec;

/*
 * CAllreduceSpec: parameters for one async allreduce call.
 */
typedef struct {
    /* p2p_reduce_residual_tp{N} function pointers */
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
    uint32_t _pad;
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
        plan->hipSetDevice_fn(ar->device_ids[i]);
        plan->hipEventRecord_fn(
            (void *)(uintptr_t)ar->compute_events[i],
            (void *)(uintptr_t)ar->compute_streams[i]);
    }

    /* Step 2: GPU0's allreduce stream waits for all compute events */
    plan->hipSetDevice_fn(ar->device_ids[0]);
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
    plan->hipSetDevice_fn(ar->device_ids[0]);
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
    plan->hipSetDevice_fn(ar->device_ids[0]);
    plan->hipEventRecord_fn(
        (void *)(uintptr_t)ar->ar_done_events[0], ar_stream0);

    for (i = 1; i < tp; i++) {
        plan->hipSetDevice_fn(ar->device_ids[i]);
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
    plan->hipSetDevice_fn(ar->device_ids[0]);
    plan->hipEventRecord_fn(
        (void *)(uintptr_t)ar->ar_done_events[0], ar_stream0);

    return 0;
}

static void wait_for_allreduce(CAllreduceSpec *ar, CDispatchPlan *plan)
{
    int tp = ar->tp_size;
    int i;
    for (i = 0; i < tp; i++) {
        plan->hipSetDevice_fn(ar->device_ids[i]);
        plan->hipStreamWaitEvent_fn(
            (void *)(uintptr_t)ar->compute_streams[i],
            (void *)(uintptr_t)ar->ar_done_events[i],
            0);
    }
}

/* ---------------------------------------------------------------------- */
/* Main dispatch function                                                   */
/* ---------------------------------------------------------------------- */

/*
 * c_dispatch_step: dispatch all layers for one decode step.
 *
 * Parameters:
 *   plan_ptr:   address of CDispatchPlan (as uint64)
 *   cos_offset: byte offset into cos/sin tables = position * half_rotary * 2
 *   seq_len:    decode attention seq_len = kv_cache.current_len + 1
 *
 * Returns: 0 on success, non-zero HIP error code on failure.
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

            plan->hipSetDevice_fn(attn_ar->device_ids[engine_idx]);

            /* RMSNorm */
            err = launch_kernel(&es->attn_rmsnorm, plan);
            if (err) return err;

            if (es->layer_type == 0) {
                /* ---- Full attention ---- */
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
            }
        }

        /* Async allreduce for attention partials */
        if (use_overlap) {
            err = do_allreduce_async(attn_ar, plan);
            if (err) return err;
            wait_for_allreduce(attn_ar, plan);
        }

        /* Per-engine FFN kernels */
        for (engine_idx = 0; engine_idx < num_engines; engine_idx++) {
            CEngineLayerSpec *es =
                &all_specs[layer_idx * num_engines + engine_idx];
            plan->hipSetDevice_fn(ffn_ar->device_ids[engine_idx]);
            err = launch_kernel(&es->ffn_rmsnorm, plan);
            if (err) return err;
            err = launch_kernel(&es->ffn_gate_up_silu, plan);
            if (err) return err;
            err = launch_kernel(&es->ffn_down, plan);
            if (err) return err;
        }

        /* Async allreduce for FFN partials (non-blocking) */
        if (use_overlap) {
            err = do_allreduce_async(ffn_ar, plan);
            if (err) return err;
        }
    }

    /* Wait for the last FFN allreduce */
    if (use_overlap) {
        wait_for_allreduce(&ffn_ars[num_layers - 1], plan);
    }

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
