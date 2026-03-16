/*
 * c_graph_dispatch.c: Tight C loop for HIP graph replay in TP=4 decode.
 *
 * Eliminates the Python overhead that makes graph dispatch ~25% slower than
 * C dispatch. The Python-level replay loop adds ~8ms/token overhead from 512
 * hipGraphLaunch + 256 hipGraphExecKernelNodeSetParams ctypes calls.
 * This C extension does the same work in a tight C loop with minimal overhead.
 *
 * Build command (on dev server, inside Docker):
 *   gcc -O3 -shared -fPIC -I/opt/rocm/include -L/opt/rocm/lib -lamdhip64 \
 *       -o src/runtime/c_graph_dispatch.so src/runtime/c_graph_dispatch.c
 *
 * Key design:
 * - Python captures HIP graphs and builds graph exec handles + mutable node handles
 * - Python serializes graph specs into CGraphDispatchPlan ctypes structures
 * - C extension iterates layers, updates mutable params, replays graphs, handles allreduce
 * - No Python callback between layers — the whole step is one C function call
 * - Same allreduce protocol as c_dispatch.c (do_allreduce_async + wait_for_allreduce)
 *
 * Mutable params updated per step:
 * - cos/sin table pointers (position-dependent, full attention layers only)
 * - seq_len for decode attention (grows each step)
 * - KV cache write pointers (for direct-KV-write mode)
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---------------------------------------------------------------------- */
/* HIP API function typedefs (no HIP headers — use function pointers)      */
/* ---------------------------------------------------------------------- */

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
typedef int (*hipMemcpyAsync_t)(void*, void*, size_t, int, void*);

/* hipError_t hipGetLastError() */
typedef int (*hipGetLastError_t)(void);

/* hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) */
typedef int (*hipGraphLaunch_t)(void*, void*);

/* hipError_t hipGraphExecKernelNodeSetParams(
 *     hipGraphExec_t hGraphExec, hipGraphNode_t node,
 *     const hipKernelNodeParams* pNodeParams) */
typedef int (*hipGraphExecKernelNodeSetParams_t)(void*, void*, const void*);

/* ---------------------------------------------------------------------- */
/* hipKernelNodeParams (must match ROCm ABI exactly)                       */
/* ---------------------------------------------------------------------- */

/*
 * ROCm hipKernelNodeParams layout (as confirmed on gfx906/ROCm 7.1):
 *   blockDim: {x, y, z}  (3 × uint32 = 12 bytes)
 *   extra:                (1 × pointer = 8 bytes on 64-bit)
 *   func:                 (1 × pointer = 8 bytes)
 *   gridDim: {x, y, z}   (3 × uint32 = 12 bytes)
 *   kernelParams:         (1 × pointer = 8 bytes)
 *   sharedMemBytes:       (1 × uint32 = 4 bytes)
 *
 * Total: 12 + 8 + 8 + 12 + 8 + 4 = 52 bytes (with natural alignment padding)
 * Note: on 64-bit systems, 'extra' and 'func' require 8-byte alignment.
 * With padding: blockDim (12) + 4 pad + extra (8) + func (8) + gridDim (12) +
 *               kernelParams (8) + sharedMemBytes (4) + 4 pad = 60 bytes
 */
typedef struct {
    uint32_t blockDimX;
    uint32_t blockDimY;
    uint32_t blockDimZ;
    uint32_t _pad1;       /* alignment padding before 8-byte pointer */
    void     *extra;
    void     *func;
    uint32_t gridDimX;
    uint32_t gridDimY;
    uint32_t gridDimZ;
    uint32_t _pad2;       /* alignment padding before 8-byte pointer */
    void     *kernelParams;
    uint32_t sharedMemBytes;
    uint32_t _pad3;       /* alignment at end */
} HipKernelNodeParams;

/* ---------------------------------------------------------------------- */
/* Mutable param descriptor — one per mutable kernel param slot             */
/* ---------------------------------------------------------------------- */

/*
 * CMutableParam: describes one mutable slot in a captured kernel node.
 *
 * The C code uses this to:
 * 1. Read the current value from params_array[param_index]
 * 2. Write new_value_ptr there
 * 3. Call hipGraphExecKernelNodeSetParams with the updated HipKernelNodeParams
 *
 * Each CGraphLayerSpec contains a flat array of CMutableParam entries for
 * all mutable slots across all kernels in the attention segment.
 */
#define MUTABLE_TYPE_COS_PTR    1  /* cos table ptr (uint64 param) */
#define MUTABLE_TYPE_SIN_PTR    2  /* sin table ptr (uint64 param) */
#define MUTABLE_TYPE_SEQ_LEN    3  /* seq_len (uint32 param) */
#define MUTABLE_TYPE_KV_K_PTR   4  /* KV cache K write ptr (uint64 param) */
#define MUTABLE_TYPE_KV_V_PTR   5  /* KV cache V write ptr (uint64 param) */

typedef struct {
    uint64_t  node;           /* hipGraphNode_t (as uint64) */
    uint64_t  graph_exec;     /* hipGraphExec_t (as uint64) — owns the update */
    uint64_t  params_array;   /* void** pointing to ctypes param objects */
    uint32_t  param_index;    /* index into params_array for this mutable slot */
    uint32_t  mutable_type;   /* MUTABLE_TYPE_* constant */
    /* Full hipKernelNodeParams for this node — updated in-place before SetParams */
    HipKernelNodeParams kparams;
    /* Base KV cache pointer and stride for KV_K_PTR / KV_V_PTR types */
    uint64_t  kv_cache_base;  /* kv_cache_k_base or kv_cache_v_base */
    uint32_t  kv_stride;      /* stride per position in bytes */
    uint32_t  _pad;
    /* cos/sin base pointers for COS_PTR / SIN_PTR types */
    uint64_t  d_cos_base;
    uint64_t  d_sin_base;
} CMutableParam;

/* ---------------------------------------------------------------------- */
/* CGraphLayerSpec: per-GPU, per-layer struct with graph handles            */
/* ---------------------------------------------------------------------- */

/*
 * Maximum mutable params per layer:
 * qknorm_q: 2 (cos, sin)
 * qknorm_k: 2 (cos, sin) + 1 (k_cache_ptr) = 3
 * gemv_v_cache: 1 (v_cache_ptr)
 * decode_attn: 1 (seq_len)
 * Total: 7 max for full_attention
 */
#define MAX_MUTABLE_PARAMS 8

typedef struct {
    /* Graph exec handles (one per segment per layer, per GPU) */
    uint64_t attn_graph_exec;  /* hipGraphExec_t for attention segment */
    uint64_t ffn_graph_exec;   /* hipGraphExec_t for FFN segment */

    /* Mutable params for the attention segment */
    CMutableParam mutable_params[MAX_MUTABLE_PARAMS];
    uint32_t      num_mutable;

    int layer_type;  /* 0=full_attention, 1=deltanet */
    uint32_t _pad;
} CGraphLayerSpec;

/* ---------------------------------------------------------------------- */
/* CAllreduceSpec: same as in c_dispatch.c (copy for independence)         */
/* ---------------------------------------------------------------------- */

/*
 * kernel_p2p_allreduce_residual_tp4 signature:
 *   (hidden, partial_local, peer0, peer1, peer2, n, stream) → int
 * One call per GPU on its allreduce stream; all 4 run concurrently.
 */
typedef int (*kernel_p2p_ar_tp4_t)(void *hidden,
                                    void *partial_local,
                                    void *peer0, void *peer1, void *peer2,
                                    uint32_t n, void *stream);

typedef struct {
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

    /* Kernel P2P allreduce support (use_kernel_p2p = 1 → use this path) */
    uint32_t use_kernel_p2p;   /* 0=standard P2P, 1=kernel P2P */
    uint32_t _pad2;
    kernel_p2p_ar_tp4_t kernel_p2p_fn;  /* function ptr (NULL if not available) */
    /* peer_ptrs[gpu][peer]: partial_ptrs of OTHER GPUs from each GPU's perspective */
    /* For GPU i: peer_ptrs[i][0..2] = partial_ptrs of the 3 OTHER GPUs */
    uint64_t peer_ptrs[4][3];  /* peer partial ptrs for kernel P2P allreduce */
} CGraphAllreduceSpec;

/* ---------------------------------------------------------------------- */
/* CGraphDispatchPlan: top-level plan passed from Python                   */
/* ---------------------------------------------------------------------- */

typedef struct {
    int     num_layers;
    int     num_engines;
    uint64_t graph_layer_specs;   /* CGraphLayerSpec[num_layers * num_engines] */
    uint64_t attn_allreduce_specs;/* CGraphAllreduceSpec[num_layers] */
    uint64_t ffn_allreduce_specs; /* CGraphAllreduceSpec[num_layers] */

    /* HIP API function pointers */
    hipSetDevice_t                   hipSetDevice_fn;
    hipStreamSynchronize_t           hipStreamSynchronize_fn;
    hipEventRecord_t                 hipEventRecord_fn;
    hipStreamWaitEvent_t             hipStreamWaitEvent_fn;
    hipMemcpyPeerAsync_t             hipMemcpyPeerAsync_fn;
    hipMemcpyAsync_t                 hipMemcpyAsync_fn;
    hipGetLastError_t                hipGetLastError_fn;
    hipGraphLaunch_t                 hipGraphLaunch_fn;
    hipGraphExecKernelNodeSetParams_t hipGraphExecKernelNodeSetParams_fn;
} CGraphDispatchPlan;

/* ---------------------------------------------------------------------- */
/* Internal helpers                                                         */
/* ---------------------------------------------------------------------- */

static int do_graph_allreduce_async(CGraphAllreduceSpec *ar,
                                     CGraphDispatchPlan   *plan)
{
    int tp = ar->tp_size;
    if (tp <= 1) return 0;

    int i, err;
    uint32_t n = ar->num_elems;
    size_t size = (size_t)n * 2;

    /* === Kernel P2P allreduce path (fast: single kernel per GPU, no copies) === */
    if (ar->use_kernel_p2p && ar->kernel_p2p_fn && tp == 4) {
        /* Step 1: Record compute events on each GPU's compute stream */
        for (i = 0; i < tp; i++) {
            plan->hipSetDevice_fn(ar->device_ids[i]);
            plan->hipEventRecord_fn(
                (void *)(uintptr_t)ar->compute_events[i],
                (void *)(uintptr_t)ar->compute_streams[i]);
        }

        /* Step 2: Each GPU's allreduce stream waits for ALL compute events
         * (required so no GPU reads peer partials before ALL GPUs have finished
         * their compute) */
        for (i = 0; i < tp; i++) {
            void *ar_si = (void *)(uintptr_t)ar->allreduce_streams[i];
            plan->hipSetDevice_fn(ar->device_ids[i]);
            int j;
            for (j = 0; j < tp; j++) {
                plan->hipStreamWaitEvent_fn(
                    ar_si,
                    (void *)(uintptr_t)ar->compute_events[j], 0);
            }
        }

        /* Step 3: Launch kernel P2P allreduce on each GPU's allreduce stream */
        for (i = 0; i < tp; i++) {
            plan->hipSetDevice_fn(ar->device_ids[i]);
            void *ar_si = (void *)(uintptr_t)ar->allreduce_streams[i];
            err = ar->kernel_p2p_fn(
                (void *)(uintptr_t)ar->hidden_ptrs[i],
                (void *)(uintptr_t)ar->partial_ptrs[i],
                (void *)(uintptr_t)ar->peer_ptrs[i][0],
                (void *)(uintptr_t)ar->peer_ptrs[i][1],
                (void *)(uintptr_t)ar->peer_ptrs[i][2],
                n, ar_si);
            if (err) return err;
            /* Record done event on each GPU's allreduce stream */
            plan->hipEventRecord_fn(
                (void *)(uintptr_t)ar->ar_done_events[i], ar_si);
        }

        return 0;
    }

    /* === Standard P2P allreduce path (gather-reduce-broadcast via GPU0) === */
    void *ar_stream0 = (void *)(uintptr_t)ar->allreduce_streams[0];

    /* Step 1: Record compute events on each GPU's compute stream */
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

static void wait_for_graph_allreduce(CGraphAllreduceSpec *ar,
                                      CGraphDispatchPlan   *plan)
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

/*
 * update_mutable_params: update position-dependent params in the graph exec.
 *
 * For each CMutableParam in the layer spec's mutable list:
 *   1. Compute the new value based on mutable_type
 *   2. Write the new value into the ctypes param object at params_array[param_index]
 *   3. Call hipGraphExecKernelNodeSetParams with the updated kparams
 *      (kparams.kernelParams still points to the same ctypes params_array,
 *       so the updated value is picked up automatically)
 */
static int update_mutable_params(CGraphLayerSpec *gs,
                                  CGraphDispatchPlan *plan,
                                  uint64_t cos_offset,
                                  uint32_t seq_len)
{
    uint32_t i;
    for (i = 0; i < gs->num_mutable; i++) {
        CMutableParam *mp = &gs->mutable_params[i];
        void **params = (void **)(uintptr_t)mp->params_array;
        uint32_t idx  = mp->param_index;

        switch (mp->mutable_type) {
        case MUTABLE_TYPE_COS_PTR:
            *((uint64_t *)params[idx]) = mp->d_cos_base + cos_offset;
            break;
        case MUTABLE_TYPE_SIN_PTR:
            *((uint64_t *)params[idx]) = mp->d_sin_base + cos_offset;
            break;
        case MUTABLE_TYPE_SEQ_LEN:
            *((uint32_t *)params[idx]) = seq_len;
            break;
        case MUTABLE_TYPE_KV_K_PTR:
        case MUTABLE_TYPE_KV_V_PTR: {
            /* KV cache write ptr = base + (seq_len - 1) * stride */
            uint64_t pos = (uint64_t)(seq_len - 1);
            uint64_t ptr = mp->kv_cache_base + pos * (uint64_t)mp->kv_stride;
            *((uint64_t *)params[idx]) = ptr;
            break;
        }
        default:
            break;
        }

        /* Update the graph exec node with the new params */
        int err = plan->hipGraphExecKernelNodeSetParams_fn(
            (void *)(uintptr_t)mp->graph_exec,
            (void *)(uintptr_t)mp->node,
            &mp->kparams
        );
        if (err) return err;
    }
    return 0;
}

/* ---------------------------------------------------------------------- */
/* Main entry point                                                         */
/* ---------------------------------------------------------------------- */

/*
 * c_graph_dispatch_step: replay all HIP graphs for one decode step.
 *
 * Parameters:
 *   plan_ptr:   address of CGraphDispatchPlan (as uint64)
 *   cos_offset: byte offset into cos/sin tables = position * half_rotary * 2
 *   seq_len:    decode attention seq_len = kv_cache.current_len + 1
 *
 * Returns: 0 on success, non-zero HIP error code on failure.
 *
 * Per layer:
 *   1. Wait for previous FFN allreduce (stream event gating, GPU-side)
 *   2. For each GPU: update mutable params (cos/sin, seq_len, kv_ptr)
 *      via hipGraphExecKernelNodeSetParams
 *   3. For each GPU: hipGraphLaunch attention segment
 *   4. Async allreduce (attention partials → d_hidden)
 *   5. Wait for attention allreduce
 *   6. For each GPU: hipGraphLaunch FFN segment
 *   7. Async allreduce (FFN partials → d_hidden, non-blocking)
 * After all layers: wait for final FFN allreduce
 */
int c_graph_dispatch_step(uint64_t plan_ptr, uint64_t cos_offset,
                           uint32_t seq_len)
{
    CGraphDispatchPlan *plan = (CGraphDispatchPlan *)(uintptr_t)plan_ptr;
    int num_layers  = plan->num_layers;
    int num_engines = plan->num_engines;

    CGraphLayerSpec *all_specs =
        (CGraphLayerSpec *)(uintptr_t)plan->graph_layer_specs;
    CGraphAllreduceSpec *attn_ars =
        (CGraphAllreduceSpec *)(uintptr_t)plan->attn_allreduce_specs;
    CGraphAllreduceSpec *ffn_ars =
        (CGraphAllreduceSpec *)(uintptr_t)plan->ffn_allreduce_specs;

    int layer_idx, engine_idx;
    int err = 0;

    /* Clear any stale HIP errors */
    if (plan->hipGetLastError_fn) {
        (void)plan->hipGetLastError_fn();
    }

    for (layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        CGraphAllreduceSpec *attn_ar = &attn_ars[layer_idx];
        CGraphAllreduceSpec *ffn_ar  = &ffn_ars[layer_idx];

        /* Wait for previous layer's FFN allreduce (GPU-side event gating) */
        if (layer_idx > 0) {
            wait_for_graph_allreduce(ffn_ar, plan);
        }

        /* Per-GPU: update mutable params and replay attention segment */
        for (engine_idx = 0; engine_idx < num_engines; engine_idx++) {
            CGraphLayerSpec *gs =
                &all_specs[layer_idx * num_engines + engine_idx];

            plan->hipSetDevice_fn(attn_ar->device_ids[engine_idx]);

            /* Update mutable params (cos/sin, seq_len, kv ptrs) */
            if (gs->layer_type == 0 && gs->num_mutable > 0) {
                err = update_mutable_params(gs, plan, cos_offset, seq_len);
                if (err) return err;
            }

            /* Replay attention graph segment */
            if (gs->attn_graph_exec) {
                err = plan->hipGraphLaunch_fn(
                    (void *)(uintptr_t)gs->attn_graph_exec, NULL /* default stream */);
                if (err) return err;
            }
        }

        /* Async allreduce for attention partials */
        err = do_graph_allreduce_async(attn_ar, plan);
        if (err) return err;

        /* Wait for attention allreduce before FFN */
        wait_for_graph_allreduce(attn_ar, plan);

        /* Per-GPU: replay FFN segment */
        for (engine_idx = 0; engine_idx < num_engines; engine_idx++) {
            CGraphLayerSpec *gs =
                &all_specs[layer_idx * num_engines + engine_idx];

            plan->hipSetDevice_fn(ffn_ar->device_ids[engine_idx]);

            if (gs->ffn_graph_exec) {
                err = plan->hipGraphLaunch_fn(
                    (void *)(uintptr_t)gs->ffn_graph_exec, NULL /* default stream */);
                if (err) return err;
            }
        }

        /* Async allreduce for FFN partials (non-blocking for next layer) */
        err = do_graph_allreduce_async(ffn_ar, plan);
        if (err) return err;
    }

    /* Wait for the last FFN allreduce */
    wait_for_graph_allreduce(&ffn_ars[num_layers - 1], plan);

    return 0;
}

/* ---------------------------------------------------------------------- */
/* Size query functions (for Python ctypes struct size verification)        */
/* ---------------------------------------------------------------------- */

int c_graph_dispatch_get_layer_spec_size(void)
{
    return (int)sizeof(CGraphLayerSpec);
}

int c_graph_dispatch_get_mutable_param_size(void)
{
    return (int)sizeof(CMutableParam);
}

int c_graph_dispatch_get_allreduce_spec_size(void)
{
    return (int)sizeof(CGraphAllreduceSpec);
}

int c_graph_dispatch_get_plan_size(void)
{
    return (int)sizeof(CGraphDispatchPlan);
}

int c_graph_dispatch_get_kparams_size(void)
{
    return (int)sizeof(HipKernelNodeParams);
}
