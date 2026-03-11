// mi50grad GCN assembly macros for gfx906
// Common macros used across all assembly kernels

// ============================================================
// Kernel descriptor helpers
// ============================================================

// AMDGPU kernel descriptor (.amdhsa_kernel) boilerplate
// Usage: KERNEL_BEGIN name
.macro KERNEL_BEGIN name
.text
.globl \name
.p2align 8
.type \name,@function
\name:
.endm

// ============================================================
// Timing macros for ISA probes
// ============================================================

// Read hardware cycle counter (s_memrealtime returns 64-bit timestamp)
// Result in s[dst:dst+1]
.macro READ_CLOCK dst
    s_memrealtime s[\dst:\dst+1]
    s_waitcnt lgkmcnt(0)
.endm

// ============================================================
// Workgroup / thread ID helpers
// ============================================================

// Get flattened local thread ID into VGPR dst
.macro GET_LOCAL_ID dst
    v_mov_b32 \dst, v0
.endm

// Get workgroup ID (dimension 0) into SGPR dst
.macro GET_WORKGROUP_ID dst
    s_mov_b32 \dst, s2
.endm

// ============================================================
// Memory operation helpers
// ============================================================

// Global load dwordx4 with offset
.macro GLOBAL_LOAD_DWORDX4 dst, addr, offset
    global_load_dwordx4 \dst, \addr, off offset:\offset
.endm

// Global store dwordx4 with offset
.macro GLOBAL_STORE_DWORDX4 addr, src, offset
    global_store_dwordx4 \addr, \src, off offset:\offset
.endm

// LDS load/store
.macro LDS_READ_B32 dst, addr
    ds_read_b32 \dst, \addr
.endm

.macro LDS_WRITE_B32 addr, src
    ds_write_b32 \addr, \src
.endm

// ============================================================
// Synchronization
// ============================================================

.macro BARRIER
    s_barrier
.endm

.macro WAIT_VMCNT count=0
    s_waitcnt vmcnt(\count)
.endm

.macro WAIT_LGKMCNT count=0
    s_waitcnt lgkmcnt(\count)
.endm

.macro WAIT_ALL
    s_waitcnt vmcnt(0) lgkmcnt(0)
.endm

// ============================================================
// DPP helpers (for warp-level reductions)
// ============================================================

// Row shift right by 1 (within 16 lanes)
.macro DPP_ROW_SHR1 dst, src
    v_mov_b32 \dst, \src row_shr:1
.endm

// Row shift right by 2
.macro DPP_ROW_SHR2 dst, src
    v_mov_b32 \dst, \src row_shr:2
.endm

// Row shift right by 4
.macro DPP_ROW_SHR4 dst, src
    v_mov_b32 \dst, \src row_shr:4
.endm

// Row shift right by 8
.macro DPP_ROW_SHR8 dst, src
    v_mov_b32 \dst, \src row_shr:8
.endm

// Broadcast lane 15 of each row to all lanes in row
.macro DPP_ROW_BCAST15 dst, src
    v_mov_b32 \dst, \src row_bcast:15
.endm

// Broadcast lane 31 to lanes 32-63
.macro DPP_ROW_BCAST31 dst, src
    v_mov_b32 \dst, \src row_bcast:31
.endm

// ============================================================
// Reduction helpers (FP32 add across wavefront using DPP)
// ============================================================

// Full wavefront (64 lanes) FP32 sum reduction
// Input: val register. Result broadcast to all lanes.
// Clobbers: tmp register
.macro WAVE_REDUCE_ADD_F32 val, tmp
    // Phase 1: reduce within each row of 16
    v_add_f32 \tmp, \val, \val row_shr:1
    v_add_f32 \tmp, \tmp, \tmp row_shr:2
    v_add_f32 \tmp, \tmp, \tmp row_shr:4
    v_add_f32 \tmp, \tmp, \tmp row_shr:8
    // Phase 2: broadcast row results and combine
    v_add_f32 \tmp, \tmp, \tmp row_bcast:15
    v_add_f32 \tmp, \tmp, \tmp row_bcast:31
    v_mov_b32 \val, \tmp
.endm

// Full wavefront FP32 max reduction
.macro WAVE_REDUCE_MAX_F32 val, tmp
    v_max_f32 \tmp, \val, \val row_shr:1
    v_max_f32 \tmp, \tmp, \tmp row_shr:2
    v_max_f32 \tmp, \tmp, \tmp row_shr:4
    v_max_f32 \tmp, \tmp, \tmp row_shr:8
    v_max_f32 \tmp, \tmp, \tmp row_bcast:15
    v_max_f32 \tmp, \tmp, \tmp row_bcast:31
    v_mov_b32 \val, \tmp
.endm
