// RMSNorm kernel for gfx906 (MI50)
// output[i] = (input[i] / sqrt(mean(input^2) + eps)) * weight[i]
//
// Each workgroup processes one row. Grid: (num_rows,1,1), Block: (256,1,1)
// hidden_dim must be multiple of 256.
//
// Kernarg:
//   [0:7]   ptr input  (FP16, [num_rows, hidden_dim])
//   [8:15]  ptr weight  (FP16, [hidden_dim])
//   [16:23] ptr output  (FP16, [num_rows, hidden_dim])
//   [24:27] uint32 hidden_dim
//   [28:31] float  eps

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
.amdhsa_code_object_version 6

.globl rmsnorm_fp16
.p2align 8
.type rmsnorm_fp16,@function
rmsnorm_fp16:
    s_load_dwordx2 s[4:5], s[0:1], 0x0
    s_load_dwordx2 s[6:7], s[0:1], 0x8
    s_load_dwordx2 s[8:9], s[0:1], 0x10
    s_load_dword s10, s[0:1], 0x18
    s_load_dword s11, s[0:1], 0x1C
    s_waitcnt lgkmcnt(0)

    // Row offset
    s_mul_i32 s12, s2, s10
    s_lshl_b32 s12, s12, 1
    s_add_u32 s4, s4, s12
    s_addc_u32 s5, s5, 0
    s_add_u32 s8, s8, s12
    s_addc_u32 s9, s9, 0

    // Phase 1: Partial sum of squares
    s_lshr_b32 s13, s10, 8                  // elems_per_thread
    v_mov_b32 v1, 0                          // sum_sq

    v_lshlrev_b32 v2, 1, v0
    v_mov_b32 v3, s5
    v_add_co_u32 v2, vcc, s4, v2
    v_addc_co_u32 v3, vcc, v3, 0, vcc

    s_movk_i32 s14, 512                      // stride

    s_mov_b32 s15, s13
.Lsum_sq_loop:
    global_load_ushort v4, v[2:3], off
    s_waitcnt vmcnt(0)
    v_cvt_f32_f16 v4, v4
    v_fmac_f32 v1, v4, v4

    v_add_co_u32 v2, vcc, v2, s14
    v_addc_co_u32 v3, vcc, v3, 0, vcc

    s_sub_u32 s15, s15, 1
    s_cmp_gt_u32 s15, 0
    s_cbranch_scc1 .Lsum_sq_loop

    // Phase 2: Tree reduction via LDS
    // All 256 threads write, all hit every barrier, use exec save/restore
    v_lshlrev_b32 v5, 2, v0                 // tid * 4
    ds_write_b32 v5, v1
    s_waitcnt lgkmcnt(0)
    s_barrier

    // Reduce 256 -> 128: threads 0-127 add thread+128's value
    .macro TREE_REDUCE n, byte_offset
        s_mov_b64 s[16:17], exec             // save exec
        .if \n > 64
            s_movk_i32 s18, \n
            v_cmp_lt_u32 vcc, v0, s18
        .else
            v_cmp_lt_u32 vcc, v0, \n
        .endif
        s_and_b64 exec, exec, vcc
        // Active threads do the work
        ds_read_b32 v6, v5 offset:0
        ds_read_b32 v7, v5 offset:\byte_offset
        s_waitcnt lgkmcnt(0)
        v_add_f32 v6, v6, v7
        ds_write_b32 v5, v6
        s_waitcnt lgkmcnt(0)
        s_mov_b64 exec, s[16:17]             // restore exec
        s_barrier
    .endm

    TREE_REDUCE 128, 512
    TREE_REDUCE 64,  256
    TREE_REDUCE 32,  128
    TREE_REDUCE 16,  64
    TREE_REDUCE 8,   32
    TREE_REDUCE 4,   16
    TREE_REDUCE 2,   8
    TREE_REDUCE 1,   4

    // All threads read final sum from LDS[0]
    v_mov_b32 v5, 0
    ds_read_b32 v1, v5
    s_waitcnt lgkmcnt(0)

    // mean = sum / hidden_dim
    v_cvt_f32_u32 v4, s10
    v_rcp_f32 v4, v4
    v_mul_f32 v1, v1, v4

    // rsqrt(mean + eps)
    v_mov_b32 v4, s11
    v_add_f32 v1, v1, v4
    v_rsq_f32 v1, v1

    // Phase 3: Normalize and scale
    v_lshlrev_b32 v2, 1, v0
    v_mov_b32 v3, s5
    v_add_co_u32 v2, vcc, s4, v2
    v_addc_co_u32 v3, vcc, v3, 0, vcc

    v_lshlrev_b32 v12, 1, v0
    v_mov_b32 v13, s9
    v_add_co_u32 v12, vcc, s8, v12
    v_addc_co_u32 v13, vcc, v13, 0, vcc

    v_lshlrev_b32 v14, 1, v0
    v_mov_b32 v15, s7
    v_add_co_u32 v14, vcc, s6, v14
    v_addc_co_u32 v15, vcc, v15, 0, vcc

    s_mov_b32 s15, s13
.Lnorm_loop:
    global_load_ushort v4, v[2:3], off
    global_load_ushort v5, v[14:15], off
    s_waitcnt vmcnt(0)

    v_cvt_f32_f16 v4, v4
    v_cvt_f32_f16 v5, v5
    v_mul_f32 v4, v4, v1
    v_mul_f32 v4, v4, v5
    v_cvt_f16_f32 v4, v4

    global_store_short v[12:13], v4, off

    v_add_co_u32 v2, vcc, v2, s14
    v_addc_co_u32 v3, vcc, v3, 0, vcc
    v_add_co_u32 v12, vcc, v12, s14
    v_addc_co_u32 v13, vcc, v13, 0, vcc
    v_add_co_u32 v14, vcc, v14, s14
    v_addc_co_u32 v15, vcc, v15, 0, vcc

    s_sub_u32 s15, s15, 1
    s_cmp_gt_u32 s15, 0
    s_cbranch_scc1 .Lnorm_loop

    s_waitcnt vmcnt(0)
    s_endpgm
.Lfunc_end_rmsnorm:
    .size rmsnorm_fp16, .Lfunc_end_rmsnorm - rmsnorm_fp16

.rodata
.p2align 6
.amdhsa_kernel rmsnorm_fp16
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_private_segment_buffer 0
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_private_segment_wavefront_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 20
    .amdhsa_reserve_vcc 1
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_ieee_mode 1
    .amdhsa_dx10_clamp 1
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_fp16_overflow 0
    .amdhsa_uses_dynamic_stack 0
.end_amdhsa_kernel

    .amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 8
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 16
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 24
        .size: 4
        .value_kind: by_value
      - .offset: 28
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 256
    .name:           rmsnorm_fp16
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .symbol:         rmsnorm_fp16.kd
    .vgpr_count:     16
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
