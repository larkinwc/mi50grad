// Residual add kernel for gfx906: dst[i] += src[i] (FP16)
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
// Kernarg: [0:7] dst, [8:15] src, [16:19] n

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
.amdhsa_code_object_version 6

.globl residual_add_fp16
.p2align 8
.type residual_add_fp16,@function
residual_add_fp16:
    s_load_dwordx2 s[4:5], s[0:1], 0x0
    s_load_dwordx2 s[6:7], s[0:1], 0x8
    s_load_dword s8, s[0:1], 0x10
    s_waitcnt lgkmcnt(0)

    s_lshl_b32 s9, s2, 8
    v_add_co_u32 v1, vcc, s9, v0            // global_id

    v_cmp_lt_u32 vcc, v1, s8
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Ldone

    v_lshlrev_b32 v2, 1, v1                 // * 2 bytes

    // Load dst
    v_mov_b32 v4, s5
    v_add_co_u32 v3, vcc, s4, v2
    v_addc_co_u32 v4, vcc, v4, 0, vcc
    global_load_ushort v5, v[3:4], off

    // Load src
    v_mov_b32 v7, s7
    v_add_co_u32 v6, vcc, s6, v2
    v_addc_co_u32 v7, vcc, v7, 0, vcc
    global_load_ushort v8, v[6:7], off

    s_waitcnt vmcnt(0)

    v_cvt_f32_f16 v5, v5
    v_cvt_f32_f16 v8, v8
    v_add_f32 v5, v5, v8
    v_cvt_f16_f32 v5, v5

    global_store_short v[3:4], v5, off

.Ldone:
    s_waitcnt vmcnt(0)
    s_endpgm
.Lfunc_end_residual_add:
    .size residual_add_fp16, .Lfunc_end_residual_add - residual_add_fp16

.rodata
.p2align 6
.amdhsa_kernel residual_add_fp16
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_private_segment_buffer 0
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_private_segment_wavefront_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 9
    .amdhsa_next_free_sgpr 10
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
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .max_flat_workgroup_size: 256
    .name:           residual_add_fp16
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         residual_add_fp16.kd
    .vgpr_count:     9
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
