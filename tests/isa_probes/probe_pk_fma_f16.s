// ISA Probe: v_pk_fma_f16 throughput on gfx906
// This is the fallback compute instruction if dot instructions are broken.

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
.amdhsa_code_object_version 6

.globl probe_pk_fma_f16
.p2align 8
.type probe_pk_fma_f16,@function
probe_pk_fma_f16:
    s_load_dword s4, s[0:1], 0x0
    s_load_dwordx2 s[2:3], s[0:1], 0x8
    s_waitcnt lgkmcnt(0)

    // Init accumulators (packed FP16 pairs)
    v_mov_b32 v0, 0
    v_mov_b32 v1, 0
    v_mov_b32 v2, 0
    v_mov_b32 v3, 0
    v_mov_b32 v4, 0
    v_mov_b32 v5, 0
    v_mov_b32 v6, 0
    v_mov_b32 v7, 0

    v_mov_b32 v8, 0x3C003C00   // {1.0h, 1.0h}
    v_mov_b32 v9, 0x3C003C00

    s_memrealtime s[6:7]
    s_waitcnt lgkmcnt(0)

    s_mov_b32 s8, s4

.Lloop_pkfma:
    // v_pk_fma_f16: dst = src0 * src1 + src2 (packed 2xFP16)
    v_pk_fma_f16 v0, v8, v9, v0
    v_pk_fma_f16 v1, v8, v9, v1
    v_pk_fma_f16 v2, v8, v9, v2
    v_pk_fma_f16 v3, v8, v9, v3
    v_pk_fma_f16 v4, v8, v9, v4
    v_pk_fma_f16 v5, v8, v9, v5
    v_pk_fma_f16 v6, v8, v9, v6
    v_pk_fma_f16 v7, v8, v9, v7

    s_sub_u32 s8, s8, 1
    s_cbranch_scc0 .Lloop_pkfma

    s_memrealtime s[10:11]
    s_waitcnt lgkmcnt(0)
    s_sub_u32 s10, s10, s6
    s_subb_u32 s11, s11, s7

    v_readfirstlane_b32 s12, v0

    v_mov_b32 v10, s2
    v_mov_b32 v11, s3
    v_mov_b32 v12, s10
    v_mov_b32 v13, s11
    v_mov_b32 v14, s12

    global_store_dword v[10:11], v12, off offset:0
    global_store_dword v[10:11], v13, off offset:4
    global_store_dword v[10:11], v14, off offset:8

    s_waitcnt vmcnt(0)
    s_endpgm
.Lfunc_end_pkfma:
    .size probe_pk_fma_f16, .Lfunc_end_pkfma - probe_pk_fma_f16

// Kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel probe_pk_fma_f16
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_private_segment_buffer 0
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_private_segment_wavefront_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 32
    .amdhsa_next_free_sgpr 16
    .amdhsa_reserve_vcc 0
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
        .size: 4
        .value_kind: by_value
      - .offset: 4
        .size: 4
        .value_kind: by_value
      - .offset: 8
        .size: 8
        .value_kind: global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           probe_pk_fma_f16
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .symbol:         probe_pk_fma_f16.kd
    .vgpr_count:     32
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
