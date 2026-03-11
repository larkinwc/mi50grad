// ISA Probe: v_fmac_f32 throughput on gfx906
// FP32 fused multiply-add-to-accumulator

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
.amdhsa_code_object_version 6

.globl probe_fmac_f32
.p2align 8
.type probe_fmac_f32,@function
probe_fmac_f32:
    s_load_dword s4, s[0:1], 0x0
    s_load_dwordx2 s[2:3], s[0:1], 0x8
    s_waitcnt lgkmcnt(0)

    v_mov_b32 v0, 0
    v_mov_b32 v1, 0
    v_mov_b32 v2, 0
    v_mov_b32 v3, 0
    v_mov_b32 v4, 0
    v_mov_b32 v5, 0
    v_mov_b32 v6, 0
    v_mov_b32 v7, 0

    v_mov_b32 v8, 0x3F800000   // 1.0f
    v_mov_b32 v9, 0x3F800000

    s_memrealtime s[6:7]
    s_waitcnt lgkmcnt(0)

    s_mov_b32 s8, s4

.Lloop_fmac:
    v_fmac_f32 v0, v8, v9
    v_fmac_f32 v1, v8, v9
    v_fmac_f32 v2, v8, v9
    v_fmac_f32 v3, v8, v9
    v_fmac_f32 v4, v8, v9
    v_fmac_f32 v5, v8, v9
    v_fmac_f32 v6, v8, v9
    v_fmac_f32 v7, v8, v9

    s_sub_u32 s8, s8, 1
    s_cbranch_scc0 .Lloop_fmac

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
.Lfunc_end_fmac:
    .size probe_fmac_f32, .Lfunc_end_fmac - probe_fmac_f32

// Kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel probe_fmac_f32
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
    .name:           probe_fmac_f32
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .symbol:         probe_fmac_f32.kd
    .vgpr_count:     32
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
