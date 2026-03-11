// RoPE (Rotary Position Embedding) kernel for gfx906
// Applies rotary position embedding to query/key vectors.
//
// For each position pos, dimension pair (2i, 2i+1):
//   theta = pos * freq[i]  where freq[i] = 1 / (base^(2i/dim))
//   x[2i]   = x[2i]  * cos(theta) - x[2i+1] * sin(theta)
//   x[2i+1] = x[2i]  * sin(theta) + x[2i+1] * cos(theta)
//
// We precompute freq table: freq[i] = 1/base^(2i/dim) for i in [0,dim/2)
// and pass cos/sin tables: cos_table[pos, i], sin_table[pos, i]
//
// For simplicity this kernel takes precomputed cos/sin tables.
//
// Grid: (num_tokens, num_heads, 1), Block: (head_dim/2, 1, 1)
// head_dim/2 must be <= 256
//
// Kernarg:
//   [0:7]   ptr x       (FP16, [num_tokens, num_heads, head_dim])
//   [8:15]  ptr cos_tab  (FP16, [num_tokens, head_dim/2])
//   [16:23] ptr sin_tab  (FP16, [num_tokens, head_dim/2])
//   [24:27] uint32 head_dim
//   [28:31] uint32 num_heads

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
.amdhsa_code_object_version 6

.globl rope_fp16
.p2align 8
.type rope_fp16,@function
rope_fp16:
    s_load_dwordx2 s[4:5], s[0:1], 0x0      // x ptr
    s_load_dwordx2 s[6:7], s[0:1], 0x8      // cos_tab
    s_load_dwordx2 s[8:9], s[0:1], 0x10     // sin_tab
    s_load_dword s10, s[0:1], 0x18           // head_dim
    s_load_dword s11, s[0:1], 0x1C           // num_heads
    s_waitcnt lgkmcnt(0)

    // wg_id_x = token index (position)
    // wg_id_y = head index
    // tid = dimension pair index i (processes x[2i] and x[2i+1])

    // Bounds check: tid < head_dim/2
    s_lshr_b32 s12, s10, 1                  // head_dim / 2
    v_cmp_lt_u32 vcc, v0, s12
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Lrope_done

    // Compute x offset: token * num_heads * head_dim + head * head_dim + 2*tid
    // All in FP16 bytes
    s_mul_i32 s13, s11, s10                  // num_heads * head_dim
    s_mul_i32 s14, s2, s13                   // token * num_heads * head_dim
    s_mul_i32 s15, s3, s10                   // head * head_dim
    s_add_u32 s14, s14, s15                 // + head * head_dim

    v_lshlrev_b32 v1, 1, v0                 // 2 * tid
    v_add_co_u32 v1, vcc, s14, v1            // + base offset
    v_lshlrev_b32 v1, 1, v1                 // * 2 (FP16 bytes)

    // x address
    v_mov_b32 v3, s5
    v_add_co_u32 v2, vcc, s4, v1
    v_addc_co_u32 v3, vcc, v3, 0, vcc

    // Load x[2i] and x[2i+1] as a dword (two packed FP16)
    global_load_dword v4, v[2:3], off
    // v4 = {x[2i+1], x[2i]} packed

    // cos/sin offset: token * (head_dim/2) + tid, in FP16 bytes
    s_mul_i32 s14, s2, s12                   // token * head_dim/2
    v_add_co_u32 v5, vcc, s14, v0            // + tid
    v_lshlrev_b32 v5, 1, v5                 // * 2 bytes

    // cos address
    v_mov_b32 v7, s7
    v_add_co_u32 v6, vcc, s6, v5
    v_addc_co_u32 v7, vcc, v7, 0, vcc
    global_load_ushort v8, v[6:7], off       // cos(theta)

    // sin address
    v_mov_b32 v7, s9
    v_add_co_u32 v6, vcc, s8, v5
    v_addc_co_u32 v7, vcc, v7, 0, vcc
    global_load_ushort v9, v[6:7], off       // sin(theta)

    s_waitcnt vmcnt(0)

    // Unpack x values
    v_cvt_f32_f16 v10, v4                    // x[2i] (low 16 bits)
    v_lshrrev_b32 v11, 16, v4               // extract high 16 bits
    v_cvt_f32_f16 v11, v11                   // x[2i+1]

    v_cvt_f32_f16 v8, v8                     // cos
    v_cvt_f32_f16 v9, v9                     // sin

    // Rotation:
    // new_x0 = x0 * cos - x1 * sin
    // new_x1 = x0 * sin + x1 * cos
    v_mul_f32 v12, v10, v8                   // x0 * cos
    v_mul_f32 v13, v11, v9                   // x1 * sin
    v_sub_f32 v12, v12, v13                  // new_x0 = x0*cos - x1*sin

    v_mul_f32 v14, v10, v9                   // x0 * sin
    v_fmac_f32 v14, v11, v8                  // + x1 * cos -> new_x1

    // Pack back to FP16 pair
    v_cvt_pkrtz_f16_f32 v4, v12, v14        // {new_x1, new_x0}

    // Store
    global_store_dword v[2:3], v4, off

.Lrope_done:
    s_waitcnt vmcnt(0)
    s_endpgm
.Lfunc_end_rope:
    .size rope_fp16, .Lfunc_end_rope - rope_fp16

.rodata
.p2align 6
.amdhsa_kernel rope_fp16
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_private_segment_buffer 0
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_private_segment_wavefront_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 16
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 256
    .name:           rope_fp16
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .symbol:         rope_fp16.kd
    .vgpr_count:     16
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
