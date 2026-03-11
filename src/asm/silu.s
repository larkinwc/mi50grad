// SiLU (Swish) activation kernel for gfx906
// output[i] = input[i] * sigmoid(input[i])
// where sigmoid(x) = 1 / (1 + exp(-x))
//
// Also supports fused SiLU+multiply (for gated FFN):
//   output[i] = silu(gate[i]) * up[i]
//
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
//
// Kernarg:
//   [0:7]   ptr gate  (FP16)
//   [8:15]  ptr up    (FP16, or 0 for pure SiLU -> output = silu(gate))
//   [16:23] ptr output (FP16)
//   [24:27] uint32 n (number of elements)

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
.amdhsa_code_object_version 6

.globl silu_fp16
.p2align 8
.type silu_fp16,@function
silu_fp16:
    s_load_dwordx2 s[4:5], s[0:1], 0x0      // gate ptr
    s_load_dwordx2 s[6:7], s[0:1], 0x8      // up ptr
    s_load_dwordx2 s[8:9], s[0:1], 0x10     // output ptr
    s_load_dword s10, s[0:1], 0x18           // n
    s_waitcnt lgkmcnt(0)

    // Global index = wg_id_x * 256 + tid
    s_lshl_b32 s11, s2, 8                   // wg_id_x * 256
    v_add_co_u32 v1, vcc, s11, v0            // global_id

    // Bounds check
    v_cmp_lt_u32 vcc, v1, s10
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Lsilu_done

    // Compute byte offset
    v_lshlrev_b32 v2, 1, v1                 // * 2 (FP16)

    // Load gate value
    v_mov_b32 v4, s5
    v_add_co_u32 v3, vcc, s4, v2
    v_addc_co_u32 v4, vcc, v4, 0, vcc
    global_load_ushort v5, v[3:4], off
    s_waitcnt vmcnt(0)

    // Convert to FP32
    v_cvt_f32_f16 v5, v5

    // Compute sigmoid(x) = 1/(1+exp(-x))
    // exp(-x): negate, then v_exp_f32 computes 2^x, so exp(-x) = 2^(-x * log2(e))
    // log2(e) = 1.4426950408...
    v_mul_f32 v6, 0xBFB8AA3B, v5            // -x * log2(e) = -x * 1.44269504
    // Actually 0x3FB8AA3B = 1.44269504, negate: need -log2e
    // Let me use: v_mul_f32 v6, v5, literal then negate
    // -log2(e) as float: -1.44269504 = 0xBFB8AA3B
    v_exp_f32 v6, v6                         // 2^(-x*log2e) = exp(-x)
    v_add_f32 v6, 1.0, v6                   // 1 + exp(-x)
    v_rcp_f32 v6, v6                         // sigmoid = 1/(1+exp(-x))

    // silu(x) = x * sigmoid(x)
    v_mul_f32 v5, v5, v6

    // Check if up ptr is non-null (fused mode)
    s_cmp_eq_u64 s[6:7], 0
    s_cbranch_scc1 .Lsilu_store

    // Load up value and multiply
    v_mov_b32 v4, s7
    v_add_co_u32 v3, vcc, s6, v2
    v_addc_co_u32 v4, vcc, v4, 0, vcc
    global_load_ushort v7, v[3:4], off
    s_waitcnt vmcnt(0)
    v_cvt_f32_f16 v7, v7
    v_mul_f32 v5, v5, v7                     // silu(gate) * up

.Lsilu_store:
    // Convert back to FP16 and store
    v_cvt_f16_f32 v5, v5
    v_mov_b32 v4, s9
    v_add_co_u32 v3, vcc, s8, v2
    v_addc_co_u32 v4, vcc, v4, 0, vcc
    global_store_short v[3:4], v5, off

.Lsilu_done:
    s_waitcnt vmcnt(0)
    s_endpgm
.Lfunc_end_silu:
    .size silu_fp16, .Lfunc_end_silu - silu_fp16

.rodata
.p2align 6
.amdhsa_kernel silu_fp16
    .amdhsa_group_segment_fixed_size 0
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
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 12
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .max_flat_workgroup_size: 256
    .name:           silu_fp16
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         silu_fp16.kd
    .vgpr_count:     8
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
