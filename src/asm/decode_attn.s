// Decode Attention kernel for gfx906 (MI50)
// Single-query attention: out = softmax(Q * K^T / sqrt(d)) * V
//
// One wavefront (64 threads) per head. Each thread handles 2 dims of head_dim=128.
// Online softmax (streaming through K/V without materializing scores).
//
// Grid: (num_heads, batch_size, 1), Block: (256, 1, 1)
// Only first 64 threads active.
//
// Kernarg:
//   [0:7]   ptr Q      (FP16, [batch, num_heads, head_dim])
//   [8:15]  ptr K      (FP16, [batch, seq_len, num_kv_heads, head_dim])
//   [16:23] ptr V      (FP16, [batch, seq_len, num_kv_heads, head_dim])
//   [24:31] ptr out     (FP16, [batch, num_heads, head_dim])
//   [32:35] uint32 seq_len
//   [36:39] uint32 head_dim
//   [40:43] uint32 num_heads
//   [44:47] uint32 num_kv_heads

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
.amdhsa_code_object_version 6

.globl decode_attn_fp16
.p2align 8
.type decode_attn_fp16,@function
decode_attn_fp16:
    s_load_dwordx2 s[4:5], s[0:1], 0x0
    s_load_dwordx2 s[6:7], s[0:1], 0x8
    s_load_dwordx2 s[8:9], s[0:1], 0x10
    s_load_dwordx2 s[10:11], s[0:1], 0x18
    s_load_dword s12, s[0:1], 0x20
    s_load_dword s13, s[0:1], 0x24
    s_load_dword s14, s[0:1], 0x28
    s_load_dword s15, s[0:1], 0x2C
    s_waitcnt lgkmcnt(0)

    // Only first 64 threads
    v_cmp_lt_u32 vcc, v0, 64
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Lattn_done

    // GQA: kv_head = head * num_kv_heads / num_heads (via FP rcp)
    s_mul_i32 s16, s2, s15
    v_mov_b32 v10, s16
    v_cvt_f32_u32 v11, s14
    v_rcp_f32 v11, v11
    v_cvt_f32_u32 v10, v10
    v_mul_f32 v10, v10, v11
    v_cvt_u32_f32 v10, v10
    v_readfirstlane_b32 s16, v10

    // Q offset: batch * num_heads * head_dim + head * head_dim
    s_mul_i32 s17, s14, s13
    s_mul_i32 s18, s3, s17
    s_mul_i32 s19, s2, s13
    s_add_u32 s18, s18, s19
    s_lshl_b32 s18, s18, 1
    s_add_u32 s4, s4, s18
    s_addc_u32 s5, s5, 0
    s_add_u32 s10, s10, s18
    s_addc_u32 s11, s11, 0

    // KV offset: batch * seq_len * num_kv_heads * head_dim + kv_head * head_dim
    s_mul_i32 s17, s15, s13
    s_mul_i32 s18, s12, s17
    s_mul_i32 s18, s3, s18
    s_mul_i32 s19, s16, s13
    s_add_u32 s18, s18, s19
    s_lshl_b32 s18, s18, 1
    s_add_u32 s6, s6, s18
    s_addc_u32 s7, s7, 0
    s_add_u32 s8, s8, s18
    s_addc_u32 s9, s9, 0

    // KV stride per position (bytes)
    s_mul_i32 s17, s15, s13
    s_lshl_b32 s17, s17, 1

    // dim_idx = tid * 2
    v_lshlrev_b32 v1, 1, v0

    // Load Q
    v_lshlrev_b32 v2, 1, v1
    v_mov_b32 v3, s5
    v_add_co_u32 v2, vcc, s4, v2
    v_addc_co_u32 v3, vcc, v3, 0, vcc
    global_load_dword v4, v[2:3], off
    s_waitcnt vmcnt(0)

    v_cvt_f32_f16 v5, v4
    v_lshrrev_b32 v6, 16, v4
    v_cvt_f32_f16 v6, v6

    // Scale Q by 1/sqrt(head_dim)
    v_cvt_f32_u32 v7, s13
    v_rsq_f32 v7, v7
    v_mul_f32 v5, v5, v7
    v_mul_f32 v6, v6, v7

    // Online softmax state
    v_mov_b32 v8, 0                          // acc0
    v_mov_b32 v9, 0                          // acc1
    v_mov_b32 v10, 0xFF800000               // global_max = -inf
    v_mov_b32 v11, 0                         // global_sum

    // log2(e) constant in SGPR for literal-free multiply
    s_mov_b32 s18, 0x3FB8AA3B               // log2(e) = 1.44269504

    s_mov_b32 s19, 0                         // pos
    s_mov_b32 s16, s12                       // remaining

.Lkv_loop:
    // K addr = K_base + pos * kv_stride + dim_idx * 2
    v_mov_b32 v12, s19
    v_mul_lo_u32 v12, v12, s17
    v_lshlrev_b32 v13, 1, v1
    v_add_co_u32 v12, vcc, v12, v13
    v_mov_b32 v13, s7
    v_add_co_u32 v12, vcc, s6, v12
    v_addc_co_u32 v13, vcc, v13, 0, vcc
    global_load_dword v14, v[12:13], off
    s_waitcnt vmcnt(0)

    // Partial dot: q0*k0 + q1*k1
    v_cvt_f32_f16 v15, v14
    v_lshrrev_b32 v16, 16, v14
    v_cvt_f32_f16 v16, v16
    v_mul_f32 v17, v5, v15
    v_fmac_f32 v17, v6, v16

    // DPP wavefront reduction for full dot product
    v_add_f32 v17, v17, v17 row_shr:1
    v_add_f32 v17, v17, v17 row_shr:2
    v_add_f32 v17, v17, v17 row_shr:4
    v_add_f32 v17, v17, v17 row_shr:8
    // Lane 0 of each 16-lane row has the row sum; combine and broadcast
    v_readlane_b32 s0, v17, 0
    v_readlane_b32 s1, v17, 16
    v_readlane_b32 s2, v17, 32
    v_readlane_b32 s3, v17, 48
    v_mov_b32 v17, s0
    v_add_f32 v17, v17, s1
    v_add_f32 v17, v17, s2
    v_add_f32 v17, v17, s3
    // v17 = QK^T score, all lanes

    // Online softmax
    v_max_f32 v19, v10, v17                  // new_max

    v_sub_f32 v20, v10, v19                  // old_max - new_max
    v_mul_f32 v20, v20, s18                  // * log2(e)
    v_exp_f32 v20, v20                       // correction

    v_sub_f32 v21, v17, v19                  // score - new_max
    v_mul_f32 v21, v21, s18                  // * log2(e)
    v_exp_f32 v21, v21                       // p

    v_mul_f32 v11, v11, v20                  // sum *= correction
    v_add_f32 v11, v11, v21                  // sum += p

    v_mul_f32 v8, v8, v20                    // acc *= correction
    v_mul_f32 v9, v9, v20

    // V addr = V_base + pos * kv_stride + dim_idx * 2
    v_mov_b32 v12, s19
    v_mul_lo_u32 v12, v12, s17
    v_lshlrev_b32 v13, 1, v1
    v_add_co_u32 v12, vcc, v12, v13
    v_mov_b32 v13, s9
    v_add_co_u32 v12, vcc, s8, v12
    v_addc_co_u32 v13, vcc, v13, 0, vcc
    global_load_dword v14, v[12:13], off
    s_waitcnt vmcnt(0)

    v_cvt_f32_f16 v15, v14
    v_lshrrev_b32 v16, 16, v14
    v_cvt_f32_f16 v16, v16

    v_fmac_f32 v8, v21, v15                  // acc0 += p * v0
    v_fmac_f32 v9, v21, v16                  // acc1 += p * v1

    v_mov_b32 v10, v19                       // update max

    s_add_u32 s19, s19, 1
    s_sub_u32 s16, s16, 1
    s_cmp_gt_u32 s16, 0
    s_cbranch_scc1 .Lkv_loop

    // Normalize: out = acc / sum
    v_rcp_f32 v11, v11
    v_mul_f32 v8, v8, v11
    v_mul_f32 v9, v9, v11

    v_cvt_pkrtz_f16_f32 v4, v8, v9

    v_lshlrev_b32 v2, 1, v1
    v_mov_b32 v3, s11
    v_add_co_u32 v2, vcc, s10, v2
    v_addc_co_u32 v3, vcc, v3, 0, vcc
    global_store_dword v[2:3], v4, off

.Lattn_done:
    s_waitcnt vmcnt(0)
    s_endpgm
.Lfunc_end_decode_attn:
    .size decode_attn_fp16, .Lfunc_end_decode_attn - decode_attn_fp16

.rodata
.p2align 6
.amdhsa_kernel decode_attn_fp16
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 48
    .amdhsa_user_sgpr_private_segment_buffer 0
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_private_segment_wavefront_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 22
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
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 32
        .size: 4
        .value_kind: by_value
      - .offset: 36
        .size: 4
        .value_kind: by_value
      - .offset: 40
        .size: 4
        .value_kind: by_value
      - .offset: 44
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 256
    .name:           decode_attn_fp16
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .symbol:         decode_attn_fp16.kd
    .vgpr_count:     22
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
