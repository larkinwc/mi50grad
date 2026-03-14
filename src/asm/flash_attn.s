// FlashAttention prefill kernel for gfx906 (MI50)
// Multi-token attention: out = softmax(Q * K^T / scale) * V with online softmax
// Supports causal masking for autoregressive prefill.
//
// Design: 4 wavefronts per workgroup, each wavefront handles ONE Q row independently.
// Each wavefront of 64 threads covers head_dim=128 (2 dims per thread).
// Inner loop streams through all K/V positions with online softmax.
// This gives 4x throughput over decode_attn.s for prefill workloads.
//
// Grid: (num_heads, ceil(seq_len/4), 1), Block: (256, 1, 1)
//
// Kernarg:
//   [0:7]   ptr Q          (FP16, [seq_len, num_heads, head_dim])
//   [8:15]  ptr K          (FP16, [seq_len, num_kv_heads, head_dim])
//   [16:23] ptr V          (FP16, [seq_len, num_kv_heads, head_dim])
//   [24:31] ptr Out        (FP16, [seq_len, num_heads, head_dim])
//   [32:35] uint32 seq_len
//   [36:39] uint32 head_dim    (128)
//   [40:43] uint32 num_heads
//   [44:47] uint32 num_kv_heads
//   [48:51] float  scale       (1/sqrt(head_dim), as float32 bits)
//   [52:55] uint32 causal      (1 = causal masking, 0 = no masking)

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
.amdhsa_code_object_version 6

.globl flash_attn_fp16
.p2align 8
.type flash_attn_fp16,@function
flash_attn_fp16:
    // Load kernargs
    s_load_dwordx2 s[4:5], s[0:1], 0x0      // Q ptr
    s_load_dwordx2 s[6:7], s[0:1], 0x8      // K ptr
    s_load_dwordx2 s[8:9], s[0:1], 0x10     // V ptr
    s_load_dwordx2 s[10:11], s[0:1], 0x18   // Out ptr
    s_load_dword s12, s[0:1], 0x20           // seq_len
    s_load_dword s13, s[0:1], 0x24           // head_dim
    s_load_dword s14, s[0:1], 0x28           // num_heads
    s_load_dword s15, s[0:1], 0x2C           // num_kv_heads
    s_load_dword s16, s[0:1], 0x30           // scale (float bits)
    s_load_dword s17, s[0:1], 0x34           // causal
    s_waitcnt lgkmcnt(0)

    // Determine wavefront_id and lane_id from v0 (thread ID within workgroup)
    // wavefront_id = v0 >> 6, lane_id = v0 & 63
    v_lshrrev_b32 v1, 6, v0                  // wf_id
    v_and_b32 v2, 63, v0                      // lane_id

    // q_row = workgroup_y * 4 + wavefront_id
    // s3 = workgroup_id_y (q_block)
    v_readfirstlane_b32 s18, v1               // can't use v1 directly for SGPR math
    // Actually we need per-wavefront q_row. All lanes in a WF have the same wf_id.
    // Use v1 as VGPR, compute q_row in VGPR then readfirstlane.
    s_lshl_b32 s18, s3, 2                    // workgroup_y * 4
    v_add_u32 v1, s18, v1                     // q_row = workgroup_y * 4 + wf_id

    // Bounds check: if q_row >= seq_len, this wavefront is idle
    v_readfirstlane_b32 s18, v1               // q_row in SGPR for comparison
    s_cmp_ge_u32 s18, s12                     // q_row >= seq_len?
    s_cbranch_scc1 .Lflash_done

    // Store q_row in s18 for this wavefront
    // s18 = q_row (uniform within wavefront)

    // head = workgroup_id_x = s2
    // GQA: kv_head = head * num_kv_heads / num_heads
    s_mul_i32 s19, s2, s15                    // head * num_kv_heads
    v_mov_b32 v3, s19
    v_cvt_f32_u32 v4, s14                     // (float)num_heads
    v_rcp_f32 v4, v4
    v_cvt_f32_u32 v3, v3
    v_mul_f32 v3, v3, v4
    v_cvt_u32_f32 v3, v3
    v_readfirstlane_b32 s19, v3               // kv_head

    // dim_idx = lane_id * 2 (each thread handles 2 head dims)
    v_lshlrev_b32 v3, 1, v2                   // dim_idx = lane_id * 2

    // ---- Compute Q pointer for this wavefront's q_row ----
    // Q layout: [seq_len, num_heads, head_dim], row-major
    // Q_offset = (q_row * num_heads + head) * head_dim + dim_idx
    // In bytes: offset * 2 (FP16)
    s_mul_i32 s20, s14, s13                   // num_heads * head_dim (Q row stride in elements)
    s_mul_i32 s21, s18, s20                   // q_row * (num_heads * head_dim)
    s_mul_i32 s22, s2, s13                    // head * head_dim
    s_add_u32 s21, s21, s22                   // + head * head_dim
    s_lshl_b32 s21, s21, 1                    // * 2 bytes

    // Q_addr = Q_base + s21 + dim_idx * 2
    v_lshlrev_b32 v4, 1, v3                   // dim_idx * 2 (byte offset for this lane)
    v_mov_b32 v5, s5                           // Q_base_hi
    v_add_co_u32 v4, vcc, s4, v4              // Q_base_lo + dim_byte_offset
    v_addc_co_u32 v5, vcc, v5, 0, vcc
    // Add row offset (s21) - need to do 64-bit add with SGPR offset
    v_add_co_u32 v4, vcc, v4, s21
    v_addc_co_u32 v5, vcc, v5, 0, vcc

    // Load Q values (2 FP16 packed as dword)
    global_load_dword v6, v[4:5], off
    s_waitcnt vmcnt(0)

    // Unpack and convert to FP32, apply scale
    v_cvt_f32_f16 v7, v6                      // q_dim0 (FP32)
    v_lshrrev_b32 v8, 16, v6
    v_cvt_f32_f16 v8, v8                      // q_dim1 (FP32)

    v_mul_f32 v7, v7, s16                     // q_dim0 *= scale
    v_mul_f32 v8, v8, s16                     // q_dim1 *= scale

    // ---- Compute K/V base pointers ----
    // K/V layout: [seq_len, num_kv_heads, head_dim]
    // KV row stride = num_kv_heads * head_dim (elements)
    s_mul_i32 s20, s15, s13                   // kv_row_stride = num_kv_heads * head_dim
    s_lshl_b32 s23, s20, 1                    // kv_row_stride_bytes

    // KV head offset = kv_head * head_dim (elements), in bytes
    s_mul_i32 s24, s19, s13
    s_lshl_b32 s24, s24, 1                    // kv_head_offset_bytes

    // K_base_this_head = K_ptr + kv_head_offset_bytes
    s_add_u32 s25, s6, s24
    s_addc_u32 s26, s7, 0
    // V_base_this_head = V_ptr + kv_head_offset_bytes
    s_add_u32 s27, s8, s24
    s_addc_u32 s28, s9, 0

    // ---- Compute Out pointer ----
    // Out layout: same as Q: [seq_len, num_heads, head_dim]
    s_mul_i32 s20, s14, s13                   // num_heads * head_dim
    s_mul_i32 s21, s18, s20                   // q_row * (num_heads * head_dim)
    s_mul_i32 s22, s2, s13                    // head * head_dim
    s_add_u32 s21, s21, s22
    s_lshl_b32 s21, s21, 1                    // byte offset

    // ---- Initialize online softmax state ----
    // v9  = acc_dim0 (output accumulator, FP32)
    // v10 = acc_dim1
    // v11 = running_max (-inf)
    // v12 = running_sum (0)
    v_mov_b32 v9, 0                            // acc_dim0
    v_mov_b32 v10, 0                           // acc_dim1
    v_mov_b32 v11, 0xFF800000                  // running_max = -inf
    v_mov_b32 v12, 0                           // running_sum = 0

    // log2(e) for v_exp_f32 (which computes 2^x)
    s_mov_b32 s29, 0x3FB8AA3B                  // log2(e) = 1.44269504

    // -inf for causal masking
    s_mov_b32 s30, 0xFF800000                  // -inf

    // ---- KV loop: iterate over all K/V positions ----
    // s31 = kv_pos counter (0 to seq_len-1)
    s_mov_b32 s31, 0

    // Determine loop bound for causal masking
    // If causal: loop up to min(q_row, seq_len-1), i.e., kv_pos <= q_row
    // If not causal: loop up to seq_len-1
    // We'll use s20 as the loop limit (exclusive upper bound)
    s_cmp_eq_u32 s17, 0                        // causal == 0?
    s_cbranch_scc1 .Lset_full_range
    // Causal: limit = q_row + 1 (process positions 0..q_row)
    s_add_u32 s20, s18, 1
    // Clamp to seq_len
    s_min_u32 s20, s20, s12
    s_branch .Lloop_start
.Lset_full_range:
    s_mov_b32 s20, s12                         // limit = seq_len
.Lloop_start:
    // Check if we have any work
    s_cmp_ge_u32 s31, s20
    s_cbranch_scc1 .Lkv_loop_end

.Lkv_loop:
    // ---- Load K[kv_pos] for this head dim slice ----
    // K_addr = K_base_this_head + kv_pos * kv_row_stride_bytes + dim_idx * 2
    v_mov_b32 v13, s31
    v_mul_lo_u32 v13, v13, s23                 // kv_pos * kv_row_stride_bytes
    v_lshlrev_b32 v14, 1, v3                   // dim_idx * 2
    v_add_co_u32 v13, vcc, v13, v14
    v_mov_b32 v14, s26                         // K_base_hi
    v_add_co_u32 v13, vcc, s25, v13            // K_base_lo + offset
    v_addc_co_u32 v14, vcc, v14, 0, vcc

    global_load_dword v15, v[13:14], off
    s_waitcnt vmcnt(0)

    // Unpack K to FP32
    v_cvt_f32_f16 v16, v15                     // k_dim0
    v_lshrrev_b32 v17, 16, v15
    v_cvt_f32_f16 v17, v17                     // k_dim1

    // Partial dot product: q0*k0 + q1*k1
    v_mul_f32 v18, v7, v16
    v_fmac_f32 v18, v8, v17

    // DPP wavefront reduction for full dot product across 64 lanes
    // Step 1: reduce within each row of 16 lanes
    v_add_f32 v18, v18, v18 row_shr:1
    v_add_f32 v18, v18, v18 row_shr:2
    v_add_f32 v18, v18, v18 row_shr:4
    v_add_f32 v18, v18, v18 row_shr:8
    // Step 2: extract row sums via readlane, combine, broadcast to all lanes
    // After row_shr, lane 0 of each 16-lane row has the row sum
    v_readlane_b32 s0, v18, 0
    v_readlane_b32 s1, v18, 16
    v_readlane_b32 s2, v18, 32
    v_readlane_b32 s3, v18, 48
    v_mov_b32 v18, s0
    v_add_f32 v18, v18, s1
    v_add_f32 v18, v18, s2
    v_add_f32 v18, v18, s3
    // v18 = full QK^T dot product (score), broadcast to all lanes

    // ---- Online softmax update ----
    v_max_f32 v19, v11, v18                    // new_max = max(running_max, score)

    // correction = exp(old_max - new_max) = 2^((old_max - new_max) * log2(e))
    v_sub_f32 v20, v11, v19                    // old_max - new_max
    v_mul_f32 v20, v20, s29                    // * log2(e)
    v_exp_f32 v20, v20                         // correction = 2^(...)

    // p = exp(score - new_max) = 2^((score - new_max) * log2(e))
    v_sub_f32 v21, v18, v19                    // score - new_max
    v_mul_f32 v21, v21, s29                    // * log2(e)
    v_exp_f32 v21, v21                         // p

    // Update running sum: sum = sum * correction + p
    v_mul_f32 v12, v12, v20                    // sum *= correction
    v_add_f32 v12, v12, v21                    // sum += p

    // Update accumulators: acc *= correction
    v_mul_f32 v9, v9, v20
    v_mul_f32 v10, v10, v20

    // ---- Load V[kv_pos] and accumulate: acc += p * v ----
    v_mov_b32 v13, s31
    v_mul_lo_u32 v13, v13, s23                 // kv_pos * kv_row_stride_bytes
    v_lshlrev_b32 v14, 1, v3                   // dim_idx * 2
    v_add_co_u32 v13, vcc, v13, v14
    v_mov_b32 v14, s28                         // V_base_hi
    v_add_co_u32 v13, vcc, s27, v13            // V_base_lo + offset
    v_addc_co_u32 v14, vcc, v14, 0, vcc

    global_load_dword v15, v[13:14], off
    s_waitcnt vmcnt(0)

    v_cvt_f32_f16 v16, v15                     // v_dim0
    v_lshrrev_b32 v17, 16, v15
    v_cvt_f32_f16 v17, v17                     // v_dim1

    v_fmac_f32 v9, v21, v16                    // acc_dim0 += p * v_dim0
    v_fmac_f32 v10, v21, v17                   // acc_dim1 += p * v_dim1

    // Update running max
    v_mov_b32 v11, v19

    // Advance kv_pos
    s_add_u32 s31, s31, 1
    s_cmp_lt_u32 s31, s20
    s_cbranch_scc1 .Lkv_loop

.Lkv_loop_end:
    // ---- Normalize output: out = acc / sum ----
    // Handle edge case: if sum == 0 (no keys attended to, e.g., causal with q_row=0 having 1 key)
    // sum should always be > 0 if at least one key was attended to
    v_rcp_f32 v12, v12                         // 1/sum
    v_mul_f32 v9, v9, v12                      // acc_dim0 / sum
    v_mul_f32 v10, v10, v12                    // acc_dim1 / sum

    // Pack to FP16
    v_cvt_pkrtz_f16_f32 v6, v9, v10

    // ---- Store output ----
    // Out_addr = Out_base + s21 + dim_idx * 2
    v_lshlrev_b32 v4, 1, v3                   // dim_idx * 2
    v_mov_b32 v5, s11                          // Out_base_hi
    v_add_co_u32 v4, vcc, s10, v4             // Out_base_lo + dim_byte
    v_addc_co_u32 v5, vcc, v5, 0, vcc
    v_add_co_u32 v4, vcc, v4, s21             // + row offset
    v_addc_co_u32 v5, vcc, v5, 0, vcc

    global_store_dword v[4:5], v6, off

.Lflash_done:
    s_waitcnt vmcnt(0)
    s_endpgm
.Lfunc_end_flash_attn:
    .size flash_attn_fp16, .Lfunc_end_flash_attn - flash_attn_fp16

.rodata
.p2align 6
.amdhsa_kernel flash_attn_fp16
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 56
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
    .amdhsa_next_free_sgpr 32
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
      - .offset: 48
        .size: 4
        .value_kind: by_value
      - .offset: 52
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .max_flat_workgroup_size: 256
    .name:           flash_attn_fp16
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .symbol:         flash_attn_fp16.kd
    .vgpr_count:     22
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
