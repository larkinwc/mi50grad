// FP16 GEMV kernel for gfx906 (MI50)
// out[N] = W[N,K] * x[K]   (FP16 in/out, FP32 accum)
//
// One workgroup per output row (one element of N dimension).
// 256 threads reduce along K dimension via LDS.
// Each thread processes K/256 elements, partial sums reduced in LDS.
//
// Grid: (N, 1, 1), Block: (256, 1, 1)
//
// Kernarg:
//   [0:7]   ptr x      (FP16, [K])
//   [8:15]  ptr W      (FP16, [N, K], row-major)
//   [16:23] ptr out     (FP16, [N])
//   [24:27] uint32 K
//   [28:31] uint32 N

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
.amdhsa_code_object_version 6

.globl gemv_fp16
.p2align 8
.type gemv_fp16,@function
gemv_fp16:
    s_load_dwordx2 s[4:5], s[0:1], 0x0      // x ptr
    s_load_dwordx2 s[6:7], s[0:1], 0x8      // W ptr
    s_load_dwordx2 s[8:9], s[0:1], 0x10     // out ptr
    s_load_dword s10, s[0:1], 0x18           // K
    s_load_dword s11, s[0:1], 0x1C           // N
    s_waitcnt lgkmcnt(0)

    // row = workgroup_id_x = s2
    // W_row = W + row * K * 2
    s_mul_i32 s12, s2, s10                   // row * K
    s_lshl_b32 s12, s12, 1                   // * 2 bytes
    s_add_u32 s6, s6, s12
    s_addc_u32 s7, s7, 0

    // Each thread processes elements: tid, tid+256, tid+512, ...
    // Accumulate partial sum in FP32
    v_mov_b32 v1, 0                          // partial_sum = 0

    // Start index = tid
    v_mov_b32 v2, v0                         // k = tid
    s_mov_b32 s12, s10                       // K (for comparison)

.Lgemv_k_loop:
    // Check bounds: k < K
    v_cmp_lt_u32 vcc, v2, s12
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Lgemv_reduce

    // Load x[k]: x_addr = x + k * 2
    v_lshlrev_b32 v3, 1, v2                 // k * 2
    v_mov_b32 v4, s5
    v_add_co_u32 v3, vcc, s4, v3
    v_addc_co_u32 v4, vcc, v4, 0, vcc
    global_load_short_d16 v5, v[3:4], off    // x[k] as FP16 in low 16 bits

    // Load W[row, k]: w_addr = W_row + k * 2
    v_lshlrev_b32 v3, 1, v2
    v_mov_b32 v4, s7
    v_add_co_u32 v3, vcc, s6, v3
    v_addc_co_u32 v4, vcc, v4, 0, vcc
    global_load_short_d16 v6, v[3:4], off

    s_waitcnt vmcnt(0)

    // Accumulate: partial_sum += x[k] * W[row,k]
    v_cvt_f32_f16 v5, v5
    v_cvt_f32_f16 v6, v6
    v_fmac_f32 v1, v5, v6

    // k += 256
    s_movk_i32 s13, 256
    v_add_co_u32 v2, vcc, v2, s13
    s_branch .Lgemv_k_loop

.Lgemv_reduce:
    // Restore exec for all threads
    s_mov_b64 exec, -1

    // Reduce partial sums across 256 threads via LDS
    // Write partial sum to LDS
    v_lshlrev_b32 v3, 2, v0                 // lds_offset = tid * 4
    ds_write_b32 v3, v1
    s_waitcnt lgkmcnt(0)
    s_barrier

    // Tree reduction in LDS: 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
    // Each step: if tid < stride, load [tid] and [tid+stride], add, store back

    s_movk_i32 s13, 128
.Lgemv_reduce_loop:
    v_cmp_lt_u32 vcc, v0, s13
    s_and_b64 exec, exec, vcc

    // Load my value and partner's value
    v_lshlrev_b32 v3, 2, v0                 // my offset
    ds_read_b32 v4, v3
    v_add_co_u32 v5, vcc, v0, s13           // partner = tid + stride
    v_lshlrev_b32 v5, 2, v5                 // partner offset
    ds_read_b32 v6, v5
    s_waitcnt lgkmcnt(0)
    v_add_f32 v4, v4, v6
    ds_write_b32 v3, v4

    s_mov_b64 exec, -1
    s_waitcnt lgkmcnt(0)
    s_barrier

    s_lshr_b32 s13, s13, 1                  // stride /= 2
    s_cmp_gt_u32 s13, 0
    s_cbranch_scc1 .Lgemv_reduce_loop

    // Thread 0 writes final result
    v_cmp_eq_u32 vcc, v0, 0
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Lgemv_done

    // Load final sum from LDS[0]
    v_mov_b32 v3, 0
    ds_read_b32 v1, v3
    s_waitcnt lgkmcnt(0)

    // Convert to FP16 and store
    v_cvt_f16_f32 v1, v1

    // out[row] = result
    s_lshl_b32 s12, s2, 1                   // row * 2
    s_add_u32 s12, s8, s12                   // out + row * 2
    v_mov_b32 v2, s12
    v_mov_b32 v3, s9
    global_store_short v[2:3], v1, off

.Lgemv_done:
    s_mov_b64 exec, -1
    s_waitcnt vmcnt(0)
    s_endpgm
.Lfunc_end_gemv_fp16:
    .size gemv_fp16, .Lfunc_end_gemv_fp16 - gemv_fp16

.rodata
.p2align 6
.amdhsa_kernel gemv_fp16
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
    .amdhsa_next_free_vgpr 7
    .amdhsa_next_free_sgpr 14
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
    .name:           gemv_fp16
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .symbol:         gemv_fp16.kd
    .vgpr_count:     7
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
