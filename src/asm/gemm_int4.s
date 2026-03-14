// INT4 Weight-Only GEMM (GEMV decode path) for gfx906
// C[M,N] = A[M,K] * dequant(B_q4[K/8,N])
//
// A = FP16 activations (e.g. single token, M=1 for decode)
// B = INT4 quantized weights (GPTQ format: 4-bit packed, group_size=128)
// C = FP16 output
//
// GPTQ format:
//   B_q4: uint32 packed (8 INT4 values per uint32), shape [K/8, N]
//   scales: FP16, shape [K/group_size, N]
//   zeros: FP16, shape [K/group_size, N]
//   Dequant: w_fp16 = (q4_val - zero) * scale
//
// For decode (M=1): this is a GEMV, bandwidth-bound.
// Each workgroup computes one output element (one column of B).
// Grid: (N, 1, 1), Block: (256, 1, 1)
// Each thread processes K/256 chunks.
//
// Kernarg:
//   [0:7]   ptr A       (FP16, [K])
//   [8:15]  ptr B_q4    (uint32 packed, [K/8, N])
//   [16:23] ptr scales  (FP16, [K/group_size, N])
//   [24:31] ptr zeros   (FP16, [K/group_size, N])
//   [32:39] ptr C       (FP16, [N])
//   [40:43] uint32 K
//   [44:47] uint32 N
//   [48:51] uint32 group_size (typically 128)

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
.amdhsa_code_object_version 6

.globl gemv_int4_fp16
.p2align 8
.type gemv_int4_fp16,@function
gemv_int4_fp16:
    s_load_dwordx2 s[4:5], s[0:1], 0x0      // A
    s_load_dwordx2 s[6:7], s[0:1], 0x8      // B_q4
    s_load_dwordx2 s[8:9], s[0:1], 0x10     // scales
    s_load_dwordx2 s[10:11], s[0:1], 0x18   // zeros
    s_load_dwordx2 s[12:13], s[0:1], 0x20   // C
    s_load_dword s14, s[0:1], 0x28           // K
    s_load_dword s15, s[0:1], 0x2C           // N
    s_load_dword s16, s[0:1], 0x30           // group_size
    s_waitcnt lgkmcnt(0)

    // Column index = workgroup_id_x
    // Each workgroup computes C[col] = dot(A, dequant(B[:,col]))

    // B_q4 base for this column: B_q4_base + col * 4 (uint32 stride per column)
    // B_q4 is [K/8, N], so B_q4[k_group, col] = base + k_group * N * 4 + col * 4
    s_lshl_b32 s17, s2, 2                   // col * 4
    s_add_u32 s6, s6, s17
    s_addc_u32 s7, s7, 0

    // scales/zeros base for this column
    s_lshl_b32 s17, s2, 1                   // col * 2 (FP16)
    s_add_u32 s8, s8, s17
    s_addc_u32 s9, s9, 0
    s_add_u32 s10, s10, s17
    s_addc_u32 s11, s11, 0

    // C base for this column
    s_lshl_b32 s17, s2, 1
    s_add_u32 s12, s12, s17
    s_addc_u32 s13, s13, 0

    // Each thread processes K/256 groups of 8 INT4 values
    // k_start = tid * 8 (each thread handles 8 weights per iteration)
    // k_stride = 256 * 8 = 2048 (stride between iterations)
    // num_iters = K / (256 * 8) = K / 2048

    // B_q4 row stride = N * 4 bytes
    s_lshl_b32 s17, s15, 2                  // N_stride_b = N * 4

    // scales/zeros row stride = N * 2 bytes
    s_lshl_b32 s18, s15, 1                  // N_stride_s = N * 2

    // Thread's starting k_group index in B_q4: tid
    // (thread i processes B_q4[tid, col], then B_q4[tid+256, col], etc.)
    // B_q4 addr = B_q4_base + tid * N * 4
    v_mul_lo_u32 v1, v0, s17                 // tid * N_stride_b
    v_mov_b32 v2, s7
    v_add_co_u32 v1, vcc, s6, v1
    v_addc_co_u32 v2, vcc, v2, 0, vcc
    // v[1:2] = B_q4 addr for this thread's first k_group

    // A addr: A_base + tid * 8 * 2 (8 FP16 values per thread per iter)
    v_lshlrev_b32 v3, 4, v0                 // tid * 16 (= tid * 8 * 2 bytes)
    v_mov_b32 v4, s5
    v_add_co_u32 v3, vcc, s4, v3
    v_addc_co_u32 v4, vcc, v4, 0, vcc

    // scales addr: scales_base + (tid * 8 / group_size) * N * 2
    // group_index = tid * 8 / group_size
    v_lshlrev_b32 v5, 3, v0                 // tid * 8
    // For group_size = 128: group_idx = tid * 8 / 128 = tid / 16
    // General: need division. Use shift if power of 2.
    // group_size is typically 128 = 2^7, so tid*8 >> 7 = tid >> 4
    // But let's handle general case via FP
    v_cvt_f32_u32 v6, v5                     // tid*8 as float
    v_cvt_f32_u32 v7, s16                    // group_size as float
    v_rcp_f32 v7, v7
    v_mul_f32 v6, v6, v7                     // tid*8/group_size
    v_cvt_u32_f32 v6, v6                     // group_index (truncated)

    v_mul_lo_u32 v7, v6, s18                 // group_index * N_stride_s
    v_mov_b32 v8, s9
    v_add_co_u32 v7, vcc, s8, v7
    v_addc_co_u32 v8, vcc, v8, 0, vcc
    // v[7:8] = scales addr

    v_mul_lo_u32 v9, v6, s18                 // same for zeros
    v_mov_b32 v10, s11
    v_add_co_u32 v9, vcc, s10, v9
    v_addc_co_u32 v10, vcc, v10, 0, vcc

    // Accumulator (FP32)
    v_mov_b32 v20, 0

    // Number of iterations = ceil(K / 2048) = (K + 2047) / 2048
    s_add_u32 s19, s14, 2047
    s_lshr_b32 s19, s19, 11                 // ceil(K / 2048)

    // Track current k_base for bounds checking
    // k_base = tid * 8 (per thread starting K index)
    v_lshlrev_b32 v21, 3, v0                // tid * 8

    // Strides for next iteration
    // B_q4: next iter is +256 rows -> + 256 * N * 4
    s_mul_i32 s20, s17, 256                  // b_stride_iter = 256 * N_stride_b

    // A: next iter is +2048 elements -> +2048 * 2 = +4096 bytes
    s_mov_b32 s21, 4096

    s_cmp_eq_u32 s19, 0
    s_cbranch_scc1 .Lgemv_reduce

.Lgemv_loop:
    // Bounds check: skip if this thread's k_base >= K
    v_cmp_lt_u32 vcc, v21, s14              // k_base < K ?
    s_and_saveexec_b64 s[24:25], vcc
    s_cbranch_execz .Lgemv_skip_iter

    // Load B_q4[k_group] = one uint32 = 8 packed INT4 values
    global_load_dword v11, v[1:2], off
    // Load A[k*8 .. k*8+7] = 4 dwords = 8 FP16
    global_load_dwordx4 v[12:15], v[3:4], off
    // Load scale and zero for this group
    global_load_ushort v16, v[7:8], off      // scale
    global_load_ushort v17, v[9:10], off     // zero
    s_waitcnt vmcnt(0)

    // Convert scale/zero to FP32
    v_cvt_f32_f16 v16, v16                   // scale
    v_cvt_f32_f16 v17, v17                   // zero

    // Unpack 8 INT4 values from v11 and dequantize
    // INT4 values are packed little-endian: bits [3:0] = val0, [7:4] = val1, etc.
    // Dequant: fp_val = (int4_val - zero) * scale

    // Extract and process each INT4 value paired with its FP16 activation
    // val0 = v11 & 0xF
    v_and_b32 v18, 0xF, v11
    v_cvt_f32_u32 v18, v18
    v_sub_f32 v18, v18, v17                  // - zero
    v_mul_f32 v18, v18, v16                  // * scale
    // A[0] = low 16 bits of v12
    v_cvt_f32_f16 v19, v12
    v_fmac_f32 v20, v18, v19                 // acc += w * a

    // val1 = (v11 >> 4) & 0xF
    v_lshrrev_b32 v18, 4, v11
    v_and_b32 v18, 0xF, v18
    v_cvt_f32_u32 v18, v18
    v_sub_f32 v18, v18, v17
    v_mul_f32 v18, v18, v16
    v_lshrrev_b32 v19, 16, v12              // A[1] = high 16 bits of v12
    v_cvt_f32_f16 v19, v19
    v_fmac_f32 v20, v18, v19

    // val2
    v_lshrrev_b32 v18, 8, v11
    v_and_b32 v18, 0xF, v18
    v_cvt_f32_u32 v18, v18
    v_sub_f32 v18, v18, v17
    v_mul_f32 v18, v18, v16
    v_cvt_f32_f16 v19, v13                   // A[2]
    v_fmac_f32 v20, v18, v19

    // val3
    v_lshrrev_b32 v18, 12, v11
    v_and_b32 v18, 0xF, v18
    v_cvt_f32_u32 v18, v18
    v_sub_f32 v18, v18, v17
    v_mul_f32 v18, v18, v16
    v_lshrrev_b32 v19, 16, v13              // A[3]
    v_cvt_f32_f16 v19, v19
    v_fmac_f32 v20, v18, v19

    // val4
    v_lshrrev_b32 v18, 16, v11
    v_and_b32 v18, 0xF, v18
    v_cvt_f32_u32 v18, v18
    v_sub_f32 v18, v18, v17
    v_mul_f32 v18, v18, v16
    v_cvt_f32_f16 v19, v14                   // A[4]
    v_fmac_f32 v20, v18, v19

    // val5
    v_lshrrev_b32 v18, 20, v11
    v_and_b32 v18, 0xF, v18
    v_cvt_f32_u32 v18, v18
    v_sub_f32 v18, v18, v17
    v_mul_f32 v18, v18, v16
    v_lshrrev_b32 v19, 16, v14              // A[5]
    v_cvt_f32_f16 v19, v19
    v_fmac_f32 v20, v18, v19

    // val6
    v_lshrrev_b32 v18, 24, v11
    v_and_b32 v18, 0xF, v18
    v_cvt_f32_u32 v18, v18
    v_sub_f32 v18, v18, v17
    v_mul_f32 v18, v18, v16
    v_cvt_f32_f16 v19, v15                   // A[6]
    v_fmac_f32 v20, v18, v19

    // val7
    v_lshrrev_b32 v18, 28, v11
    v_cvt_f32_u32 v18, v18                   // top 4 bits, no mask needed
    v_sub_f32 v18, v18, v17
    v_mul_f32 v18, v18, v16
    v_lshrrev_b32 v19, 16, v15              // A[7]
    v_cvt_f32_f16 v19, v19
    v_fmac_f32 v20, v18, v19

    // Advance pointers
    // B_q4 += 256 * N * 4 (next k-group for this thread)
    v_add_co_u32 v1, vcc, v1, s20
    v_addc_co_u32 v2, vcc, v2, 0, vcc
    // A += 256 * 8 * 2 = 4096 bytes
    v_add_co_u32 v3, vcc, v3, s21
    v_addc_co_u32 v4, vcc, v4, 0, vcc

    // Advance scale/zero pointers: skip 2048/128 = 16 groups per iter
    s_lshl_b32 s22, s18, 4                  // 16 * N_stride_s
    v_add_co_u32 v7, vcc, v7, s22
    v_addc_co_u32 v8, vcc, v8, 0, vcc
    v_add_co_u32 v9, vcc, v9, s22
    v_addc_co_u32 v10, vcc, v10, 0, vcc

.Lgemv_skip_iter:
    // Restore exec mask (all threads participate in control flow)
    s_mov_b64 exec, s[24:25]

    // Advance k_base for bounds checking (2048 > inline range, use s_mov)
    s_movk_i32 s22, 2048
    v_add_u32 v21, s22, v21                // k_base += 2048

    s_sub_u32 s19, s19, 1
    s_cmp_gt_u32 s19, 0
    s_cbranch_scc1 .Lgemv_loop

.Lgemv_reduce:
    // Reduce partial sums across 256 threads via LDS
    v_lshlrev_b32 v1, 2, v0                 // tid * 4
    ds_write_b32 v1, v20
    s_waitcnt lgkmcnt(0)
    s_barrier

    // Tree reduction (same pattern as rmsnorm)
    .macro GEMV_REDUCE n, byte_offset
        s_mov_b64 s[22:23], exec
        .if \n > 64
            s_movk_i32 s24, \n
            v_cmp_lt_u32 vcc, v0, s24
        .else
            v_cmp_lt_u32 vcc, v0, \n
        .endif
        s_and_b64 exec, exec, vcc
        ds_read_b32 v18, v1 offset:0
        ds_read_b32 v19, v1 offset:\byte_offset
        s_waitcnt lgkmcnt(0)
        v_add_f32 v18, v18, v19
        ds_write_b32 v1, v18
        s_waitcnt lgkmcnt(0)
        s_mov_b64 exec, s[22:23]
        s_barrier
    .endm

    GEMV_REDUCE 128, 512
    GEMV_REDUCE 64,  256
    GEMV_REDUCE 32,  128
    GEMV_REDUCE 16,  64
    GEMV_REDUCE 8,   32
    GEMV_REDUCE 4,   16
    GEMV_REDUCE 2,   8
    GEMV_REDUCE 1,   4

    // Thread 0 reads final sum and stores to C
    v_cmp_eq_u32 vcc, 0, v0
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Lgemv_done

    v_mov_b32 v1, 0
    ds_read_b32 v20, v1
    s_waitcnt lgkmcnt(0)

    // Convert to FP16 and store
    v_cvt_f16_f32 v20, v20

    v_mov_b32 v2, s13
    v_mov_b32 v1, s12
    // v[1:2] = C addr (already computed above for this column)
    global_store_short v[1:2], v20, off

.Lgemv_done:
    s_waitcnt vmcnt(0)
    s_endpgm
.Lfunc_end_gemv_int4:
    .size gemv_int4_fp16, .Lfunc_end_gemv_int4 - gemv_int4_fp16

.rodata
.p2align 6
.amdhsa_kernel gemv_int4_fp16
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 56
    .amdhsa_user_sgpr_private_segment_buffer 0
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_private_segment_wavefront_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 22
    .amdhsa_next_free_sgpr 26
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
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 40
        .size: 4
        .value_kind: by_value
      - .offset: 44
        .size: 4
        .value_kind: by_value
      - .offset: 48
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 52
    .max_flat_workgroup_size: 256
    .name:           gemv_int4_fp16
    .private_segment_fixed_size: 0
    .sgpr_count:     26
    .symbol:         gemv_int4_fp16.kd
    .vgpr_count:     22
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
