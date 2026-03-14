// FP16 GEMM kernel v5 for gfx906 (MI50)
// C[M,N] = A[M,K] * B[K,N]  (all row-major, FP16 in/out, FP32 accum)
//
// Tile: 64x64 output per workgroup, K-tiles of 16
// Workgroup = 256 threads (4 wavefronts), each thread computes 4x4 of C
//
// Key optimization: stride-36 LDS padding to eliminate bank conflicts.
// LDS rows use 36-byte stride (32 data + 4 padding), ensuring that
// the 4 unique rows within a wavefront (spaced by 4) map to different banks.
//   Bank = (row * 9 + kp) % 32, so rows 0,4,8,12 → banks offset by 4.
//
// LDS layout (9216 bytes, double-buffered):
//   Buffer 0: A[0..2303] (64 rows x 36 bytes), B[2304..4607] (64 cols x 36 bytes)
//   Buffer 1: A[4608..6911], B[6912..9215]
//
// A tile: 64 rows x 16 cols FP16, row-major with stride 36
//   A_lds[row][k] at row*36 + k*2 (bytes 0-31 used, 32-35 padding)
// B tile: transposed, 64 cols x 16 k-vals FP16, col-major with stride 36
//   B_lds[col][k] at 2304 + col*36 + k*2
//
// Kernarg: [0:7] A, [8:15] B, [16:23] C, [24:27] M, [28:31] N, [32:35] K
// Grid: (N/64, M/64, 1), Block: (256, 1, 1)

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
.amdhsa_code_object_version 6

.globl gemm_fp16_64x64
.p2align 8
.type gemm_fp16_64x64,@function
gemm_fp16_64x64:
    // ========== LOAD KERNARGS ==========
    s_load_dwordx2 s[4:5], s[0:1], 0x0      // A ptr
    s_load_dwordx2 s[6:7], s[0:1], 0x8      // B ptr
    s_load_dwordx2 s[8:9], s[0:1], 0x10     // C ptr
    s_load_dword s10, s[0:1], 0x18           // M
    s_load_dword s11, s[0:1], 0x1C           // N
    s_load_dword s12, s[0:1], 0x20           // K
    s_waitcnt lgkmcnt(0)

    // ========== THREAD DECOMPOSITION ==========
    // 16x16 thread grid, each thread computes 4x4 output elements
    v_lshrrev_b32 v37, 4, v0                // tid / 16
    v_lshlrev_b32 v37, 2, v37               // comp_row_base = (tid/16)*4
    v_and_b32 v38, 15, v0
    v_lshlrev_b32 v38, 2, v38               // comp_col_base = (tid%16)*4

    // A load (threads 0-127): load_a_row=tid/2, load_a_col=(tid&1)*8
    v_lshrrev_b32 v33, 1, v0                // load_a_row
    v_and_b32 v34, 1, v0
    v_lshlrev_b32 v34, 3, v34               // load_a_col

    // B load (threads 128-255): local_tid=tid-128
    // load_b_row=local_tid/8 (0..15), load_b_col=(local_tid&7)*8 (0,8,..,56)
    s_movk_i32 s22, 128
    v_subrev_co_u32 v47, vcc, s22, v0       // local_tid
    v_lshrrev_b32 v35, 3, v47               // load_b_row
    v_and_b32 v36, 7, v47
    v_lshlrev_b32 v36, 3, v36               // load_b_col

    // ========== WORKGROUP TILE ORIGINS ==========
    s_lshl_b32 s14, s3, 6                   // m_base = wg_y * 64
    s_lshl_b32 s15, s2, 6                   // n_base = wg_x * 64

    // A_base += m_base * K * 2
    s_mul_i32 s16, s14, s12
    s_lshl_b32 s16, s16, 1
    s_add_u32 s4, s4, s16
    s_addc_u32 s5, s5, 0

    // B_base += n_base * 2
    s_lshl_b32 s16, s15, 1
    s_add_u32 s6, s6, s16
    s_addc_u32 s7, s7, 0

    // C_base += (m_base * N + n_base) * 2
    s_mul_i32 s16, s14, s11
    s_add_u32 s16, s16, s15
    s_lshl_b32 s16, s16, 1
    s_add_u32 s8, s8, s16
    s_addc_u32 s9, s9, 0

    // ========== INIT ACCUMULATORS v1-v16 ==========
    v_mov_b32 v1, 0
    v_mov_b32 v2, 0
    v_mov_b32 v3, 0
    v_mov_b32 v4, 0
    v_mov_b32 v5, 0
    v_mov_b32 v6, 0
    v_mov_b32 v7, 0
    v_mov_b32 v8, 0
    v_mov_b32 v9, 0
    v_mov_b32 v10, 0
    v_mov_b32 v11, 0
    v_mov_b32 v12, 0
    v_mov_b32 v13, 0
    v_mov_b32 v14, 0
    v_mov_b32 v15, 0
    v_mov_b32 v16, 0

    // ========== DERIVED CONSTANTS ==========
    s_lshl_b32 s17, s12, 1                  // K_bytes = K * 2
    s_lshl_b32 s18, s11, 1                  // N_bytes = N * 2
    s_lshr_b32 s13, s12, 4                  // k_iters = K / 16
    s_mov_b32 s20, 0                         // current buffer offset
    s_movk_i32 s21, 4608                     // alternate buffer offset

    // Stride-36 row offset constants (> 64, can't be inline)
    s_movk_i32 s14, 72                       // 2 * 36
    s_movk_i32 s15, 108                      // 3 * 36

    // ========== LDS STORE ADDRESSES (within-buffer, stride 36) ==========
    // A store: row * 36 + col * 2
    // x*36 = (x<<5) + (x<<2) = x*32 + x*4
    v_lshlrev_b32 v39, 5, v33               // load_a_row * 32
    v_lshlrev_b32 v48, 2, v33               // load_a_row * 4
    v_add_co_u32 v39, vcc, v39, v48         // load_a_row * 36
    v_lshlrev_b32 v48, 1, v34               // load_a_col * 2
    v_add_co_u32 v39, vcc, v39, v48         // v39 = A LDS store base

    // B store: 2304 + col * 36 + k * 2 (transposed layout)
    s_movk_i32 s22, 2304
    v_lshlrev_b32 v48, 5, v36               // load_b_col * 32
    v_lshlrev_b32 v47, 2, v36               // load_b_col * 4
    v_add_co_u32 v48, vcc, v48, v47         // load_b_col * 36
    v_lshlrev_b32 v47, 1, v35               // load_b_row * 2
    v_add_co_u32 v40, vcc, v48, v47
    v_add_co_u32 v40, vcc, v40, s22         // v40 = B LDS store base

    // ========== GLOBAL LOAD ADDRESSES ==========
    // A: A_base + load_a_row * K_bytes + load_a_col * 2
    v_mul_lo_u32 v41, v33, s17
    v_lshlrev_b32 v48, 1, v34
    v_add_co_u32 v41, vcc, v41, v48
    v_mov_b32 v42, s5
    v_add_co_u32 v41, vcc, s4, v41
    v_addc_co_u32 v42, vcc, v42, 0, vcc     // v[41:42] = A global addr

    // B: B_base + load_b_row * N_bytes + load_b_col * 2
    v_mul_lo_u32 v43, v35, s18
    v_lshlrev_b32 v48, 1, v36
    v_add_co_u32 v43, vcc, v43, v48
    v_mov_b32 v44, s7
    v_add_co_u32 v43, vcc, s6, v43
    v_addc_co_u32 v44, vcc, v44, 0, vcc     // v[43:44] = B global addr

    // ========== LDS READ BASES (within-buffer, stride 36) ==========
    // A: comp_row_base * 36
    v_lshlrev_b32 v45, 5, v37               // comp_row_base * 32
    v_lshlrev_b32 v48, 2, v37               // comp_row_base * 4
    v_add_co_u32 v45, vcc, v45, v48         // comp_row_base * 36

    // B: 2304 + comp_col_base * 36
    v_lshlrev_b32 v46, 5, v38               // comp_col_base * 32
    v_lshlrev_b32 v48, 2, v38               // comp_col_base * 4
    v_add_co_u32 v46, vcc, v46, v48         // comp_col_base * 36
    s_movk_i32 s22, 2304
    v_add_co_u32 v46, vcc, v46, s22         // 2304 + comp_col_base * 36

    // ========== PREFILL BUFFER 0 ==========
    s_mov_b64 s[24:25], exec
    s_movk_i32 s22, 128

    // Load + store A (threads 0-127)
    v_cmp_lt_u32 vcc, v0, s22
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Lskip_a_prefill

    global_load_dwordx4 v[17:20], v[41:42], off
    s_waitcnt vmcnt(0)
    // Store with stride-36: need 4 individual ds_write_b32
    // v[17:20] = 4 dwords = 8 FP16 values at k-pairs [kp_base..kp_base+3]
    // where kp_base = load_a_col/2 = 0 or 4
    // Store each dword at: row*36 + kp*4
    // v39 already includes row*36 + load_a_col*2 = row*36 + kp_base*4
    // So offsets from v39: 0, 4, 8, 12 for the 4 k-pairs
    ds_write_b32 v39, v17 offset:0
    ds_write_b32 v39, v18 offset:4
    ds_write_b32 v39, v19 offset:8
    ds_write_b32 v39, v20 offset:12

.Lskip_a_prefill:
    // Load + store B transposed (threads 128-255)
    s_mov_b64 exec, s[24:25]
    v_cmp_ge_u32 vcc, v0, s22
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Lskip_b_prefill

    global_load_dwordx4 v[17:20], v[43:44], off
    s_waitcnt vmcnt(0)

    // Scatter-write 8 FP16 transposed: each to a different column (stride 36)
    ds_write_b16 v40, v17 offset:0
    v_lshrrev_b32 v48, 16, v17
    ds_write_b16 v40, v48 offset:36
    ds_write_b16 v40, v18 offset:72
    v_lshrrev_b32 v48, 16, v18
    ds_write_b16 v40, v48 offset:108
    ds_write_b16 v40, v19 offset:144
    v_lshrrev_b32 v48, 16, v19
    ds_write_b16 v40, v48 offset:180
    ds_write_b16 v40, v20 offset:216
    v_lshrrev_b32 v48, 16, v20
    ds_write_b16 v40, v48 offset:252

.Lskip_b_prefill:
    s_mov_b64 exec, s[24:25]
    s_waitcnt lgkmcnt(0)
    s_barrier

    // Advance global load addresses past the first K-tile
    v_add_co_u32 v41, vcc, v41, 32          // A: +16 cols * 2 bytes
    v_addc_co_u32 v42, vcc, v42, 0, vcc
    s_lshl_b32 s22, s11, 5                  // B stride: 16 * N * 2 = N << 5
    v_add_co_u32 v43, vcc, v43, s22
    v_addc_co_u32 v44, vcc, v44, 0, vcc

    s_sub_u32 s13, s13, 1                   // consumed first K-tile

    // ========== MACRO: process one k-pair with v_dot2 ==========
    // Zero address overhead: uses precomputed row/col bases (v59-v66)
    // with ds_read_b32 offset field for k-pair indexing.
    // Interleaved reads/compute hides LDS latency.
    // \off = k-pair byte offset (0, 4, 8, ..., 28)
    .macro DOT2_KPAIR off
        // 8 reads using precomputed bases + offset
        ds_read_b32 v21, v59 offset:\off              // A[row0, kp]
        ds_read_b32 v25, v63 offset:\off              // B[col0, kp]
        ds_read_b32 v22, v60 offset:\off              // A[row1, kp]
        ds_read_b32 v26, v64 offset:\off              // B[col1, kp]
        ds_read_b32 v23, v61 offset:\off              // A[row2, kp]
        ds_read_b32 v27, v65 offset:\off              // B[col2, kp]
        ds_read_b32 v24, v62 offset:\off              // A[row3, kp]
        ds_read_b32 v28, v66 offset:\off              // B[col3, kp]

        // Compute as data arrives
        s_waitcnt lgkmcnt(6)
        v_dot2_f32_f16 v1,  v21, v25, v1

        s_waitcnt lgkmcnt(4)
        v_dot2_f32_f16 v2,  v21, v26, v2
        v_dot2_f32_f16 v5,  v22, v25, v5
        v_dot2_f32_f16 v6,  v22, v26, v6

        s_waitcnt lgkmcnt(2)
        v_dot2_f32_f16 v3,  v21, v27, v3
        v_dot2_f32_f16 v7,  v22, v27, v7
        v_dot2_f32_f16 v9,  v23, v25, v9
        v_dot2_f32_f16 v10, v23, v26, v10
        v_dot2_f32_f16 v11, v23, v27, v11

        s_waitcnt lgkmcnt(0)
        v_dot2_f32_f16 v4,  v21, v28, v4
        v_dot2_f32_f16 v8,  v22, v28, v8
        v_dot2_f32_f16 v12, v23, v28, v12
        v_dot2_f32_f16 v13, v24, v25, v13
        v_dot2_f32_f16 v14, v24, v26, v14
        v_dot2_f32_f16 v15, v24, v27, v15
        v_dot2_f32_f16 v16, v24, v28, v16
    .endm

    // ========== K LOOP (double-buffered) ==========
.Lk_loop:
    // ---- Async load next K-tile (if remaining) ----
    s_cmp_eq_u32 s13, 0
    s_cbranch_scc1 .Lskip_next_load

    s_mov_b64 s[24:25], exec
    s_movk_i32 s22, 128

    // A load (wavefronts 0-1)
    v_cmp_lt_u32 vcc, v0, s22
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Lskip_a_load
    global_load_dwordx4 v[17:20], v[41:42], off
.Lskip_a_load:

    // B load (wavefronts 2-3)
    s_mov_b64 exec, s[24:25]
    v_cmp_ge_u32 vcc, v0, s22
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Lskip_b_load
    global_load_dwordx4 v[17:20], v[43:44], off
.Lskip_b_load:

    s_mov_b64 exec, s[24:25]
.Lskip_next_load:

    // ---- Compute from current buffer ----
    // Precompute 8 row/col base addresses (once per K-tile, not per k-pair)
    v_add_co_u32 v49, vcc, v45, s20         // A row0 base + buffer offset
    v_add_co_u32 v59, vcc, v49, 0           // A[row0] base (= v49)
    v_add_co_u32 v60, vcc, v49, 36          // A[row1] base
    v_add_co_u32 v61, vcc, v49, s14         // A[row2] base (+72)
    v_add_co_u32 v62, vcc, v49, s15         // A[row3] base (+108)
    v_add_co_u32 v50, vcc, v46, s20         // B col0 base + buffer offset
    v_add_co_u32 v63, vcc, v50, 0           // B[col0] base (= v50)
    v_add_co_u32 v64, vcc, v50, 36          // B[col1] base
    v_add_co_u32 v65, vcc, v50, s14         // B[col2] base (+72)
    v_add_co_u32 v66, vcc, v50, s15         // B[col3] base (+108)

    DOT2_KPAIR 0
    DOT2_KPAIR 4
    DOT2_KPAIR 8
    DOT2_KPAIR 12
    DOT2_KPAIR 16
    DOT2_KPAIR 20
    DOT2_KPAIR 24
    DOT2_KPAIR 28

    // ---- Store next tile to alternate buffer (if loaded) ----
    s_cmp_eq_u32 s13, 0
    s_cbranch_scc1 .Ldone

    s_waitcnt vmcnt(0)
    s_mov_b64 s[24:25], exec
    s_movk_i32 s22, 128

    // Store A with stride-36 (wavefronts 0-1)
    v_cmp_lt_u32 vcc, v0, s22
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Lskip_a_store
    v_add_co_u32 v48, vcc, v39, s21
    ds_write_b32 v48, v17 offset:0
    ds_write_b32 v48, v18 offset:4
    ds_write_b32 v48, v19 offset:8
    ds_write_b32 v48, v20 offset:12
.Lskip_a_store:

    // Store B transposed with stride-36 (wavefronts 2-3)
    s_mov_b64 exec, s[24:25]
    v_cmp_ge_u32 vcc, v0, s22
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Lskip_b_store

    v_add_co_u32 v48, vcc, v40, s21
    ds_write_b16 v48, v17 offset:0
    v_lshrrev_b32 v59, 16, v17
    ds_write_b16 v48, v59 offset:36
    ds_write_b16 v48, v18 offset:72
    v_lshrrev_b32 v59, 16, v18
    ds_write_b16 v48, v59 offset:108
    ds_write_b16 v48, v19 offset:144
    v_lshrrev_b32 v59, 16, v19
    ds_write_b16 v48, v59 offset:180
    ds_write_b16 v48, v20 offset:216
    v_lshrrev_b32 v59, 16, v20
    ds_write_b16 v48, v59 offset:252
.Lskip_b_store:

    s_mov_b64 exec, s[24:25]
    s_waitcnt lgkmcnt(0)
    s_barrier

    // Advance global load addresses
    v_add_co_u32 v41, vcc, v41, 32
    v_addc_co_u32 v42, vcc, v42, 0, vcc
    s_lshl_b32 s22, s11, 5                  // N << 5
    v_add_co_u32 v43, vcc, v43, s22
    v_addc_co_u32 v44, vcc, v44, 0, vcc

    // Swap buffers
    s_xor_b32 s20, s20, s21
    s_xor_b32 s21, s21, s20
    s_xor_b32 s20, s20, s21

    s_sub_u32 s13, s13, 1
    s_branch .Lk_loop

    // ========== STORE C ==========
.Ldone:
    v_cvt_pkrtz_f16_f32 v21, v1, v2         // row0: {col0, col1}
    v_cvt_pkrtz_f16_f32 v22, v3, v4         // row0: {col2, col3}
    v_cvt_pkrtz_f16_f32 v23, v5, v6         // row1: {col0, col1}
    v_cvt_pkrtz_f16_f32 v24, v7, v8         // row1: {col2, col3}
    v_cvt_pkrtz_f16_f32 v25, v9, v10        // row2: {col0, col1}
    v_cvt_pkrtz_f16_f32 v26, v11, v12       // row2: {col2, col3}
    v_cvt_pkrtz_f16_f32 v27, v13, v14       // row3: {col0, col1}
    v_cvt_pkrtz_f16_f32 v28, v15, v16       // row3: {col2, col3}

    // C address: C_base + (comp_row + r) * N_bytes + comp_col * 2
    v_lshlrev_b32 v29, 1, v38               // comp_col_base * 2

    // Row 0
    v_mov_b32 v32, s9
    v_mul_lo_u32 v30, v37, s18
    v_add_co_u32 v30, vcc, v30, v29
    v_add_co_u32 v31, vcc, s8, v30
    v_addc_co_u32 v32, vcc, v32, 0, vcc
    global_store_dwordx2 v[31:32], v[21:22], off

    // Row 1
    v_mov_b32 v32, s9
    v_add_co_u32 v33, vcc, v37, 1
    v_mul_lo_u32 v30, v33, s18
    v_add_co_u32 v30, vcc, v30, v29
    v_add_co_u32 v31, vcc, s8, v30
    v_addc_co_u32 v32, vcc, v32, 0, vcc
    global_store_dwordx2 v[31:32], v[23:24], off

    // Row 2
    v_mov_b32 v32, s9
    v_add_co_u32 v33, vcc, v37, 2
    v_mul_lo_u32 v30, v33, s18
    v_add_co_u32 v30, vcc, v30, v29
    v_add_co_u32 v31, vcc, s8, v30
    v_addc_co_u32 v32, vcc, v32, 0, vcc
    global_store_dwordx2 v[31:32], v[25:26], off

    // Row 3
    v_mov_b32 v32, s9
    v_add_co_u32 v33, vcc, v37, 3
    v_mul_lo_u32 v30, v33, s18
    v_add_co_u32 v30, vcc, v30, v29
    v_add_co_u32 v31, vcc, s8, v30
    v_addc_co_u32 v32, vcc, v32, 0, vcc
    global_store_dwordx2 v[31:32], v[27:28], off

    s_waitcnt vmcnt(0)
    s_endpgm
.Lfunc_end_gemm_fp16:
    .size gemm_fp16_64x64, .Lfunc_end_gemm_fp16 - gemm_fp16_64x64

// ========== KERNEL DESCRIPTOR ==========
.rodata
.p2align 6
.amdhsa_kernel gemm_fp16_64x64
    .amdhsa_group_segment_fixed_size 9216
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 40
    .amdhsa_user_sgpr_private_segment_buffer 0
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_private_segment_wavefront_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 67
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

// ========== METADATA ==========
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
      - .offset: 32
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 9216
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .max_flat_workgroup_size: 256
    .name:           gemm_fp16_64x64
    .private_segment_fixed_size: 0
    .sgpr_count:     26
    .symbol:         gemm_fp16_64x64.kd
    .vgpr_count:     67
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
