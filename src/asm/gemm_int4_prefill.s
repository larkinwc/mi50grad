// INT4 GEMM Prefill Kernel for gfx906 (MI50)
// C[M,N] = A[M,K] * dequant(B_q4[K/8,N])
//
// Tiled GEMM with on-the-fly INT4 dequantization for prefill (M > 1).
// Based on FP16 GEMM 64x64 tile pattern, but B is INT4 quantized (GPTQ).
//
// Tile: 64x64 output per workgroup, K-tiles of 16
// Grid: (ceil(N/64), ceil(M/64), 1), Block: (256, 1, 1)
// Thread blocking: 4x4 output elements per thread, 16 FP32 accumulators
//
// LDS layout (4096 bytes):
//   A tile: 64 rows x 16 cols, FP16, row-major = 2048 bytes [0..2047]
//   B tile: 16 rows x 64 cols, FP16, stored transposed as 64 cols x 16 k = 2048 bytes [2048..4095]
//     B is dequantized from INT4 to FP16 before writing to LDS.
//
// B matrix dequantization flow (per K-tile of 16 rows):
//   1. Load packed uint32 from B_q4 (8 INT4 values per uint32 along K)
//   2. Load scale/zero for the group
//   3. Unpack: extract 4-bit values with shift+mask
//   4. Dequant: w_fp16 = (int4_val - zero) * scale
//   5. Write FP16 to LDS B tile
//
// GPTQ format:
//   B_q4: uint32, [K/8, N] -- 8 INT4 values packed per uint32 along K
//   scales: FP16, [K/group_size, N]
//   zeros: FP16, [K/group_size, N]
//   group_size typically 128
//
// Kernarg:
//   [0:7]   ptr A       (FP16, [M, K])
//   [8:15]  ptr B_q4    (uint32, [K/8, N])
//   [16:23] ptr scales  (FP16, [K/group_size, N])
//   [24:31] ptr zeros   (FP16, [K/group_size, N])
//   [32:39] ptr C       (FP16, [M, N])
//   [40:43] uint32 M
//   [44:47] uint32 N
//   [48:51] uint32 K
//   [52:55] uint32 group_size

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
.amdhsa_code_object_version 6

.globl gemm_int4_prefill_64x64
.p2align 8
.type gemm_int4_prefill_64x64,@function
gemm_int4_prefill_64x64:
    // ============================================================
    // LOAD KERNARGS
    // ============================================================
    s_load_dwordx2 s[4:5], s[0:1], 0x0      // A ptr
    s_load_dwordx2 s[6:7], s[0:1], 0x8      // B_q4 ptr
    s_load_dwordx2 s[8:9], s[0:1], 0x10     // scales ptr
    s_load_dwordx2 s[10:11], s[0:1], 0x18   // zeros ptr
    s_load_dwordx2 s[12:13], s[0:1], 0x20   // C ptr
    s_load_dword s14, s[0:1], 0x28           // M
    s_load_dword s15, s[0:1], 0x2C           // N
    s_load_dword s16, s[0:1], 0x30           // K
    s_load_dword s17, s[0:1], 0x34           // group_size
    s_waitcnt lgkmcnt(0)

    // ============================================================
    // THREAD DECOMPOSITION
    // ============================================================
    // 256 threads, each computes 4x4 output elements
    // Thread layout: 16 threads in row-dim x 16 threads in col-dim
    // comp_row_base = (tid / 16) * 4   (rows 0,4,8,...,60)
    // comp_col_base = (tid % 16) * 4   (cols 0,4,8,...,60)

    // Compute position for output
    v_lshrrev_b32 v37, 4, v0              // comp_row_idx = tid / 16
    v_lshlrev_b32 v37, 2, v37             // comp_row_base = (tid/16) * 4
    v_and_b32 v38, 15, v0
    v_lshlrev_b32 v38, 2, v38             // comp_col_base = (tid%16) * 4

    // ============================================================
    // A TILE LOAD DECOMPOSITION (threads 0-127)
    // ============================================================
    // A tile: 64 rows x 16 cols FP16 = 2048 bytes
    // 128 threads, each loads 16 bytes (dwordx4 = 8 FP16)
    // load_a_row = tid / 2  (0..63)
    // load_a_col = (tid & 1) * 8  (0 or 8)
    v_lshrrev_b32 v33, 1, v0              // load_a_row = tid / 2
    v_and_b32 v34, 1, v0
    v_lshlrev_b32 v34, 3, v34             // load_a_col = (tid & 1) * 8

    // A LDS store addr: row * 32 + col * 2
    v_lshlrev_b32 v39, 5, v33             // load_a_row * 32
    v_lshlrev_b32 v40, 1, v34             // load_a_col * 2
    v_add_co_u32 v39, vcc, v39, v40       // v39 = A LDS store addr

    // ============================================================
    // B TILE LOAD DECOMPOSITION (threads 128-255)
    // ============================================================
    // B tile: 16 K-rows x 64 N-cols, but stored as INT4 packed.
    // B_q4 shape [K/8, N]: each uint32 = 8 INT4 values along K.
    // For K-tile of 16: need 16/8 = 2 uint32s per column.
    // Total: 2 * 64 = 128 uint32 loads. 128 threads -> 1 load each.
    //
    // Thread 128..255: local_tid = tid - 128 (0..127)
    // Each thread loads ONE uint32 from B_q4.
    // b_col = local_tid / 2  (0..63, covers 64 columns)
    // b_k_group = local_tid & 1  (0 or 1, two uint32s per column)
    //   k_group 0 = K values 0..7 within tile
    //   k_group 1 = K values 8..15 within tile
    s_movk_i32 s18, 128
    v_subrev_co_u32 v47, vcc, s18, v0     // local_tid = tid - 128
    v_lshrrev_b32 v35, 1, v47             // b_col = local_tid / 2
    v_and_b32 v36, 1, v47                 // b_k_group = local_tid & 1

    // ============================================================
    // WORKGROUP TILE ORIGINS
    // ============================================================
    s_lshl_b32 s19, s3, 6                 // m_base = wg_y * 64
    s_lshl_b32 s20, s2, 6                 // n_base = wg_x * 64

    // Useful strides
    s_lshl_b32 s21, s16, 1                // K_bytes = K * 2 (stride for A row in bytes)
    s_lshl_b32 s22, s15, 1                // N_bytes = N * 2
    s_lshl_b32 s23, s15, 2                // N_uint32_bytes = N * 4 (stride for B_q4 row)
    s_lshr_b32 s24, s16, 4                // k_iters = K / 16

    // ============================================================
    // COMPUTE A GLOBAL BASE ADDRESS
    // ============================================================
    // A_base += m_base * K * 2  (row-major, FP16)
    s_mul_i32 s25, s19, s16
    s_lshl_b32 s25, s25, 1
    s_add_u32 s4, s4, s25
    s_addc_u32 s5, s5, 0

    // ============================================================
    // COMPUTE B_q4 GLOBAL BASE ADDRESS
    // ============================================================
    // B_q4 is [K/8, N], row-major, uint32.
    // B_q4_base += n_base * 4  (offset to starting column)
    s_lshl_b32 s25, s20, 2
    s_add_u32 s6, s6, s25
    s_addc_u32 s7, s7, 0

    // ============================================================
    // COMPUTE SCALES/ZEROS GLOBAL BASE ADDRESSES
    // ============================================================
    // scales/zeros are [K/group_size, N], FP16
    // Offset by n_base columns: += n_base * 2
    s_lshl_b32 s25, s20, 1
    s_add_u32 s8, s8, s25
    s_addc_u32 s9, s9, 0
    s_add_u32 s10, s10, s25
    s_addc_u32 s11, s11, 0

    // ============================================================
    // COMPUTE C GLOBAL BASE ADDRESS
    // ============================================================
    // C_base += (m_base * N + n_base) * 2
    s_mul_i32 s25, s19, s15
    s_add_u32 s25, s25, s20
    s_lshl_b32 s25, s25, 1
    s_add_u32 s12, s12, s25
    s_addc_u32 s13, s13, 0

    // ============================================================
    // COMPUTE A GLOBAL LOAD ADDRESS (per thread)
    // ============================================================
    // A[m_base + load_a_row, k_tile_start + load_a_col]
    // = A_base + load_a_row * K * 2 + load_a_col * 2
    v_mul_lo_u32 v41, v33, s21             // load_a_row * K_bytes
    v_lshlrev_b32 v42, 1, v34             // load_a_col * 2
    v_add_co_u32 v41, vcc, v41, v42
    v_mov_b32 v42, s5
    v_add_co_u32 v41, vcc, s4, v41
    v_addc_co_u32 v42, vcc, v42, 0, vcc   // v[41:42] = A global addr

    // ============================================================
    // COMPUTE B_q4 GLOBAL LOAD ADDRESS (per thread, for threads 128-255)
    // ============================================================
    // B_q4[k_tile_start/8 + b_k_group, n_base + b_col]
    // = B_q4_base + b_k_group * N * 4 + b_col * 4
    v_mul_lo_u32 v43, v36, s23             // b_k_group * N_uint32_bytes
    v_lshlrev_b32 v44, 2, v35             // b_col * 4
    v_add_co_u32 v43, vcc, v43, v44
    v_mov_b32 v44, s7
    v_add_co_u32 v43, vcc, s6, v43
    v_addc_co_u32 v44, vcc, v44, 0, vcc   // v[43:44] = B_q4 global addr

    // ============================================================
    // COMPUTE SCALES/ZEROS GLOBAL LOAD ADDRESSES (per thread)
    // ============================================================
    // For thread's b_col: scales[k_tile_start / group_size, n_base + b_col]
    // We load scale and zero for b_col. The group index depends on k_tile_start.
    // scales_addr_base = scales_base + b_col * 2
    // zeros_addr_base = zeros_base + b_col * 2
    // (group row offset added per K-tile in the loop)
    v_lshlrev_b32 v48, 1, v35             // b_col * 2
    v_mov_b32 v46, s9
    v_add_co_u32 v45, vcc, s8, v48
    v_addc_co_u32 v46, vcc, v46, 0, vcc   // v[45:46] = scales_col_base
    v_mov_b32 v48, s11
    v_lshlrev_b32 v49, 1, v35             // b_col * 2
    v_add_co_u32 v47, vcc, s10, v49
    v_addc_co_u32 v48, vcc, v48, 0, vcc   // v[47:48] = zeros_col_base

    // ============================================================
    // INIT ACCUMULATORS (v1-v16)
    // ============================================================
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

    // ============================================================
    // B LDS STORE ADDRESS (for transposed layout)
    // ============================================================
    // B stored transposed: B_lds[col][k] at 2048 + col * 32 + k * 2
    // Thread stores 8 INT4 values unpacked to FP16 at:
    //   B_lds[b_col][b_k_group*8 + i] = 2048 + b_col * 32 + (b_k_group*8 + i) * 2
    //   base = 2048 + b_col * 32 + b_k_group * 16
    v_lshlrev_b32 v49, 5, v35             // b_col * 32
    v_lshlrev_b32 v50, 4, v36             // b_k_group * 16
    s_movk_i32 s25, 2048
    v_add_co_u32 v49, vcc, v49, v50
    v_add_co_u32 v49, vcc, v49, s25       // v49 = B LDS store base

    // ============================================================
    // LDS READ ADDRESSES (for compute phase)
    // ============================================================
    // A read base: comp_row_base * 32
    v_lshlrev_b32 v50, 5, v37             // comp_row_base * 32

    // B read base: 2048 + comp_col_base * 32
    v_lshlrev_b32 v51, 5, v38             // comp_col_base * 32
    v_add_co_u32 v51, vcc, v51, s25       // 2048 + comp_col_base * 32

    // ============================================================
    // K LOOP
    // ============================================================
    // For each K-tile of 16:
    //   1. Load A tile from global to LDS
    //   2. Load B_q4, dequant, write FP16 to LDS
    //   3. Barrier
    //   4. Compute from LDS
    //   5. Barrier, advance K

    // s24 = k_iters remaining
    // s26 = current k_tile_start (in elements), used for group index
    s_mov_b32 s26, 0                       // k_tile_start = 0

.Lk_loop_int4:
    // Save exec
    s_mov_b64 s[28:29], exec

    // ---- Load and store A (threads 0-127) ----
    s_movk_i32 s25, 128
    v_cmp_lt_u32 vcc, v0, s25
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Lskip_a_load_int4

    global_load_dwordx4 v[17:20], v[41:42], off
    s_waitcnt vmcnt(0)
    ds_write_b128 v39, v[17:20]

.Lskip_a_load_int4:
    // ---- Load B_q4, dequant, store to LDS (threads 128-255) ----
    s_mov_b64 exec, s[28:29]
    v_cmp_ge_u32 vcc, v0, s25
    s_and_b64 exec, exec, vcc
    s_cbranch_execz .Lskip_b_load_int4

    // Load one uint32 from B_q4 (8 packed INT4 values)
    global_load_dword v17, v[43:44], off

    // Compute group index for this thread's K range
    // k_abs = k_tile_start + b_k_group * 8
    // group_idx = k_abs / group_size
    // scales_addr = scales_col_base + group_idx * N * 2
    // For the scale/zero load, we need the row offset.
    v_lshlrev_b32 v52, 3, v36             // b_k_group * 8
    v_add_co_u32 v52, vcc, v52, s26       // k_abs = k_tile_start + b_k_group*8

    // Integer division: group_idx = k_abs / group_size
    // Use float approximation (safe for group_size power of 2)
    v_cvt_f32_u32 v53, v52                // k_abs as float
    v_cvt_f32_u32 v54, s17                // group_size as float
    v_rcp_f32 v54, v54
    v_mul_f32 v53, v53, v54
    v_cvt_u32_f32 v53, v53                // group_idx

    // scales_addr = scales_col_base + group_idx * N_bytes
    v_mul_lo_u32 v54, v53, s22            // group_idx * N_bytes
    v_add_co_u32 v55, vcc, v45, v54
    v_addc_co_u32 v56, vcc, v46, 0, vcc   // v[55:56] = scales addr

    // zeros_addr = zeros_col_base + group_idx * N_bytes
    v_add_co_u32 v57, vcc, v47, v54
    v_addc_co_u32 v58, vcc, v48, 0, vcc   // v[57:58] = zeros addr

    // Load scale and zero
    global_load_ushort v18, v[55:56], off  // scale (FP16)
    global_load_ushort v19, v[57:58], off  // zero (FP16)
    s_waitcnt vmcnt(0)

    // Convert scale/zero to FP32
    v_cvt_f32_f16 v18, v18                 // scale_f32
    v_cvt_f32_f16 v19, v19                 // zero_f32

    // Unpack 8 INT4 values from v17, dequant to FP16, write to LDS
    // B LDS store base = v49
    // Each FP16 value goes to v49 + i*2 (8 consecutive FP16 in the k-dimension)

    // val0 = v17 & 0xF
    v_and_b32 v20, 0xF, v17
    v_cvt_f32_u32 v20, v20
    v_sub_f32 v20, v20, v19
    v_mul_f32 v20, v20, v18
    v_cvt_f16_f32 v20, v20
    ds_write_b16 v49, v20 offset:0

    // val1 = (v17 >> 4) & 0xF
    v_lshrrev_b32 v20, 4, v17
    v_and_b32 v20, 0xF, v20
    v_cvt_f32_u32 v20, v20
    v_sub_f32 v20, v20, v19
    v_mul_f32 v20, v20, v18
    v_cvt_f16_f32 v20, v20
    ds_write_b16 v49, v20 offset:2

    // val2 = (v17 >> 8) & 0xF
    v_lshrrev_b32 v20, 8, v17
    v_and_b32 v20, 0xF, v20
    v_cvt_f32_u32 v20, v20
    v_sub_f32 v20, v20, v19
    v_mul_f32 v20, v20, v18
    v_cvt_f16_f32 v20, v20
    ds_write_b16 v49, v20 offset:4

    // val3 = (v17 >> 12) & 0xF
    v_lshrrev_b32 v20, 12, v17
    v_and_b32 v20, 0xF, v20
    v_cvt_f32_u32 v20, v20
    v_sub_f32 v20, v20, v19
    v_mul_f32 v20, v20, v18
    v_cvt_f16_f32 v20, v20
    ds_write_b16 v49, v20 offset:6

    // val4 = (v17 >> 16) & 0xF
    v_lshrrev_b32 v20, 16, v17
    v_and_b32 v20, 0xF, v20
    v_cvt_f32_u32 v20, v20
    v_sub_f32 v20, v20, v19
    v_mul_f32 v20, v20, v18
    v_cvt_f16_f32 v20, v20
    ds_write_b16 v49, v20 offset:8

    // val5 = (v17 >> 20) & 0xF
    v_lshrrev_b32 v20, 20, v17
    v_and_b32 v20, 0xF, v20
    v_cvt_f32_u32 v20, v20
    v_sub_f32 v20, v20, v19
    v_mul_f32 v20, v20, v18
    v_cvt_f16_f32 v20, v20
    ds_write_b16 v49, v20 offset:10

    // val6 = (v17 >> 24) & 0xF
    v_lshrrev_b32 v20, 24, v17
    v_and_b32 v20, 0xF, v20
    v_cvt_f32_u32 v20, v20
    v_sub_f32 v20, v20, v19
    v_mul_f32 v20, v20, v18
    v_cvt_f16_f32 v20, v20
    ds_write_b16 v49, v20 offset:12

    // val7 = v17 >> 28
    v_lshrrev_b32 v20, 28, v17
    v_cvt_f32_u32 v20, v20
    v_sub_f32 v20, v20, v19
    v_mul_f32 v20, v20, v18
    v_cvt_f16_f32 v20, v20
    ds_write_b16 v49, v20 offset:14

.Lskip_b_load_int4:
    // Restore exec
    s_mov_b64 exec, s[28:29]
    s_waitcnt lgkmcnt(0)
    s_barrier

    // ============================================================
    // COMPUTE PHASE: Read A and B from LDS, accumulate
    // ============================================================
    // A: row-major, A_lds[row][k] at row*32 + k*2
    //   ds_read_b32 at row*32 + k_pair*4 gives packed {A[row,2p], A[row,2p+1]}
    //   With XOR swizzle: addr = row*32 + (k_pair ^ (row & 7)) * 4
    //
    // B: transposed, B_lds[col][k] at 2048 + col*32 + k*2
    //   ds_read_b32 at 2048 + col*32 + k_pair*4 gives packed {B[2p,col], B[2p+1,col]}
    //   With XOR swizzle: addr = 2048 + col*32 + (k_pair ^ (col & 7)) * 4

    // Process 8 k-pairs via v_dot2_f32_f16
    .macro DOT2_KPAIR_INT4 kp
        // A read addresses (4 rows) - linear, no swizzle
        v_lshlrev_b32 v60, 2, \kp
        v_add_co_u32 v60, vcc, v50, v60          // A[row0] addr

        v_lshlrev_b32 v61, 2, \kp
        v_add_co_u32 v61, vcc, v50, v61
        v_add_co_u32 v61, vcc, v61, 32            // A[row1] addr

        v_lshlrev_b32 v62, 2, \kp
        v_add_co_u32 v62, vcc, v50, v62
        v_add_co_u32 v62, vcc, v62, 64            // A[row2] addr

        v_lshlrev_b32 v63, 2, \kp
        v_add_co_u32 v63, vcc, v50, v63
        s_movk_i32 s25, 96
        v_add_co_u32 v63, vcc, v63, s25           // A[row3] addr

        // B read addresses (4 cols) - linear, no swizzle
        v_lshlrev_b32 v64, 2, \kp
        v_add_co_u32 v64, vcc, v51, v64           // B[col0] addr

        v_lshlrev_b32 v65, 2, \kp
        v_add_co_u32 v65, vcc, v51, v65
        v_add_co_u32 v65, vcc, v65, 32            // B[col1] addr

        v_lshlrev_b32 v66, 2, \kp
        v_add_co_u32 v66, vcc, v51, v66
        v_add_co_u32 v66, vcc, v66, 64            // B[col2] addr

        v_lshlrev_b32 v67, 2, \kp
        v_add_co_u32 v67, vcc, v51, v67
        v_add_co_u32 v67, vcc, v67, s25           // B[col3] addr

        // Issue 8 LDS reads
        ds_read_b32 v21, v60                      // A[row0, kp*2:kp*2+1]
        ds_read_b32 v22, v61                      // A[row1, kp*2:kp*2+1]
        ds_read_b32 v23, v62                      // A[row2, kp*2:kp*2+1]
        ds_read_b32 v24, v63                      // A[row3, kp*2:kp*2+1]
        ds_read_b32 v25, v64                      // B[kp*2:kp*2+1, col0]
        ds_read_b32 v26, v65                      // B[kp*2:kp*2+1, col1]
        ds_read_b32 v27, v66                      // B[kp*2:kp*2+1, col2]
        ds_read_b32 v28, v67                      // B[kp*2:kp*2+1, col3]

        s_waitcnt lgkmcnt(0)

        // 16 v_dot2 for 4x4 outer product
        v_dot2_f32_f16 v1,  v21, v25, v1
        v_dot2_f32_f16 v2,  v21, v26, v2
        v_dot2_f32_f16 v3,  v21, v27, v3
        v_dot2_f32_f16 v4,  v21, v28, v4
        v_dot2_f32_f16 v5,  v22, v25, v5
        v_dot2_f32_f16 v6,  v22, v26, v6
        v_dot2_f32_f16 v7,  v22, v27, v7
        v_dot2_f32_f16 v8,  v22, v28, v8
        v_dot2_f32_f16 v9,  v23, v25, v9
        v_dot2_f32_f16 v10, v23, v26, v10
        v_dot2_f32_f16 v11, v23, v27, v11
        v_dot2_f32_f16 v12, v23, v28, v12
        v_dot2_f32_f16 v13, v24, v25, v13
        v_dot2_f32_f16 v14, v24, v26, v14
        v_dot2_f32_f16 v15, v24, v27, v15
        v_dot2_f32_f16 v16, v24, v28, v16
    .endm

    DOT2_KPAIR_INT4 0
    DOT2_KPAIR_INT4 1
    DOT2_KPAIR_INT4 2
    DOT2_KPAIR_INT4 3
    DOT2_KPAIR_INT4 4
    DOT2_KPAIR_INT4 5
    DOT2_KPAIR_INT4 6
    DOT2_KPAIR_INT4 7

    s_barrier

    // ============================================================
    // ADVANCE POINTERS FOR NEXT K-TILE
    // ============================================================
    // A: += 16 cols = 32 bytes
    v_add_co_u32 v41, vcc, v41, 32
    v_addc_co_u32 v42, vcc, v42, 0, vcc

    // B_q4: += 2 rows of [K/8, N] = 2 * N * 4 bytes
    // (16 K values = 2 uint32 rows in B_q4)
    s_lshl_b32 s25, s15, 3                // 2 * N * 4 = N << 3
    v_add_co_u32 v43, vcc, v43, s25
    v_addc_co_u32 v44, vcc, v44, 0, vcc

    // Advance k_tile_start
    s_add_u32 s26, s26, 16

    // Decrement K-tile counter
    s_sub_u32 s24, s24, 1
    s_cmp_gt_u32 s24, 0
    s_cbranch_scc1 .Lk_loop_int4

    // ============================================================
    // STORE C
    // ============================================================
    // Convert FP32 accumulators to FP16 pairs and store
    v_cvt_pkrtz_f16_f32 v21, v1, v2       // row0: [col0, col1]
    v_cvt_pkrtz_f16_f32 v22, v3, v4       // row0: [col2, col3]
    v_cvt_pkrtz_f16_f32 v23, v5, v6       // row1: [col0, col1]
    v_cvt_pkrtz_f16_f32 v24, v7, v8       // row1: [col2, col3]
    v_cvt_pkrtz_f16_f32 v25, v9, v10      // row2: [col0, col1]
    v_cvt_pkrtz_f16_f32 v26, v11, v12     // row2: [col2, col3]
    v_cvt_pkrtz_f16_f32 v27, v13, v14     // row3: [col0, col1]
    v_cvt_pkrtz_f16_f32 v28, v15, v16     // row3: [col2, col3]

    v_lshlrev_b32 v29, 1, v38             // comp_col_base * 2 (byte offset in row)

    // Row 0: C_base + comp_row_base * N * 2 + comp_col_base * 2
    v_mov_b32 v32, s13
    v_mul_lo_u32 v30, v37, s22            // comp_row_base * N_bytes
    v_add_co_u32 v30, vcc, v30, v29
    v_add_co_u32 v31, vcc, s12, v30
    v_addc_co_u32 v32, vcc, v32, 0, vcc
    global_store_dwordx2 v[31:32], v[21:22], off

    // Row 1
    v_mov_b32 v32, s13
    v_add_co_u32 v33, vcc, v37, 1
    v_mul_lo_u32 v30, v33, s22
    v_add_co_u32 v30, vcc, v30, v29
    v_add_co_u32 v31, vcc, s12, v30
    v_addc_co_u32 v32, vcc, v32, 0, vcc
    global_store_dwordx2 v[31:32], v[23:24], off

    // Row 2
    v_mov_b32 v32, s13
    v_add_co_u32 v33, vcc, v37, 2
    v_mul_lo_u32 v30, v33, s22
    v_add_co_u32 v30, vcc, v30, v29
    v_add_co_u32 v31, vcc, s12, v30
    v_addc_co_u32 v32, vcc, v32, 0, vcc
    global_store_dwordx2 v[31:32], v[25:26], off

    // Row 3
    v_mov_b32 v32, s13
    v_add_co_u32 v33, vcc, v37, 3
    v_mul_lo_u32 v30, v33, s22
    v_add_co_u32 v30, vcc, v30, v29
    v_add_co_u32 v31, vcc, s12, v30
    v_addc_co_u32 v32, vcc, v32, 0, vcc
    global_store_dwordx2 v[31:32], v[27:28], off

    s_waitcnt vmcnt(0)
    s_endpgm
.Lfunc_end_gemm_int4_prefill:
    .size gemm_int4_prefill_64x64, .Lfunc_end_gemm_int4_prefill - gemm_int4_prefill_64x64

.rodata
.p2align 6
.amdhsa_kernel gemm_int4_prefill_64x64
    .amdhsa_group_segment_fixed_size 4096
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
    .amdhsa_next_free_vgpr 68
    .amdhsa_next_free_sgpr 30
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
      - .offset: 52
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .max_flat_workgroup_size: 256
    .name:           gemm_int4_prefill_64x64
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .symbol:         gemm_int4_prefill_64x64.kd
    .vgpr_count:     68
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
