// FP16 GEMM kernel v3 for gfx906 (MI50)
// C[M,N] = A[M,K] * B[K,N]  (all row-major, FP16 in/out, FP32 accum)
//
// Tile: 64x64 output per workgroup, K-tiles of 16
// Workgroup = 256 threads, each thread computes 4x4 of C
//
// v3: software-pipelined LDS reads (pre-fetch k+1 while computing k)
// Uses v_fmac_f32 with individual FP16->FP32 conversion
//
// LDS: A(64x16) + B(16x64) = 4096 bytes (single buffer, reloaded each K-tile)
// Kernarg: [0:7] A, [8:15] B, [16:23] C, [24:27] M, [28:31] N, [32:35] K
// Grid: (N/64, M/64, 1), Block: (256, 1, 1)

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
.amdhsa_code_object_version 6

.globl gemm_fp16_64x64
.p2align 8
.type gemm_fp16_64x64,@function
gemm_fp16_64x64:
    // Load kernargs
    s_load_dwordx2 s[4:5], s[0:1], 0x0      // A
    s_load_dwordx2 s[6:7], s[0:1], 0x8      // B
    s_load_dwordx2 s[8:9], s[0:1], 0x10     // C
    s_load_dword s10, s[0:1], 0x18           // M
    s_load_dword s11, s[0:1], 0x1C           // N
    s_load_dword s12, s[0:1], 0x20           // K
    s_waitcnt lgkmcnt(0)

    // Thread decomposition
    v_lshrrev_b32 v33, 2, v0                // load_a_row = tid/4
    v_and_b32 v34, 3, v0
    v_lshlrev_b32 v34, 2, v34               // load_a_col = (tid%4)*4

    v_lshrrev_b32 v35, 4, v0                // load_b_row = tid/16
    v_and_b32 v36, 15, v0
    v_lshlrev_b32 v36, 2, v36               // load_b_col = (tid%16)*4

    // Workgroup tile origins
    s_lshl_b32 s14, s3, 6                   // m_base
    s_lshl_b32 s15, s2, 6                   // n_base

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

    // Init accumulators
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

    s_lshl_b32 s17, s12, 1                  // K_bytes
    s_lshl_b32 s18, s11, 1                  // N_bytes
    s_lshr_b32 s13, s12, 4                  // k_iters = K/16
    s_movk_i32 s19, 2048                     // B LDS offset

    // ============================================================
    // K LOOP
    // ============================================================
.Lk_loop:
    // ---- Load A tile (64x16) from global ----
    v_mul_lo_u32 v21, v33, s17
    v_lshlrev_b32 v24, 1, v34
    v_add_co_u32 v21, vcc, v21, v24
    v_mov_b32 v22, s5
    v_add_co_u32 v21, vcc, s4, v21
    v_addc_co_u32 v22, vcc, v22, 0, vcc
    global_load_dwordx2 v[17:18], v[21:22], off

    // ---- Load B tile (16x64) from global ----
    v_mul_lo_u32 v21, v35, s18
    v_lshlrev_b32 v24, 1, v36
    v_add_co_u32 v21, vcc, v21, v24
    v_mov_b32 v22, s7
    v_add_co_u32 v21, vcc, s6, v21
    v_addc_co_u32 v22, vcc, v22, 0, vcc
    global_load_dwordx2 v[19:20], v[21:22], off

    s_waitcnt vmcnt(0)

    // Store A to LDS (offset 0)
    v_lshlrev_b32 v23, 5, v33
    v_lshlrev_b32 v24, 1, v34
    v_add_co_u32 v23, vcc, v23, v24
    ds_write_b64 v23, v[17:18]

    // Store B to LDS (offset 2048)
    v_lshlrev_b32 v23, 7, v35
    v_lshlrev_b32 v24, 1, v36
    v_add_co_u32 v23, vcc, v23, v24
    v_add_co_u32 v23, vcc, v23, s19
    ds_write_b64 v23, v[19:20]

    s_waitcnt lgkmcnt(0)
    s_barrier

    // ---- Compute from LDS with software pipelining ----
    // Compute position
    // comp_row = (tid/16)*4, A_lds_base = comp_row * 32
    // comp_col = (tid%16)*4, B_lds_base = 2048 + comp_col * 2
    v_lshrrev_b32 v37, 4, v0
    v_lshlrev_b32 v37, 2, v37
    v_lshlrev_b32 v37, 5, v37               // A_lds_base = comp_row * 32

    v_and_b32 v38, 15, v0
    v_lshlrev_b32 v38, 2, v38
    v_lshlrev_b32 v38, 1, v38               // comp_col * 2
    v_add_co_u32 v38, vcc, v38, s19          // B_lds_base = 2048 + comp_col*2

    // Pre-fetch k=0: 4 A values + 4 B values
    ds_read_u16 v25, v37 offset:0            // A[row+0, 0]
    ds_read_u16 v26, v37 offset:32           // A[row+1, 0]
    ds_read_u16 v27, v37 offset:64           // A[row+2, 0]
    ds_read_u16 v28, v37 offset:96           // A[row+3, 0]
    ds_read_u16 v29, v38 offset:0            // B[0, col+0]
    ds_read_u16 v30, v38 offset:2            // B[0, col+1]
    ds_read_u16 v31, v38 offset:4            // B[0, col+2]
    ds_read_u16 v32, v38 offset:6            // B[0, col+3]

    // Software-pipelined k loop: k=0..14
    // For each k: wait for reads, start reads for k+1, then compute k
    .macro PIPE_K kk, next_kk
        s_waitcnt lgkmcnt(0)

        // Start reads for k+1 into alternate register set v[39:46]
        ds_read_u16 v39, v37 offset:(\next_kk*2)
        ds_read_u16 v40, v37 offset:(32 + \next_kk*2)
        ds_read_u16 v41, v37 offset:(64 + \next_kk*2)
        ds_read_u16 v42, v37 offset:(96 + \next_kk*2)
        ds_read_u16 v43, v38 offset:(\next_kk*128)
        ds_read_u16 v44, v38 offset:(\next_kk*128 + 2)
        ds_read_u16 v45, v38 offset:(\next_kk*128 + 4)
        ds_read_u16 v46, v38 offset:(\next_kk*128 + 6)

        // Compute k (data in v25-v32)
        v_cvt_f32_f16 v25, v25
        v_cvt_f32_f16 v26, v26
        v_cvt_f32_f16 v27, v27
        v_cvt_f32_f16 v28, v28
        v_cvt_f32_f16 v29, v29
        v_cvt_f32_f16 v30, v30
        v_cvt_f32_f16 v31, v31
        v_cvt_f32_f16 v32, v32

        v_fmac_f32 v1,  v25, v29
        v_fmac_f32 v2,  v25, v30
        v_fmac_f32 v3,  v25, v31
        v_fmac_f32 v4,  v25, v32
        v_fmac_f32 v5,  v26, v29
        v_fmac_f32 v6,  v26, v30
        v_fmac_f32 v7,  v26, v31
        v_fmac_f32 v8,  v26, v32
        v_fmac_f32 v9,  v27, v29
        v_fmac_f32 v10, v27, v30
        v_fmac_f32 v11, v27, v31
        v_fmac_f32 v12, v27, v32
        v_fmac_f32 v13, v28, v29
        v_fmac_f32 v14, v28, v30
        v_fmac_f32 v15, v28, v31
        v_fmac_f32 v16, v28, v32
    .endm

    .macro PIPE_K_SWAP kk, next_kk
        s_waitcnt lgkmcnt(0)

        // Start reads for k+1 into v25-v32
        ds_read_u16 v25, v37 offset:(\next_kk*2)
        ds_read_u16 v26, v37 offset:(32 + \next_kk*2)
        ds_read_u16 v27, v37 offset:(64 + \next_kk*2)
        ds_read_u16 v28, v37 offset:(96 + \next_kk*2)
        ds_read_u16 v29, v38 offset:(\next_kk*128)
        ds_read_u16 v30, v38 offset:(\next_kk*128 + 2)
        ds_read_u16 v31, v38 offset:(\next_kk*128 + 4)
        ds_read_u16 v32, v38 offset:(\next_kk*128 + 6)

        // Compute k (data in v39-v46)
        v_cvt_f32_f16 v39, v39
        v_cvt_f32_f16 v40, v40
        v_cvt_f32_f16 v41, v41
        v_cvt_f32_f16 v42, v42
        v_cvt_f32_f16 v43, v43
        v_cvt_f32_f16 v44, v44
        v_cvt_f32_f16 v45, v45
        v_cvt_f32_f16 v46, v46

        v_fmac_f32 v1,  v39, v43
        v_fmac_f32 v2,  v39, v44
        v_fmac_f32 v3,  v39, v45
        v_fmac_f32 v4,  v39, v46
        v_fmac_f32 v5,  v40, v43
        v_fmac_f32 v6,  v40, v44
        v_fmac_f32 v7,  v40, v45
        v_fmac_f32 v8,  v40, v46
        v_fmac_f32 v9,  v41, v43
        v_fmac_f32 v10, v41, v44
        v_fmac_f32 v11, v41, v45
        v_fmac_f32 v12, v41, v46
        v_fmac_f32 v13, v42, v43
        v_fmac_f32 v14, v42, v44
        v_fmac_f32 v15, v42, v45
        v_fmac_f32 v16, v42, v46
    .endm

    // k=0: data in v25-v32, prefetch k=1 into v39-v46
    PIPE_K 0, 1
    // k=1: data in v39-v46, prefetch k=2 into v25-v32
    PIPE_K_SWAP 1, 2
    PIPE_K 2, 3
    PIPE_K_SWAP 3, 4
    PIPE_K 4, 5
    PIPE_K_SWAP 5, 6
    PIPE_K 6, 7
    PIPE_K_SWAP 7, 8
    PIPE_K 8, 9
    PIPE_K_SWAP 9, 10
    PIPE_K 10, 11
    PIPE_K_SWAP 11, 12
    PIPE_K 12, 13
    PIPE_K_SWAP 13, 14
    PIPE_K 14, 15

    // k=15 (last): data in v39-v46, no prefetch needed
    s_waitcnt lgkmcnt(0)
    v_cvt_f32_f16 v39, v39
    v_cvt_f32_f16 v40, v40
    v_cvt_f32_f16 v41, v41
    v_cvt_f32_f16 v42, v42
    v_cvt_f32_f16 v43, v43
    v_cvt_f32_f16 v44, v44
    v_cvt_f32_f16 v45, v45
    v_cvt_f32_f16 v46, v46
    v_fmac_f32 v1,  v39, v43
    v_fmac_f32 v2,  v39, v44
    v_fmac_f32 v3,  v39, v45
    v_fmac_f32 v4,  v39, v46
    v_fmac_f32 v5,  v40, v43
    v_fmac_f32 v6,  v40, v44
    v_fmac_f32 v7,  v40, v45
    v_fmac_f32 v8,  v40, v46
    v_fmac_f32 v9,  v41, v43
    v_fmac_f32 v10, v41, v44
    v_fmac_f32 v11, v41, v45
    v_fmac_f32 v12, v41, v46
    v_fmac_f32 v13, v42, v43
    v_fmac_f32 v14, v42, v44
    v_fmac_f32 v15, v42, v45
    v_fmac_f32 v16, v42, v46

    s_barrier

    // Advance A += 16 cols (32 bytes)
    s_add_u32 s4, s4, 32
    s_addc_u32 s5, s5, 0
    // Advance B += 16 rows (16 * N * 2)
    s_lshl_b32 s16, s11, 5
    s_add_u32 s6, s6, s16
    s_addc_u32 s7, s7, 0

    s_sub_u32 s13, s13, 1
    s_cmp_gt_u32 s13, 0
    s_cbranch_scc1 .Lk_loop

    // ============================================================
    // STORE
    // ============================================================
    v_lshrrev_b32 v37, 4, v0
    v_lshlrev_b32 v37, 2, v37               // comp_row
    v_and_b32 v38, 15, v0
    v_lshlrev_b32 v38, 2, v38               // comp_col

    v_cvt_pkrtz_f16_f32 v39, v1, v2
    v_cvt_pkrtz_f16_f32 v40, v3, v4
    v_cvt_pkrtz_f16_f32 v41, v5, v6
    v_cvt_pkrtz_f16_f32 v42, v7, v8
    v_cvt_pkrtz_f16_f32 v43, v9, v10
    v_cvt_pkrtz_f16_f32 v44, v11, v12
    v_cvt_pkrtz_f16_f32 v45, v13, v14
    v_cvt_pkrtz_f16_f32 v46, v15, v16

    v_lshlrev_b32 v24, 1, v38               // comp_col * 2

    // Row 0
    v_mov_b32 v48, s9
    v_mul_lo_u32 v21, v37, s18
    v_add_co_u32 v21, vcc, v21, v24
    v_add_co_u32 v47, vcc, s8, v21
    v_addc_co_u32 v48, vcc, v48, 0, vcc
    global_store_dwordx2 v[47:48], v[39:40], off

    // Row 1
    v_mov_b32 v48, s9
    v_add_co_u32 v21, vcc, v37, 1
    v_mul_lo_u32 v21, v21, s18
    v_add_co_u32 v21, vcc, v21, v24
    v_add_co_u32 v47, vcc, s8, v21
    v_addc_co_u32 v48, vcc, v48, 0, vcc
    global_store_dwordx2 v[47:48], v[41:42], off

    // Row 2
    v_mov_b32 v48, s9
    v_add_co_u32 v21, vcc, v37, 2
    v_mul_lo_u32 v21, v21, s18
    v_add_co_u32 v21, vcc, v21, v24
    v_add_co_u32 v47, vcc, s8, v21
    v_addc_co_u32 v48, vcc, v48, 0, vcc
    global_store_dwordx2 v[47:48], v[43:44], off

    // Row 3
    v_mov_b32 v48, s9
    v_add_co_u32 v21, vcc, v37, 3
    v_mul_lo_u32 v21, v21, s18
    v_add_co_u32 v21, vcc, v21, v24
    v_add_co_u32 v47, vcc, s8, v21
    v_addc_co_u32 v48, vcc, v48, 0, vcc
    global_store_dwordx2 v[47:48], v[45:46], off

    s_waitcnt vmcnt(0)
    s_endpgm
.Lfunc_end_gemm_fp16:
    .size gemm_fp16_64x64, .Lfunc_end_gemm_fp16 - gemm_fp16_64x64

.rodata
.p2align 6
.amdhsa_kernel gemm_fp16_64x64
    .amdhsa_group_segment_fixed_size 4096
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
    .amdhsa_next_free_vgpr 49
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
        .size: 4
        .value_kind: by_value
      - .offset: 28
        .size: 4
        .value_kind: by_value
      - .offset: 32
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .max_flat_workgroup_size: 256
    .name:           gemm_fp16_64x64
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .symbol:         gemm_fp16_64x64.kd
    .vgpr_count:     49
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
