// ISA Probe: v_dot2_f32_f16 throughput on gfx906
// Measures instructions/cycle by running a tight loop of v_dot2_f32_f16
// with no data dependencies between iterations (parallel chains).
//
// Build: /opt/rocm/llvm/bin/llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx906 -filetype=obj probe_dot2_f32_f16.s -o probe_dot2_f32_f16.o
//        /opt/rocm/llvm/bin/ld.lld -shared probe_dot2_f32_f16.o -o probe_dot2_f32_f16.hsaco
// Or use the HIP host wrapper to compile and launch.

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
.amdhsa_code_object_version 6

.globl probe_dot2_f32_f16
.p2align 8
.type probe_dot2_f32_f16,@function
probe_dot2_f32_f16:
    // Kernarg: s[0:1] = kernarg pointer
    // kernarg[0] = uint32 num_iters
    // kernarg[8] = uint64 output_ptr (for results)

    // Load kernarg
    s_load_dword s4, s[0:1], 0x0        // num_iters
    s_load_dwordx2 s[2:3], s[0:1], 0x8  // output_ptr
    s_waitcnt lgkmcnt(0)

    // Initialize accumulators (8 independent chains to saturate pipeline)
    v_mov_b32 v0, 0                      // acc0
    v_mov_b32 v1, 0                      // acc1
    v_mov_b32 v2, 0                      // acc2
    v_mov_b32 v3, 0                      // acc3
    v_mov_b32 v4, 0                      // acc4
    v_mov_b32 v5, 0                      // acc5
    v_mov_b32 v6, 0                      // acc6
    v_mov_b32 v7, 0                      // acc7

    // Initialize source operands (packed FP16 pairs)
    // v8 = {1.0h, 1.0h}, v9 = {1.0h, 1.0h}
    v_mov_b32 v8, 0x3C003C00            // {1.0h, 1.0h}
    v_mov_b32 v9, 0x3C003C00            // {1.0h, 1.0h}

    // Read start time
    s_memrealtime s[6:7]
    s_waitcnt lgkmcnt(0)

    // Loop counter
    s_mov_b32 s8, s4                     // s8 = num_iters

.Lloop_dot2:
    // 8 independent v_dot2_f32_f16 per iteration
    // Each: dst = src0[0]*src1[0] + src0[1]*src1[1] + dst
    v_dot2_f32_f16 v0, v8, v9, v0
    v_dot2_f32_f16 v1, v8, v9, v1
    v_dot2_f32_f16 v2, v8, v9, v2
    v_dot2_f32_f16 v3, v8, v9, v3
    v_dot2_f32_f16 v4, v8, v9, v4
    v_dot2_f32_f16 v5, v8, v9, v5
    v_dot2_f32_f16 v6, v8, v9, v6
    v_dot2_f32_f16 v7, v8, v9, v7

    s_sub_u32 s8, s8, 1
    s_cbranch_scc0 .Lloop_dot2

    // Read end time
    s_memrealtime s[10:11]
    s_waitcnt lgkmcnt(0)

    // Compute elapsed cycles: s[10:11] - s[6:7]
    s_sub_u32 s10, s10, s6
    s_subb_u32 s11, s11, s7

    // Only lane 0 writes result
    v_cmp_eq_u32 vcc, v0, v0            // dummy - we want lane 0
    v_cndmask_b32 v16, 0, 1, vcc
    v_readfirstlane_b32 s12, v0         // read acc0 to prevent DCE

    // Thread 0 of workgroup 0 writes output
    // output[0] = elapsed_lo, output[1] = elapsed_hi, output[2] = acc (prevent DCE)
    v_mov_b32 v10, s2                    // output_ptr lo
    v_mov_b32 v11, s3                    // output_ptr hi
    v_mov_b32 v12, s10                   // elapsed_lo
    v_mov_b32 v13, s11                   // elapsed_hi
    v_mov_b32 v14, s12                   // acc (anti-DCE)

    // Only thread 0 writes
    v_cmp_eq_u32 vcc, v0, v0
    s_and_b64 exec, exec, vcc

    global_store_dword v[10:11], v12, off offset:0
    global_store_dword v[10:11], v13, off offset:4
    global_store_dword v[10:11], v14, off offset:8

    s_waitcnt vmcnt(0)
    s_endpgm
.Lfunc_end_dot2:
    .size probe_dot2_f32_f16, .Lfunc_end_dot2 - probe_dot2_f32_f16

// Kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel probe_dot2_f32_f16
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
    .name:           probe_dot2_f32_f16
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .symbol:         probe_dot2_f32_f16.kd
    .vgpr_count:     32
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

    .end_amdgpu_metadata
