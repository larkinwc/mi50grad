#!/usr/bin/env python3
"""
Test: DeltaNet v3 GPU path is always taken for linear attention layers.

Verifies:
1. _deltanet_gpu is always True after engine init (kernel loads or raises error)
2. decode_step never calls _decode_linear_attention (CPU fallback)
3. No D2H/H2D transfers occur during linear attention decode (all-GPU path)
4. If DeltaNet v3 kernel is missing, engine raises RuntimeError at init time
"""

import sys
import ctypes
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import QwenConfig


def make_default_config():
    """Use the default Qwen3.5 27B config (matching what deltanet_v3.hip expects)."""
    cfg = QwenConfig()
    # Use the default full model dimensions — deltanet_v3 is compiled for NUM_V_HEADS=48
    # We keep 2 layers only to speed up the test
    cfg.num_hidden_layers = 2
    # Layer pattern: linear, linear (avoid full attention to skip flash_attn_256)
    cfg.layer_types = ["linear_attention", "linear_attention"]
    return cfg


def test_deltanet_gpu_flag_true():
    """After engine init, _deltanet_gpu must always be True.

    The field must never be False when decode is called — the GPU path is mandatory.
    """
    from src.inference.engine import InferenceEngine
    cfg = make_default_config()
    engine = InferenceEngine(cfg, device_id=0, max_seq_len=32)
    assert engine._deltanet_gpu is True, (
        f"Expected _deltanet_gpu=True after init, got {engine._deltanet_gpu}. "
        "The CPU fallback has been removed; GPU path is mandatory."
    )
    assert engine._deltanet_v3 is True, (
        f"Expected _deltanet_v3=True after init, got {engine._deltanet_v3}."
    )
    print(f"  _deltanet_gpu={engine._deltanet_gpu}, _deltanet_v3={engine._deltanet_v3}")
    engine.cleanup()
    return True


def test_cpu_fallback_method_unreachable():
    """Verify _decode_linear_attention (CPU) is never called during decode_step.

    Patches the CPU method with a sentinel that raises AssertionError if called.
    """
    from src.inference.engine import InferenceEngine
    cfg = make_default_config()
    engine = InferenceEngine(cfg, device_id=0, max_seq_len=32)
    assert engine._deltanet_gpu is True

    # Install a sentinel: if CPU path is invoked, it raises AssertionError
    def must_not_be_called(*args, **kwargs):
        raise AssertionError(
            "_decode_linear_attention (CPU fallback) was called! "
            "The GPU path must always be taken for linear attention."
        )

    engine._decode_linear_attention = must_not_be_called

    # Also patch _decode_full_attention to no-op (we only care about linear attention here)
    engine._decode_full_attention = lambda *a, **kw: None

    # Load minimal synthetic weights
    _load_synthetic_weights(engine, cfg)

    # Run a decode step — should not trigger CPU fallback
    hidden = np.zeros(cfg.hidden_size, dtype=np.float16)
    # decode_step calls device.upload for the hidden state (expected H2D)
    # and device.download for the output (expected D2H), but NOT for linear attention
    engine.decode_step(hidden, position=0)
    engine.device.synchronize()

    print("  CPU fallback sentinel not triggered")
    engine.cleanup()
    return True


def test_no_d2h_h2d_for_linear_attention():
    """No D2H (download) or H2D (upload) should occur INSIDE _decode_linear_attention_gpu.

    The GPU-only path must keep all linear attention state/computation on device.
    Only the initial token embedding H2D and final hidden-state D2H are expected
    at the decode_step level, NOT inside the linear attention sub-routine itself.
    """
    from src.inference.engine import InferenceEngine
    cfg = make_default_config()
    engine = InferenceEngine(cfg, device_id=0, max_seq_len=32)
    assert engine._deltanet_gpu is True

    _load_synthetic_weights(engine, cfg)

    # Track transfers inside _decode_linear_attention_gpu
    uploads_in_la = []
    downloads_in_la = []
    currently_in_la = [False]

    orig_la_gpu = engine._decode_linear_attention_gpu

    def tracking_la_gpu(layer_idx, lw, position):
        currently_in_la[0] = True
        try:
            orig_la_gpu(layer_idx, lw, position)
        finally:
            currently_in_la[0] = False

    engine._decode_linear_attention_gpu = tracking_la_gpu

    orig_upload = engine.device.upload
    orig_download = engine.device.download

    def spy_upload(ptr, data):
        if currently_in_la[0]:
            uploads_in_la.append(('H2D', len(data)))
        return orig_upload(ptr, data)

    def spy_download(ptr, size):
        if currently_in_la[0]:
            downloads_in_la.append(('D2H', size))
        return orig_download(ptr, size)

    engine.device.upload = spy_upload
    engine.device.download = spy_download

    # Also patch full attention to no-op
    engine._decode_full_attention = lambda *a, **kw: None

    hidden = np.zeros(cfg.hidden_size, dtype=np.float16)
    engine.decode_step(hidden, position=0)
    engine.device.synchronize()

    n_uploads = len(uploads_in_la)
    n_downloads = len(downloads_in_la)
    print(f"  H2D transfers inside linear attention: {n_uploads} {uploads_in_la}")
    print(f"  D2H transfers inside linear attention: {n_downloads} {downloads_in_la}")

    assert n_uploads == 0, (
        f"Expected 0 H2D transfers inside linear attention GPU path, got {n_uploads}: "
        f"{uploads_in_la}. All data should stay on GPU."
    )
    assert n_downloads == 0, (
        f"Expected 0 D2H transfers inside linear attention GPU path, got {n_downloads}: "
        f"{downloads_in_la}. All data should stay on GPU."
    )
    print("  No D2H/H2D inside linear attention")
    engine.cleanup()
    return True


def test_init_raises_on_missing_kernel():
    """Engine init must raise RuntimeError if deltanet_v3.hip source is missing."""
    import src.inference.engine as engine_module
    from src.inference.engine import InferenceEngine
    from pathlib import Path

    cfg = make_default_config()

    # Point HIP_DIR to a nonexistent path
    original_hip_dir = engine_module.HIP_DIR
    engine_module.HIP_DIR = Path("/nonexistent_path_that_cannot_exist")

    raised = False
    error_msg = ""
    try:
        engine = InferenceEngine(cfg, device_id=0, max_seq_len=32)
        engine.cleanup()
    except RuntimeError as e:
        raised = True
        error_msg = str(e)
    finally:
        engine_module.HIP_DIR = original_hip_dir

    assert raised, (
        "Expected RuntimeError when deltanet_v3.hip source is missing, "
        "but no exception was raised. Silent fallback is NOT acceptable."
    )
    print(f"  Got expected RuntimeError: {error_msg[:80]}...")
    return True


def test_init_raises_on_compile_failure():
    """Engine init must propagate RuntimeError (not silently fallback) on compile failure."""
    import src.inference.engine as engine_module
    from src.inference.engine import InferenceEngine

    cfg = make_default_config()

    original_get_hip = engine_module.KernelCache.get_hip
    call_count = [0]

    def failing_get_hip(self, kernel_name, hip_name, extra_flags=None, hsaco_suffix=""):
        if "deltanet_decode_v3" in kernel_name:
            call_count[0] += 1
            raise RuntimeError("Simulated HIP compilation failure for deltanet_decode_v3")
        return original_get_hip(self, kernel_name, hip_name,
                                 extra_flags=extra_flags, hsaco_suffix=hsaco_suffix)

    engine_module.KernelCache.get_hip = failing_get_hip
    raised = False
    error_msg = ""
    try:
        engine = InferenceEngine(cfg, device_id=0, max_seq_len=32)
        engine.cleanup()
    except RuntimeError as e:
        raised = True
        error_msg = str(e)
    finally:
        engine_module.KernelCache.get_hip = original_get_hip

    assert raised, (
        "Expected RuntimeError to propagate when DeltaNet v3 fails to compile, "
        f"but it was swallowed. call_count={call_count[0]}. "
        "Silent fallback is NOT acceptable."
    )
    assert call_count[0] >= 1, "get_hip mock was never called for deltanet"
    print(f"  Got expected RuntimeError (compile failure): {error_msg[:80]}...")
    return True


def _load_synthetic_weights(engine, cfg):
    """Load minimal synthetic weights for all layers in the engine."""
    from src.inference.engine import LayerWeights

    h = cfg.hidden_size
    la_qkv_dim = engine.la_qkv_dim
    la_dt_dim = engine.la_dt_dim
    la_z_dim = engine.la_z_dim
    la_total = engine.la_total_dim
    num_la_v_heads = engine.local_linear_num_v_heads
    num_la_k_heads = engine.local_linear_num_k_heads
    k_head_dim = cfg.linear_key_head_dim
    v_head_dim = cfg.linear_value_head_dim
    conv_kernel = cfg.linear_conv_kernel_dim
    inter = engine.local_intermediate_size

    # Quantized INT4 weight shapes: B_q4[K//8, N] uint32, scales/zeros[K//group_size, N] fp16
    def make_int4_weights(K, N, group_size=128):
        qweight = np.zeros((K // 8, N), dtype=np.uint32)
        scales = np.ones((K // group_size, N), dtype=np.float16) * 0.01
        zeros = np.zeros((K // group_size, N), dtype=np.float16)
        return qweight, scales, zeros

    while len(engine.layers) < cfg.num_hidden_layers:
        engine.layers.append(LayerWeights())

    for i in range(cfg.num_hidden_layers):
        lw = engine.layers[i]

        def upload_raw(data_bytes, size):
            ptr = engine.device.malloc(size)
            engine.device.upload(ptr, data_bytes)
            return ptr

        def upload_fp16(arr):
            a = arr.astype(np.float16)
            return upload_raw(a.tobytes(), a.nbytes)

        def upload_fp32(arr):
            a = arr.astype(np.float32)
            return upload_raw(a.tobytes(), a.nbytes)

        # RMSNorm weights
        lw.attn_norm = upload_fp16(np.ones(h))
        lw.ffn_norm = upload_fp16(np.ones(h))

        # FFN weights (INT4)
        gw, gs, gz = make_int4_weights(h, inter)
        lw.gate_qweight = upload_raw(gw.tobytes(), gw.nbytes)
        lw.gate_scales = upload_fp16(gs)
        lw.gate_zeros = upload_fp16(gz)
        uw, us, uz = make_int4_weights(h, inter)
        lw.up_qweight = upload_raw(uw.tobytes(), uw.nbytes)
        lw.up_scales = upload_fp16(us)
        lw.up_zeros = upload_fp16(uz)
        dw, ds, dz = make_int4_weights(inter, h)
        lw.down_qweight = upload_raw(dw.tobytes(), dw.nbytes)
        lw.down_scales = upload_fp16(ds)
        lw.down_zeros = upload_fp16(dz)

        lw.layer_type = cfg.layer_types[i]

        if cfg.is_linear_attention(i):
            # Fused in_proj weight: [la_total, h]
            la_in = np.zeros((la_total, h), dtype=np.float16)
            lw.la_in_proj_fused = upload_fp16(la_in)
            off = 0
            lw.la_in_proj_qkv = lw.la_in_proj_fused + off
            off += la_qkv_dim * h * 2
            lw.la_in_proj_a = lw.la_in_proj_fused + off
            off += la_dt_dim * h * 2
            lw.la_in_proj_b = lw.la_in_proj_fused + off
            off += la_dt_dim * h * 2
            lw.la_in_proj_z = lw.la_in_proj_fused + off

            # Out projection: [h, la_z_dim]
            la_out = np.zeros((h, la_z_dim), dtype=np.float16)
            lw.la_out_proj = upload_fp16(la_out)

            # GPU small weights (conv, A_log, dt_bias, norm)
            conv_w = np.ones((la_qkv_dim, conv_kernel), dtype=np.float32) * 0.25
            lw.d_la_conv_weight = upload_fp32(conv_w)

            lw.d_la_A_log = upload_fp32(np.zeros(num_la_v_heads, dtype=np.float32))
            lw.d_la_dt_bias = upload_fp32(np.zeros(num_la_v_heads, dtype=np.float32))
            lw.d_la_norm = upload_fp32(np.ones(v_head_dim, dtype=np.float32))

        elif cfg.is_full_attention(i):
            # Minimal full attention weights
            q_dim = engine.local_num_attention_heads * cfg.head_dim
            kv_dim = engine.local_num_kv_heads * cfg.head_dim
            lw.q_fused_weight = upload_fp16(np.zeros((2 * q_dim, h), dtype=np.float16))
            lw.q_weight = lw.q_fused_weight
            lw.q_gate_weight = lw.q_fused_weight + q_dim * h * 2
            lw.kv_fused_weight = upload_fp16(np.zeros((2 * kv_dim, h), dtype=np.float16))
            lw.k_weight = lw.kv_fused_weight
            lw.v_weight = lw.kv_fused_weight + kv_dim * h * 2
            lw.o_weight = upload_fp16(np.zeros((h, q_dim), dtype=np.float16))
            lw.q_norm = upload_fp16(np.ones(cfg.head_dim))
            lw.k_norm = upload_fp16(np.ones(cfg.head_dim))


if __name__ == "__main__":
    tests = [
        ("DeltaNet GPU flag always True after init", test_deltanet_gpu_flag_true),
        ("CPU fallback method unreachable during decode", test_cpu_fallback_method_unreachable),
        ("No D2H/H2D inside linear attention", test_no_d2h_h2d_for_linear_attention),
        ("Hard error on missing kernel source", test_init_raises_on_missing_kernel),
        ("Hard error on compile failure", test_init_raises_on_compile_failure),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            result = fn()
            passed += 1
            print(f"  -> PASS")
        except AssertionError as e:
            failed += 1
            print(f"  -> FAIL: {e}")
        except Exception as e:
            import traceback
            failed += 1
            print(f"  -> ERROR: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{passed+failed} passed")
    if failed > 0:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
