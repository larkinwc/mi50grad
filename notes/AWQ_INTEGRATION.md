# AWQ Model Integration for Qwen3.5-27B on MI50

## Overview

This document describes the integration of AWQ (Activation-aware Weight Quantization) models into the mi50grad inference engine for Qwen3.5-27B on AMD MI50 (gfx906) GPUs.

## AWQ vs GPTQ

**Key Differences:**
- **GPTQ**: `w = (q - zero) * scale` (requires zero-point tensor)
- **AWQ**: `w = q * scale` (zero-point = 0, no subtraction needed)

**Benefits of AWQ:**
- Eliminates 8 `v_sub_f32` instructions per uint32 word in GEMV kernels
- No zeros tensor memory traffic (saves memory bandwidth)
- Expected speedup: ~5-10% on memory-bound GEMV operations

## Model Download

### Download Script

```bash
# Download AWQ-quantized Qwen3.5-27B from HuggingFace
./scripts/download_awq_model.sh
```

This downloads `QuantTrio/Qwen3.5-27B-AWQ` to `/opt/models/Qwen3.5-27B-AWQ`.

**Model Details:**
- Repository: https://huggingface.co/QuantTrio/Qwen3.5-27B-AWQ
- Size: ~21GB (8 safetensors files)
- Format: AWQ INT4 (no qzeros tensors)
- Architecture: 64 layers, hidden_size=5120, intermediate_size=17408

### Manual Download

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='QuantTrio/Qwen3.5-27B-AWQ',
    local_dir='/opt/models/Qwen3.5-27B-AWQ',
    local_dir_use_symlinks=False,
    ignore_patterns=['*.git*', '*.md', '*.jpg', '*.png']
)
```

## Format Detection

The AWQ loader automatically detects the quantization format:

```python
from src.model.awq_loader import detect_awq_format

format = detect_awq_format('/opt/models/Qwen3.5-27B-AWQ')
# Returns: 'awq', 'gptq', or 'fp16'

# Detection logic:
# - AWQ: has qweight + scales, NO qzeros
# - GPTQ: has qweight + scales + qzeros
# - FP16: no qweight (full precision)
```

## Weight Loading

### Automatic Detection and Loading

```python
from src.model.awq_loader import load_awq_or_gptq
from src.model.qwen import load_config_from_json

config = load_config_from_json(model_dir)
loader, detected_format = load_awq_or_gptq(model_dir, config)

# loader is either AWQWeightLoader or QwenWeightLoader
# detected_format is 'awq', 'gptq', or 'fp16'
```

### Manual AWQ Loading

```python
from src.model.awq_loader import AWQWeightLoader

config = load_config_from_json(model_dir)
loader = AWQWeightLoader(model_dir, config)

# Load layer weights
for layer_idx in range(config.num_hidden_layers):
    weights = loader.load_layer(layer_idx)
    engine.load_layer_weights(layer_idx, weights)

engine.load_final_norm(loader.load_final_norm())
engine.load_lm_head(loader.load_lm_head())
```

## AWQ Kernel Integration

### Kernel Availability

The AWQ GEMV kernel (`gemv_int4_v5_awq.hip`) is automatically compiled if the source file exists in `src/kernels/`. It provides three variants:
- `gemv_int4_v5_awq_t16` (16 threads/column, default for most cases)
- `gemv_int4_v5_awq_t8` (8 threads/column)
- `gemv_int4_v5_awq_t4` (4 threads/column)

### Enabling AWQ Mode

```python
from src.inference.tp_engine import TPInferenceEngine

# Create engine and load weights
engine = TPInferenceEngine(config, device_ids=[0,1,2,3], max_seq_len=512)

# ... load weights ...

# Enable AWQ mode BEFORE building dispatch cache
engine.set_awq_mode(True)

# Now build dispatch cache (will use AWQ kernels)
engine.build_dispatch_cache()

# Enable other optimizations
engine.set_direct_kv_write(True)
engine.set_kernel_p2p_allreduce(True)
engine.set_c_dispatch(True)
```

**IMPORTANT**: Call `set_awq_mode()` BEFORE `build_dispatch_cache()` to ensure the cached launch specs use AWQ kernels. If called after, the cache will be automatically rebuilt.

### Kernel Selection Logic

The engine selects kernels based on mode and availability:

```python
# For FFN down projection:
if self._awq_mode and self._gemv_int4_v5_awq:
    kernel = gemv_int4_v5_awq_t16  # AWQ kernel (no zero-point)
elif self._gemv_int4_v5:
    kernel = gemv_int4_v5_t16      # Standard v5 kernel
else:
    kernel = gemv_int4_v3_t16      # Fallback v3 kernel
```

### C Dispatch Integration

AWQ kernels are fully integrated into the C dispatch path:

1. When `set_awq_mode(True)` is called, the engine sets `_awq_mode = True`
2. When `build_dispatch_cache()` is called, it checks `_awq_mode` and selects AWQ kernels
3. The cached launch specs (LaunchSpec objects) contain the AWQ kernel function pointers
4. C dispatch uses these cached specs directly, so AWQ kernels are used without additional overhead

## Testing

### Test Files

1. **`tests/test_awq_loader.py`**: Unit tests for AWQ weight loader
   - Shape/dtype validation
   - No zero-point tensor verification
   - Dequantization correctness
   - AWQ/GPTQ format compatibility

2. **`tests/test_awq_model_load.py`**: Integration tests for AWQ model
   - Format detection
   - Weight loader functionality
   - Single-GPU engine loading
   - TP=4 engine loading
   - C dispatch plan verification

3. **`tests/test_awq_e2e.py`**: End-to-end tests
   - AWQ kernel availability
   - TP=4 decode coherence
   - AWQ vs GPTQ throughput comparison

### Running Tests

```bash
# Test AWQ loader (no GPU needed)
python3 tests/test_awq_loader.py

# Test AWQ model load and integration (requires GPU + model)
python3 tests/test_awq_model_load.py

# End-to-end test with coherence check
python3 tests/test_awq_e2e.py
```

## Performance Expectations

### Theoretical Speedup

- **GEMV down_proj**: ~8% speedup (eliminates 8 v_sub_f32 per uint32)
- **Overall decode**: ~3-5% speedup (down_proj is ~30-40% of layer time)

### Memory Savings

- **Zeros tensor eliminated**: 136 * 5120 * 2 bytes = ~1.4MB per down_proj layer
- **Total for 64 layers**: ~90MB saved in weight memory

### Benchmark Comparison

| Mode | Throughput | Speedup |
|------|------------|---------|
| GPTQ baseline | 38.3 tok/s | 1.00x |
| AWQ kernel mode | ~40-42 tok/s | ~1.05-1.10x |

*Note: Actual speedup depends on memory bandwidth utilization and kernel occupancy.*

## Known Limitations

1. **AWQ kernel only for down_proj**: The `gemv_int4_dual_fused` kernel (used for gate+up projections) still uses zero-point subtraction. An AWQ variant would provide additional speedup.

2. **No AWQ-specific calibration**: The current integration uses the same kernel parameters as GPTQ. AWQ-specific tuning could improve performance.

3. **Limited to INT4**: AWQ support is only for INT4 weights. Other quantization formats (INT8, FP8) are not affected.

## Troubleshooting

### AWQ Kernel Not Compiled

**Symptom**: `WARNING: AWQ GEMV kernel not available`

**Solution**:
```bash
# Check if source exists
ls src/kernels/gemv_int4_v5_awq.hip

# Recompile kernels
cd /opt/mi50grad
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $(pwd):/workspace mi50grad make kernels
```

### Model Not Found

**Symptom**: `AWQ model not found at /opt/models/Qwen3.5-27B-AWQ`

**Solution**: Run the download script:
```bash
./scripts/download_awq_model.sh
```

### Format Detection Returns 'gptq' for AWQ Model

**Possible causes**:
1. Model has unexpected qzeros tensors (not a true AWQ model)
2. Corrupted model files

**Debug**:
```python
from src.model.awq_loader import detect_awq_format
import json

# Check format
format = detect_awq_format(model_dir)
print(f"Detected: {format}")

# Inspect weight map
with open(f"{model_dir}/model.safetensors.index.json") as f:
    index = json.load(f)
    
# Look for qzeros keys
qzeros_keys = [k for k in index['weight_map'].keys() if 'qzeros' in k]
print(f"qzeros keys: {qzeros_keys}")
# Should be empty for AWQ models
```

## Future Improvements

1. **AWQ gate+up kernel**: Create `gemv_int4_dual_awq.hip` to eliminate zero-point from gate+up projections

2. **AWQ-specific tuning**: Optimize kernel parameters for AWQ weight distributions

3. **FP8 AWQ**: Support AWQ quantization with FP8 activations for additional speedup

4. **Automatic format detection in engine**: Detect format during weight loading and automatically enable AWQ mode

## References

- AWQ Paper: https://arxiv.org/abs/2306.00978
- AutoAWQ: https://github.com/casper-hansen/AutoAWQ
- HuggingFace AWQ Model: https://huggingface.co/QuantTrio/Qwen3.5-27B-AWQ
- LLVM gfx906 ISA: https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX906.html
