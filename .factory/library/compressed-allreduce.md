# Compressed Allreduce

Design notes for INT8-compressed P2P allreduce on 4x MI50.

## Quantization Scheme

Block-wise INT8 quantization with block_size=32:
- For each block of 32 FP16 values: compute absmax, scale = absmax / 127.0
- Store as: 32 INT8 values (32 bytes) + 1 FP16 scale (2 bytes) = 34 bytes per block
- Uncompressed: 32 * 2 = 64 bytes per block
- Compression ratio: 34/64 = 53.1%

## Memory Layout for Compressed Partials

For 5120 FP16 elements (hidden_size=5120):
- 160 blocks of 32 elements each
- Data section: 5120 INT8 values = 5120 bytes
- Scale section: 160 FP16 scales = 320 bytes
- Total compressed: 5440 bytes (vs 10240 bytes uncompressed)
- Layout in memory: [INT8 data (5120B)] [FP16 scales (320B)]

### Actual Implementation (Fused Kernel)

The fused kernel uses workgroup-based scales rather than block-based:
- Each workgroup (16 columns) produces one FP16 scale
- For n_local = hidden_size/tp_size = 1280 columns: num_wgs = ceil(1280/16) = 80
- Compressed size = n_local + num_wgs*2 = 1280 + 160 = 1440 bytes/GPU
- This is 56% of uncompressed FP16 buffer (2560 bytes)

## Kernel Integration Points

### Standalone allreduce (kernel_p2p_allreduce_compressed.hip)
- Phase 1: Each GPU quantizes its local partial buffer to INT8+scale format
- Phase 2: Each GPU reads 3 peer compressed partials via BAR1 P2P
- Phase 3: Dequantize each peer's data to FP32, sum all 4 partials
- Phase 4: Write FP16 result to hidden buffer

### Fused GEMV+AR+RMSNorm (gemv_int4_p2p_allreduce_rmsnorm_compressed.hip)
- Phase 1-2: GEMV (unchanged from uncompressed version)
- Phase 2b: After GEMV output written to partial_local, quantize it to INT8+scale
- Phase 3: P2P reads of INT8 peer data, dequantize to FP32 for sum-of-squares
- Phase 4: RMSNorm using FP32 sum-of-squares (unchanged logic)

## QuickReduce Reference

AMD's QuickReduce library uses the same inline compression idea on MI300X:
- Block_size=32, supports FP8/Q8/Q6/Q4
- Uses packed math instructions (v_pk_max_f16, v_cvt_pkrtz_f16_f32)
- TwoShot algorithm (reduce + broadcast phases)
- Our case is simpler: star-topology P2P reads, no separate reduce/broadcast

## Expected Performance

- PCIe BAR1 read is the bottleneck (~12 GB/s effective)
- Halving transfer volume: 10KB -> 5.4KB per call
- Theoretical latency reduction: ~79us * (5.4/10.24) = ~42us + quantize/dequantize overhead
- Realistic estimate: ~45-55us per call (quantize/dequantize adds ~5-10us)
- Total savings: 64 calls * (79-50)us = ~1.9ms per token
- Expected throughput: ~57-58 tok/s (from 53.74 baseline)
