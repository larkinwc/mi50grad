# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Dev Server (TP=4 Optimization Mission)

**Server:** root@192.168.1.198 (direct SSH, key auth, no jump host)
**CPUs:** 64 cores, 247GB RAM, 1.5TB storage
**GPUs:** 4x MI50/MI60 (gfx906 Vega 20), all PCIe x16 Gen4
- All GPUs are 2 hops apart via PCIe (through CPU/chipset, not direct)
- P2P peer access confirmed for all 12 GPU pairs (hipDeviceCanAccessPeer = 1)
- 16GB VRAM per GPU
- Device IDs: 0x66a0 (all four)

**PCIe Topology:**
- All GPU-to-GPU links: PCIE, weight=40, hops=2
- Link speed: 16.0 GT/s PCIe, width=16
- Theoretical bidirectional bandwidth per link: ~32 GB/s
- Practical P2P bandwidth: ~12-16 GB/s (2-hop overhead)

**Docker:**
- Image: `mi50grad` (based on mixa3607/rocm-gfx906:7.1.0-complete)
- ROCm 7.1.0, hipcc, llvm-mc, ld.lld
- Python 3 with numpy
- Access GPUs via `--device=/dev/kfd --device=/dev/dri --group-add video`

**vLLM Container:**
- Name: `vllm-mobydick` (larkinwc/vllm-gfx906:mobydick-main-rocm-6.3.4)
- Serves Qwen3.5-27B-AWQ on port 8000, TP=4
- Uses 93% VRAM on all 4 GPUs - MUST be stopped before TP benchmarks
- Host networking mode

**Model:** /opt/models/Qwen3.5-27B-GPTQ-Int4
- ~30GB total (mixed FP16 attention + INT4 FFN weights)
- Requires all 4 GPUs for TP=4 (each gets ~7.5GB of weights)

## Performance Baselines (measured 2026-03-14)

- vLLM (TP=4, AWQ, all 4 GPUs): 46.9 tok/s, TTFT 173ms
- mi50grad single-GPU (GPTQ-Int4): 20.3 tok/s (49.3 ms/tok)
- Theoretical TP=4 ceiling: ~81 tok/s (assuming perfect scaling)

## Development Machine
- macOS (local editing only, no GPU)
- Code lives at `/Users/larkinwc/personal/ml/mi50grad`

## Validation LXC (LXC 108)
- Host: Proxmox at `root@wittymantis.netbird.selfhosted`
- LXC IP: `192.168.1.189` (access via SSH ProxyJump)
- SSH command: `ssh -J root@wittymantis.netbird.selfhosted root@192.168.1.189`
- GPU: MI60 32GB (gfx906, same ISA as MI50)
- ROCm: 7.2.0 (native install, no Docker)
- Python: 3.12.3
- PyTorch: 2.7.1+rocm6.2.4
- numpy: 2.3.5
- safetensors, transformers installed
- RAM: 32GB, Disk: 346GB free
- VRAM: 32GB total, ~20GB in use (check before large allocations)
- ROCM_PATH must be set to `/opt/rocm` for builds

## Tool Paths on LXC
- hipcc: `/opt/rocm/bin/hipcc`
- llvm-mc: `/opt/rocm/lib/llvm/bin/llvm-mc` (also via make as `$(ROCM_PATH)/llvm/bin/llvm-mc`)
- ld.lld: `/opt/rocm/lib/llvm/bin/ld.lld`
- rocm-smi: `/opt/rocm/bin/rocm-smi`

## SSH Auth on macOS
- SSH_AUTH_SOCK must be set to `/private/tmp/com.apple.launchd.*/Listeners` for key auth to work through ProxyJump
- If SSH key auth fails, check: `export SSH_AUTH_SOCK=$(ls /private/tmp/com.apple.launchd.*/Listeners 2>/dev/null | head -1)`

## Known Quirks
- ROCm 7.2.0 dropped official MI50/MI60 support but gfx906 still works
- The `which ld.lld` doesn't resolve on LXC but the full path works
- Python import requires `PYTHONPATH=/root/mi50grad` on LXC
- HIP kernels are compiled as shared objects (.so) not HSACO
