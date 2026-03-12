# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

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

## Known Quirks
- ROCm 7.2.0 dropped official MI50/MI60 support but gfx906 still works
- The `which ld.lld` doesn't resolve on LXC but the full path works
- Python import requires `PYTHONPATH=/root/mi50grad` on LXC
- HIP kernels are compiled as shared objects (.so) not HSACO
