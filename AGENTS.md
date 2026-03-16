
This project is dedicated to optimizing token throughput for qwen3.5 27B and similar models running of gfx906 architecture.

We have a development ML server (root@192.168.1.198, accessible via ssh key (default ssh auth)) that has 4x mi50s (gfx906, 32GB HBM2 each). These are the focus, and should be used to validate all of our work here.

## References (Primary Sources)
AMD ROCm GPU architecture specs (Instinct table):
https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
AMD ROCm HIP hardware implementation:
https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
LLVM AMDGPU usage/reference (target features, restrictions, target IDs):
https://llvm.org/docs/AMDGPUUsage.html
LLVM gfx906 instruction syntax (per-target assembler reference):
https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX906.html
LLVM gfx908 instruction syntax (contrast target showing MFMA forms):
https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX908.html
AMD IR launch release (Nov 6, 2018), MI60/MI50: https://ir.amd.com/news-events/press-releases/detail/859/amd-unveils-worlds-first-7nm-datacenter-gpus----powering-the-next-era-of-artificial-intelligence-cloud-computing-and-high-performance-computing-hpc
AMD Vega 7nm Shader ISA PDF: https://gpuopen.com/wp-content/uploads/2019/11/Vega_7nm_Shader_ISA_26November2019.pdf


## Secondary source
@reference/wiki-gfx906