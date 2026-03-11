# mi50grad development container
# Based on pre-patched ROCm for gfx906
#
# Build: docker build -t mi50grad .
# Run:   docker run --device=/dev/kfd --device=/dev/dri --group-add video \
#          -v $(pwd):/workspace mi50grad

FROM mixa3607/rocm-gfx906:7.1.0-complete

# Install Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Set up ROCm paths
ENV PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"
ENV HIP_VISIBLE_DEVICES=0,1,2

WORKDIR /workspace

# Default: build everything
CMD ["make", "all"]
