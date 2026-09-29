# ==============================================================================
# Multi-Stage Build for NVIDIA Nsight Systems Profiling
# Builder: Official CUDA 12.8 Toolkit (multi-architecture compilation + NVTX)
# Runner : Official NVIDIA NGC Nsight Systems CLI Container
# Reference: https://catalog.ngc.nvidia.com/orgs/nvidia/devtools/containers/nsight-systems-cli
# ==============================================================================

ARG NSYS_TAG=2025.6.1-ubuntu22.04

# Stage 1: Build the CUDA application
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY heat_diffusion.cu /build/

# Compile for modern NVIDIA GPU architectures: Turing, Ampere, Ada Lovelace, Hopper, Blackwell
# Statically link cudart and link NVTX for profiling instrumentation
RUN nvcc -O3 -lineinfo -std=c++17 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_100,code=sm_100 \
    -gencode arch=compute_120,code=sm_120 \
    -gencode arch=compute_100,code=compute_100 \
    -cudart static -lnvToolsExt \
    heat_diffusion.cu -o heat_diffusion

# Stage 2: Official NVIDIA NGC Nsight Systems CLI Container
FROM nvcr.io/nvidia/devtools/nsight-systems-cli:${NSYS_TAG}

LABEL maintainer="AzDevops143"
LABEL description="Official NVIDIA Nsight Systems CLI Container for CUDA Heat Diffusion Profiling"
LABEL org.opencontainers.image.source="https://github.com/AzDevops143/Nsightprofiling-GPU"

WORKDIR /workspace

# Copy compiled binary, source code, and entrypoint
COPY --from=builder /build/heat_diffusion /workspace/heat_diffusion
COPY heat_diffusion.cu /workspace/heat_diffusion.cu
COPY entrypoint.sh /workspace/entrypoint.sh

RUN chmod +x /workspace/heat_diffusion /workspace/entrypoint.sh && \
    mkdir -p /reports

VOLUME ["/reports"]
WORKDIR /workspace

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["256", "1e-4", "1000"]
