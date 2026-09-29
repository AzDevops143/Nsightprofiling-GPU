# Nsightprofiling-GPU: NVIDIA Nsight Systems Profiling Suite

[![Nsight Systems CI](https://github.com/AzDevops143/Nsightprofiling-GPU/actions/workflows/nsight-profiling.yml/badge.svg)](https://github.com/AzDevops143/Nsightprofiling-GPU/actions)
[![Official NGC Container](https://img.shields.io/badge/NVIDIA%20NGC-nsight--systems--cli-76B900?logo=nvidia)](https://catalog.ngc.nvidia.com/orgs/nvidia/devtools/containers/nsight-systems-cli)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

Production-grade CUDA performance profiling workflow using **NVIDIA Nsight Systems** and **Nsight Compute**. The container uses the official NGC Nsight Systems CLI image and NVIDIA's Nsight Compute CLI package.

This repository profiles and benchmarks a 2D Heat Diffusion 5-point Jacobi Stencil comparing **Global Memory** against **Shared Memory Tiled Stencil with Halo Loading**, instrumented with **NVTX (NVIDIA Tools Extension)** range markers.

---

## What Does NVIDIA Nsight Profiling Output Look Like?

NVIDIA Nsight Systems provides two primary views: **Terminal Summary Tables (CLI)** and **Interactive Timeline Visualization (GUI)**.

### 1. Terminal / CLI Summary (`nsys stats`)

When profiling finishes, Nsight Systems extracts kernel, API, and memory statistics into structured tables:

#### CUDA Kernel Summary (`cuda_gpu_kern_sum`):
```text
 Time (%)  Total Time (ns)  Instances    Avg (ns)       Med (ns)       Min (ns)      Max (ns)     StdDev (ns)              Name
 --------  ---------------  ---------  ------------  --------------  ------------  ------------  ------------  ----------------------------
     54.8        6,984,210      1,000       6,984.2         6,912.0       6,880.0       8,192.0         124.5  heatKernelShared(const float*, float*, int)
     45.2        5,762,340      1,000       5,762.3         5,728.0       5,696.0       7,040.0         112.8  heatKernelGlobal(const float*, float*, int)
```

#### CUDA Runtime API Summary (`cuda_api_sum`):
```text
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)       Min (ns)      Max (ns)     StdDev (ns)              Name
 --------  ---------------  ---------  ------------  --------------  ------------  ------------  ------------  ----------------------------
     51.3       12,410,500      2,000       6,205.2         5,980.0       5,120.0      18,432.0         412.3  cudaMemcpyFromSymbol
     28.7        6,942,100      2,000       3,471.0         3,210.0       2,840.0      14,200.0         285.6  cudaLaunchKernel
     12.1        2,928,400      2,000       1,464.2         1,380.0       1,210.0       9,820.0         194.2  cudaMemcpyToSymbol
      6.4        1,548,200          4     387,050.0       312,000.0     280,000.0     644,200.0     172,310.0  cudaMalloc
```

#### Memory Operations Summary (`cuda_gpu_mem_time_sum`):
```text
 Time (%)  Total Time (ns)  Operations   Avg (ns)       Min (ns)      Max (ns)      Total (MB)   Operation
 --------  ---------------  ----------  ------------  ------------  ------------  ------------  ----------------------
     66.8           48,512           4      12,128.0      11,904.0      12,600.0        1.0486  [CUDA memcpy HtoD]
     33.2           24,088           2      12,044.0      11,840.0      12,248.0        0.5243  [CUDA memcpy DtoH]
```

---

### 2. Graphical UI Timeline (`nsys-ui report.nsys-rep`)

Opening the `.nsys-rep` file in the Nsight Systems desktop GUI displays time-aligned activity tracks:

```text
[ Timeline View: Time Axis ----------------------------------------------------------------------------------------------------> ]

Thread 1 (Main)   | [====================================== Whole Benchmark Run ==============================================]
------------------+---------------------------------------------------------------------------------------------------------------
NVTX Ranges       | [cudaMalloc & HtoD] | [====== Global Memory Solver ======] | [============= Shared Memory Solver ============]
                  |                     |  [Iter Loop: heatKernelGlobal]       |  [Iter Loop: heatKernelShared]
------------------+---------------------+--------------------------------------+--------------------------------------------------
CUDA API Calls    | [cudaMalloc] [Memcpy] [Launch] [CopySym] [Launch] [CopySym] | [Launch] [CopySym] [Launch] [CopySym] [cudaSync]
------------------+---------------------+--------------------------------------+--------------------------------------------------
GPU Context       |                     |                                      |
  Stream 7 (Kern) |                     | |||||||||||||||||||||||||||||||||||| | ||||||||||||||||||||||||||||||||||||||||||||||||
                  |                     | heatKernelGlobal (5.76 us/launch)    | heatKernelShared (6.98 us/launch)
------------------+---------------------+--------------------------------------+--------------------------------------------------
Memory Engine     | [HtoD 512KB]        |                                      |                                   [DtoH 256KB]
```

* **NVTX Spans:** Color-coded logical phases created via `nvtxRangePushEx()`.
* **Kernel Bars:** Exact execution duration and concurrency of CUDA kernel launches.
* **Driver & Runtime APIs:** Reveals synchronization stalls (`cudaDeviceSynchronize`, `cudaMemcpyFromSymbol`).

> [!TIP]
> Read the complete architectural report in [`reports/NSIGHT_PROFILING_REPORT.md`](reports/NSIGHT_PROFILING_REPORT.md).

---

## Repository Structure

```text
Nsightprofiling-GPU/
├── .github/
│   └── workflows/
│       └── nsight-profiling.yml   # CI workflow: multi-stage build, test & artifact upload
├── Dockerfile                     # Multi-stage Dockerfile based on official NGC container
├── docker-compose.yml             # Docker compose with GPU reservations & privileges
├── Makefile                       # Convenience targets: build, profile, stats, clean
├── entrypoint.sh                  # Container entrypoint for automated trace capture
├── heat_diffusion.cu              # CUDA C++ source instrumented with NVTX markers
├── reports/                       # Pre-packaged Nsight profiling reports & traces
│   ├── heat_diffusion_profile.nsys-rep # Native Nsight Systems report
│   └── heat_diffusion_profile.ncu-rep  # Native Nsight Compute report
├── .gitignore
└── README.md
```

---

## Using the Official NGC Container

This repository uses a multi-stage Docker build with the official NGC image:
- **Builder Stage:** `nvidia/cuda:12.8.0-devel-ubuntu22.04` (compiles with `-O3 -lineinfo -lnvToolsExt -cudart static`).
- **Runner Stage:** `nvcr.io/nvidia/devtools/nsight-systems-cli:2025.6.1-ubuntu22.04` with Nsight Compute CLI 2026.3.0 installed from NVIDIA's Ubuntu repository.

### 1. Build the Docker Image
```bash
docker build -t ghcr.io/azdevops143/nsightprofiling-gpu:latest .
```
Or via Makefile:
```bash
make build
```

### 2. Run Profiling on GPU
Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed on your host:
```bash
docker run --gpus all --privileged --ipc=host --rm \
  -v $(pwd)/reports:/reports \
  ghcr.io/azdevops143/nsightprofiling-gpu:latest 256 1e-4 1000
```
Or with Docker Compose:
```bash
docker compose up
```

### 3. Print Profiling Summary from Host
```bash
make stats
```

---

## How to Inspect the Generated Trace in Nsight GUI

1. Install the desktop **NVIDIA Nsight Systems GUI** from [NVIDIA Developer](https://developer.nvidia.com/nsight-systems).
2. Open the GUI and load the trace file:
   ```bash
   nsys-ui reports/heat_diffusion_profile.nsys-rep
   ```
3. Expand **CUDA HW** and **NVTX** to inspect the kernel timings and thread execution.

---

## Pushing to GitHub

To push to your new repository:
```bash
git remote set-url origin https://github.com/AzDevops143/Nsightprofiling-GPU.git
git add .
git commit -m "feat: setup clean NVIDIA Nsight profiling suite with official NGC container"
git push -u origin main
```
