# NVIDIA Nsight Systems Profiling Artifacts

This directory contains the complete performance profiling reports, traces, and tabular metrics generated using the **official NVIDIA NGC Nsight Systems CLI container** (`nvcr.io/nvidia/devtools/nsight-systems-cli`).

## Files in this Directory

| File | Type | Description |
| :--- | :--- | :--- |
| [`NSIGHT_PROFILING_REPORT.md`](NSIGHT_PROFILING_REPORT.md) | Markdown | **Main Report**: Detailed architectural analysis, ASCII timeline, tables, and insights. |
| [`profile_summary.txt`](profile_summary.txt) | Text | Clean terminal-formatted summary tables (`nsys stats`) for quick CLI viewing. |
| [`cuda_gpu_kern_sum.txt`](cuda_gpu_kern_sum.txt) | Text | Kernel execution breakdown (Global vs Shared memory Jacobi stencil). |
| [`cuda_api_sum.txt`](cuda_api_sum.txt) | Text | CUDA Runtime API call frequencies and latency distribution. |
| [`cuda_mem_sum.txt`](cuda_mem_sum.txt) | Text | Host-to-Device and Device-to-Host transfer volumes and throughput. |
| `heat_diffusion_profile.nsys-rep` | Binary Trace | Native Nsight Systems trace file. Open directly in **Nsight Systems GUI (`nsys-ui`)**. |
| `heat_diffusion_profile.sqlite` | SQLite DB | Relational database containing CUPTI kernel, API, and memory events for SQL queries. |

## Quick Analysis Commands

### 1. View text summary:
```bash
cat reports/profile_summary.txt
```

### 2. Inspect with SQLite:
```bash
sqlite3 reports/heat_diffusion_profile.sqlite "SELECT demangledName, COUNT(*), AVG(end-start)/1000.0 FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY demangledName;"
```

### 3. Open in NVIDIA Nsight Systems GUI:
```bash
nsys-ui reports/heat_diffusion_profile.nsys-rep
```
