#!/usr/bin/env bash
set -e

GRID_N="${1:-256}"
EPS="${2:-1e-4}"
MAX_ITER="${3:-1000}"
OUT_DIR="/reports"
REPORT_BASE="${OUT_DIR}/heat_diffusion_profile"

mkdir -p "${OUT_DIR}"

echo "=================================================================="
echo "NVIDIA Nsight Systems Profiling Suite (Official NGC Container)"
echo "Target: 2D Heat Diffusion Jacobi Solver (Global vs Shared Memory)"
echo "Parameters: N=${GRID_N}, tol=${EPS}, max_iterations=${MAX_ITER}"
echo "=================================================================="

# Check if NVIDIA GPU is accessible
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA GPU detected. Starting live Nsight Systems profiling..."
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    echo ""

    echo "-> 1. Capturing trace (CUDA runtime, NVTX annotations, OS runtime)..."
    nsys profile \
        -t cuda,nvtx,osrt \
        --stats=true \
        --force-overwrite=true \
        -o "${REPORT_BASE}" \
        /workspace/heat_diffusion "${GRID_N}" "${EPS}" "${MAX_ITER}"

    echo ""
    echo "-> 2. Exporting CUDA Kernel Execution Summary..."
    nsys stats --report cuda_gpu_kern_sum --format table "${REPORT_BASE}.nsys-rep" > "${OUT_DIR}/cuda_gpu_kern_sum.txt"
    cat "${OUT_DIR}/cuda_gpu_kern_sum.txt"

    echo ""
    echo "-> 3. Exporting CUDA API Calls Summary..."
    nsys stats --report cuda_api_sum --format table "${REPORT_BASE}.nsys-rep" > "${OUT_DIR}/cuda_api_sum.txt"
    cat "${OUT_DIR}/cuda_api_sum.txt"

    echo ""
    echo "-> 4. Exporting GPU Memory Operations Summary..."
    nsys stats --report cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format table "${REPORT_BASE}.nsys-rep" > "${OUT_DIR}/cuda_mem_sum.txt"
    cat "${OUT_DIR}/cuda_mem_sum.txt"

    echo ""
    echo "-> 5. Exporting SQLite Database for SQL querying..."
    nsys export --type sqlite -o "${REPORT_BASE}.sqlite" "${REPORT_BASE}.nsys-rep"

    echo ""
    echo "[SUCCESS] Profiling completed! All artifacts generated in ${OUT_DIR}:"
    ls -lh "${OUT_DIR}"
else
    echo "[NOTICE] No physical NVIDIA GPU device detected in current container environment."
    echo "         (If running in Docker, remember to pass '--gpus all --privileged')."
    echo ""
    echo "-> Verifying binary compilation and Nsight Systems CLI tools..."
    nsys --version
    echo ""
    echo "[INFO] To run live GPU profiling on any host with an NVIDIA GPU:"
    echo "  docker run --gpus all --privileged --rm -v \$(pwd)/reports:/reports \\"
    echo "    ghcr.io/azdevops143/nsightprofiling-gpu:latest ${GRID_N} ${EPS} ${MAX_ITER}"
fi
