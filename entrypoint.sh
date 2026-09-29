#!/usr/bin/env bash
set -e

# Support direct passthrough commands like "bash", "sh", "nsys", "ncu", or "--version"
if [ "$1" = "nsys" ] || [ "$1" = "ncu" ] || [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    exec "$@"
fi

if [ "$1" = "--version" ] || [ "$1" = "-v" ] || [ "$1" = "version" ]; then
    echo "=================================================================="
    echo "NVIDIA Nsight Systems CLI Profiler Container"
    echo "=================================================================="
    nsys --version
    exit 0
fi

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
        --force-overwrite=true \
        -o "${REPORT_BASE}" \
        /workspace/heat_diffusion "${GRID_N}" "${EPS}" "${MAX_ITER}"

    echo ""
    echo "-> 2. Capturing Nsight Compute kernel report..."
    ncu --set basic \
        --force-overwrite \
        -o "${REPORT_BASE}" \
        /workspace/heat_diffusion "${GRID_N}" "${EPS}" 4

    echo ""
    echo "[SUCCESS] Native Nsight reports generated in ${OUT_DIR}:"
    find "${OUT_DIR}" -maxdepth 1 -type f \( -name '*.nsys-rep' -o -name '*.ncu-rep' \) -printf '%f\n'
else
    echo "[NOTICE] No physical NVIDIA GPU device detected in current container environment."
    echo "         (If running in Docker, pass '--gpus all --privileged')."
    echo ""
    echo "-> Verifying binary compilation and Nsight Systems CLI tools..."
    nsys --version
    echo ""
    echo "[INFO] To run live GPU profiling on any host with an NVIDIA GPU:"
    echo "  docker run --gpus all --privileged --rm -v \$(pwd)/reports:/reports \\"
    echo "    ghcr.io/azdevops143/nsightprofiling-gpu:latest ${GRID_N} ${EPS} ${MAX_ITER}"
fi
