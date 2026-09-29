#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define USE_NVTX 1
#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>

inline void nvtx_push(const char* name, uint32_t color = 0xFF2ECC71) {
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    nvtxRangePushEx(&eventAttrib);
}

inline void nvtx_pop() {
    nvtxRangePop();
}
#else
inline void nvtx_push(const char*, uint32_t = 0) {}
inline void nvtx_pop() {}
#endif

#define BLOCK 16
#define GLOBAL_MODE 0
#define SHARED_MODE 1

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

__device__ float g_maxdiff;

// ----------------------------------------------------------------------------
// Global-Memory 5-Point Jacobi Stencil Kernel
// ----------------------------------------------------------------------------
__global__ void heatKernelGlobal(const float* __restrict__ T, float* __restrict__ Tnew, int N)
{
    extern __shared__ float sdata[];
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float diff = 0.0f;
    if (row <= N - 2 && col <= N - 2) {
        int idx = row * N + col;
        float up    = T[idx - N];
        float down  = T[idx + N];
        float left  = T[idx - 1];
        float right = T[idx + 1];
        float newval = 0.25f * (up + down + left + right);
        diff = fabsf(newval - T[idx]);
        Tnew[idx] = newval;
    }

    sdata[tid] = diff;
    __syncthreads();

    // In-block tree reduction for maximum difference
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) atomicMax((int*)&g_maxdiff, __float_as_int(sdata[0]));
}

// ----------------------------------------------------------------------------
// Shared-Memory Tiled 5-Point Jacobi Stencil Kernel with Halo Boundaries
// ----------------------------------------------------------------------------
__global__ void heatKernelShared(const float* __restrict__ T, float* __restrict__ Tnew, int N)
{
    extern __shared__ float smemAll[];
    int tileDim = blockDim.x + 2;
    float* tile = smemAll;
    float* sdata = smemAll + tileDim * tileDim;

    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    int colc = min(col, N - 1);
    int rowc = min(row, N - 1);

    // Load center cell
    tile[ly * tileDim + lx] = T[rowc * N + colc];

    // Load halo cells (left, right, top, bottom)
    if (threadIdx.x == 0) {
        int leftCol = max(col - 1, 0);
        tile[ly * tileDim + 0] = T[rowc * N + leftCol];
    }
    if (threadIdx.x == blockDim.x - 1) {
        int rightCol = min(col + 1, N - 1);
        tile[ly * tileDim + (lx + 1)] = T[rowc * N + rightCol];
    }
    if (threadIdx.y == 0) {
        int upRow = max(row - 1, 0);
        tile[0 * tileDim + lx] = T[upRow * N + colc];
    }
    if (threadIdx.y == blockDim.y - 1) {
        int downRow = min(row + 1, N - 1);
        tile[(ly + 1) * tileDim + lx] = T[downRow * N + colc];
    }

    __syncthreads();

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float diff = 0.0f;
    bool valid = (row <= N - 2 && col <= N - 2);

    if (valid) {
        float up    = tile[(ly - 1) * tileDim + lx];
        float down  = tile[(ly + 1) * tileDim + lx];
        float left  = tile[ly * tileDim + (lx - 1)];
        float right = tile[ly * tileDim + (lx + 1)];
        float newval = 0.25f * (up + down + left + right);
        int idx = row * N + col;
        diff = fabsf(newval - T[idx]);
        Tnew[idx] = newval;
    }

    sdata[tid] = diff;
    __syncthreads();

    // In-block tree reduction
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) atomicMax((int*)&g_maxdiff, __float_as_int(sdata[0]));
}

struct SimResult {
    long long iterations;
    float time_ms;
    bool converged;
    float final_max_diff;
    std::vector<float> grid;
};

void initGrid(std::vector<float>& g, int N, float topT, float bottomT, float leftT, float rightT, float initTemp)
{
    g.assign((size_t)N * N, initTemp);
    for (int j = 0; j < N; j++) {
        g[0 * N + j] = topT;
        g[(N - 1) * N + j] = bottomT;
    }
    for (int i = 0; i < N; i++) {
        g[i * N + 0] = leftT;
        g[i * N + (N - 1)] = rightT;
    }
}

SimResult runSimulation(int N, float tol, long long maxIter, int mode,
                         float topT, float bottomT, float leftT, float rightT, float initTemp)
{
    const char* modeName = (mode == GLOBAL_MODE) ? "Global Memory Solver" : "Shared Memory Tiled Solver";
    uint32_t modeColor = (mode == GLOBAL_MODE) ? 0xFF3498DB : 0xFFE67E22; // Blue vs Orange

    nvtx_push(modeName, modeColor);

    SimResult result;
    std::vector<float> h_init;
    initGrid(h_init, N, topT, bottomT, leftT, rightT, initTemp);

    float *d_A, *d_B;
    size_t bytes = (size_t)N * N * sizeof(float);
    
    nvtx_push("cudaMalloc & cudaMemcpy HtoD", 0xFF9B59B6);
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_init.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_init.data(), bytes, cudaMemcpyHostToDevice));
    nvtx_pop();

    dim3 block(BLOCK, BLOCK);
    int interior = N - 2;
    dim3 grid((interior + BLOCK - 1) / BLOCK, (interior + BLOCK - 1) / BLOCK);

    size_t sharedBytesGlobal = block.x * block.y * sizeof(float);
    size_t tileDim = block.x + 2;
    size_t sharedBytesShared = tileDim * tileDim * sizeof(float) + block.x * block.y * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float* dA = d_A;
    float* dB = d_B;
    long long iter = 0;
    float diffVal = 1e30f;
    float zero = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));

    nvtx_push("Jacobi Iteration Loop", 0xFFE74C3C);
    while (diffVal > tol && iter < maxIter) {
        CUDA_CHECK(cudaMemcpyToSymbol(g_maxdiff, &zero, sizeof(float)));

        if (mode == GLOBAL_MODE) {
            heatKernelGlobal<<<grid, block, sharedBytesGlobal>>>(dA, dB, N);
        } else {
            heatKernelShared<<<grid, block, sharedBytesShared>>>(dA, dB, N);
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyFromSymbol(&diffVal, g_maxdiff, sizeof(float)));

        float* tmp = dA;
        dA = dB;
        dB = tmp;

        iter++;
    }
    nvtx_pop(); // End Jacobi Iteration Loop

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    result.iterations = iter;
    result.time_ms = ms;
    result.converged = (diffVal <= tol);
    result.final_max_diff = diffVal;

    result.grid.assign((size_t)N * N, 0.0f);
    nvtx_push("cudaMemcpy DtoH & Cleanup", 0xFF1ABC9C);
    CUDA_CHECK(cudaMemcpy(result.grid.data(), dA, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    nvtx_pop();

    nvtx_pop(); // End Solver mode range
    return result;
}

int main(int argc, char** argv)
{
    printf("==================================================================\n");
    printf("NVIDIA Nsight Systems Profiling Target: 2D Heat Diffusion (CUDA)\n");
    printf("==================================================================\n");

    int N = 256;
    float eps = 1e-4f;
    long long maxIter = 1000; // Default to 1000 for focused profiling traces
    float topT = 100.0f;
    float bottomT = 0.0f;
    float leftT = 75.0f;
    float rightT = 50.0f;
    float initTemp = 0.0f;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) eps = (float)atof(argv[2]);
    if (argc > 3) maxIter = atoll(argv[3]);
    if (argc > 4) topT = (float)atof(argv[4]);
    if (argc > 5) bottomT = (float)atof(argv[5]);
    if (argc > 6) leftT = (float)atof(argv[6]);
    if (argc > 7) rightT = (float)atof(argv[7]);
    if (argc > 8) initTemp = (float)atof(argv[8]);

    printf("Configuration:\n");
    printf("  - Grid Dimensions      : %d x %d (%zu elements)\n", N, N, (size_t)N * N);
    printf("  - Convergence Tolerance: %e\n", eps);
    printf("  - Max Iterations       : %lld\n", maxIter);
    printf("  - Block Size           : %dx%d (256 threads/block)\n", BLOCK, BLOCK);
    printf("  - NVTX Markers         : ENABLED (Colored timeline ranges)\n\n");

    nvtx_push("Whole Benchmark Run", 0xFF2C3E50);

    printf("-> Running 1/2: Global Memory Kernel...\n");
    SimResult rG = runSimulation(N, eps, maxIter, GLOBAL_MODE, topT, bottomT, leftT, rightT, initTemp);
    printf("   Global Memory: %lld iters, %.3f ms (Converged: %s, Max Diff: %e)\n\n",
           rG.iterations, rG.time_ms, rG.converged ? "YES" : "NO", rG.final_max_diff);

    printf("-> Running 2/2: Shared Memory Tiled Kernel...\n");
    SimResult rS = runSimulation(N, eps, maxIter, SHARED_MODE, topT, bottomT, leftT, rightT, initTemp);
    printf("   Shared Memory: %lld iters, %.3f ms (Converged: %s, Max Diff: %e)\n\n",
           rS.iterations, rS.time_ms, rS.converged ? "YES" : "NO", rS.final_max_diff);

    nvtx_pop(); // End Whole Benchmark Run

    float maxDiff = 0.0f;
    for (size_t i = 0; i < rG.grid.size(); i++) {
        float d = fabsf(rG.grid[i] - rS.grid[i]);
        if (d > maxDiff) maxDiff = d;
    }

    printf("==================================================================\n");
    printf("Correctness Check: Max absolute diff between Global & Shared = %e\n", maxDiff);
    printf("Speedup Ratio (Global / Shared): %.4fx\n", rG.time_ms / rS.time_ms);
    printf("==================================================================\n");

    return 0;
}
