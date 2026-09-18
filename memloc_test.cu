// memloc_test.cu  —  猜想③ 验证：内存位置是瓶颈
//
// 对应笔记：docs/cuda-managed-vs-malloc.md §3「猜想 ③」
//
// 同一个 add 内核，同样的数据量，只换内存来源（managed / cudaMalloc）。
// 实测：managed 1.218 ms  vs  cudaMalloc 0.099 ms  =  12.3x
//
// 顺带记录一个证据：在这台机器上想用 cudaMemPrefetchAsync 把 managed 内存
// 推到显存，会直接报 invalid device ordinal —— 说明这条路根本不存在。
// 这不是残留代码，是结论的一部分。
//
// 编译运行： nvcc -O3 memloc_test.cu -o memloc_test && ./memloc_test
// 看明细：   nsys profile --trace=cuda --sample=none -o p ./memloc_test
//            sqlite3 p.sqlite "select (end-start)/1e6, demangledName
//              from CUPTI_ACTIVITY_KIND_KERNEL order by start;"

#include <cstdio>

#define CHECK(msg) do { \
    cudaError_t e_ = cudaGetLastError(); \
    if (e_ != cudaSuccess) { \
        printf("[FAIL] %s -> %s\n", msg, cudaGetErrorString(e_)); \
        return 1; \
    } \
} while (0)

__global__ void add(int n, float* x, float* y)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] += x[i];
}

__global__ void set(int n, float value, float* x)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = value;
}

int main()
{
    const int N = 1000000, G = 3907;
    const size_t bytes = sizeof(float) * N;
    const int REPS = 5;

    // ================= A: cudaMallocManaged =================
    float *mx = nullptr, *my = nullptr;
    cudaMallocManaged(&mx, bytes); CHECK("cudaMallocManaged mx");
    cudaMallocManaged(&my, bytes); CHECK("cudaMallocManaged my");
    set<<<G,256>>>(N, 1.0f, mx);
    set<<<G,256>>>(N, 1.0f, my);
    cudaDeviceSynchronize(); CHECK("A set");
    for (int r = 0; r < REPS; ++r) { add<<<G,256>>>(N, mx, my); cudaDeviceSynchronize(); }
    CHECK("A add");

    // ========== 尝试把 managed 内存预取到显存（预计失败）==========
    // CUDA 13 起 cudaMemPrefetchAsync 的 location 按值传递，不再是裸 device id
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = 0;
    cudaError_t pf = cudaMemPrefetchAsync(mx, bytes, loc, 0, 0);
    printf("[证据] cudaMemPrefetchAsync(mx) -> %s\n", cudaGetErrorString(pf));
    cudaGetLastError();                     // 清掉错误标志，别污染后续调用

    // ================= C: cudaMalloc 显存 =================
    float *dx = nullptr, *dy = nullptr;
    cudaMalloc(&dx, bytes); CHECK("cudaMalloc dx");
    cudaMalloc(&dy, bytes); CHECK("cudaMalloc dy");
    set<<<G,256>>>(N, 1.0f, dx);
    set<<<G,256>>>(N, 1.0f, dy);
    cudaDeviceSynchronize(); CHECK("C set");
    for (int r = 0; r < REPS; ++r) { add<<<G,256>>>(N, dx, dy); cudaDeviceSynchronize(); }
    CHECK("C add");

    printf("done（内核耗时见 nsys，或改用 ladder.cu 看汇总）\n");
    cudaFree(mx); cudaFree(my); cudaFree(dx); cudaFree(dy);
    return 0;
}
