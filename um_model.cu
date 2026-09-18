// um_model.cu  —  判别 managed 内存是「启动时整体搬运」还是「按需远程取」
//
// 对应笔记：docs/cuda-managed-vs-malloc.md §4.2
//
// 两种互斥的机制假说：
//   甲「启动时整体搬运」→ 内核只碰 4MB，也得为整个缓冲区买单
//                        → 耗时随【分配大小】增长
//   乙「按需远程取」    → 只碰 4MB 就只付 4MB
//                        → 耗时只随【实际访问量】变化
//
// 实验设计的关键：固定访问量，只改分配大小。这样才隔离出想测的那一个变量。
// （对比"碰 4MB vs 碰 256MB"是两个变量一起动，说不清是哪个造成的差异。）
//
// 本机实测（每次只碰前 4 MB）：
//     managed 缓冲区  8 MB → 1.221 ms
//                    64 MB → 1.221 ms
//                   240 MB → 1.237 ms
//   分配大小涨 30 倍，耗时纹丝不动 → 是【乙】。
//   若按甲，240MB 那档该要约 76 ms，实测 1.24 ms，差 61 倍。
//
// 编译运行： nvcc -O3 um_model.cu -o um_model && ./um_model
// 注意 240 MB 一档需要约 480 MB 显存，小显存卡请调小 allocs[]

#include <cuda_runtime.h>
#include <cstdio>

__global__ void addk(int n, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = gridDim.x * blockDim.x;
    for (; i < n; i += s) y[i] += x[i];
}
__global__ void setk(int n, float v, float* x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = gridDim.x * blockDim.x;
    for (; i < n; i += s) x[i] = v;
}

static double bench(int n, float* x, float* y, int reps)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int r = 0; r < reps; ++r) addk<<<1024, 256>>>(n, x, y);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / reps;
}

int main()
{
    const size_t MB = 1024ull * 1024;
    const int NTOUCH = 1000000;                        // 固定只碰 4 MB
    const size_t touchBytes = (size_t)NTOUCH * 4;

    size_t allocs[] = { 8 * MB, 64 * MB, 240 * MB };   // 分配大小从小到大

    printf("固定访问量 = 4 MB，只改 managed 缓冲区的【分配大小】\n");
    printf("若启动时整体搬运，耗时该随分配大小线性增长\n\n");
    printf("%-16s %-12s %-12s %s\n", "分配大小/数组", "内核耗时", "折合带宽", "判读");
    printf("------------------------------------------------------------------\n");

    for (size_t bytes : allocs)
    {
        int nAlloc = (int)(bytes / 4);

        float *x = nullptr, *y = nullptr;
        if (cudaMallocManaged(&x, bytes) != cudaSuccess ||
            cudaMallocManaged(&y, bytes) != cudaSuccess) {
            printf("%-16zu 分配失败（显存不够）\n", bytes / MB);
            cudaFree(x); cudaFree(y);
            continue;
        }

        // 全量初始化，把整个缓冲区的页面都碰一遍（排除首次缺页的干扰）
        setk<<<1024,256>>>(nAlloc, 1.0f, x);
        setk<<<1024,256>>>(nAlloc, 1.0f, y);
        cudaDeviceSynchronize();

        double ms = bench(NTOUCH, x, y, 20);
        double gb = 3.0 * touchBytes / 1e9;

        printf("%-16zu %8.3f ms   %6.2f GB/s   %s\n",
               bytes / MB, ms, gb / (ms / 1e3),
               (ms > 55000) ? "→ 整体搬运" : "→ 与分配大小无关");

        cudaFree(x); cudaFree(y);
    }

    printf("\n参照：若真搬运整个 240MB x 2 的缓冲区，按实测 6.6 GB/s 需要约 76 ms\n");
    return 0;
}
