// readcache.cu  —  两个问题：
//   1. GPU 从主机内存读来的数据，会不会被 L2 缓存？还是每次读都真的跨 PCIe？
//   2. managed 内存和「锁页映射内存（zero-copy）」是不是同一套机制？
//
// 对应笔记：docs/cuda-managed-vs-malloc.md §4.1
//
// 为什么要问这两个：
//   §4.1 观察到 managed 带宽"跨三个数量级恒定在 9.9 GB/s"，容易推出
//   "每个字节都必须跨过 PCIe"。但这个推论太满了 —— L2 其实会缓存，
//   只是 add 内核的 working set（12 MB）远超 L2（1 MB）才看不出效果。
//
// 测法：让内核在**一次启动**里对同一块缓冲区反复读很多遍（passes 次），
// 摊薄启动开销，测到纯粹的读取带宽。
//
// 本机实测（GTX 1650，L2 = 1 MB）：
//     缓冲区     读带宽      相对 L2
//     256 KB    204 GB/s    完全装得下
//       1 MB    198 GB/s    刚好装满   ← 分界线精确落在 L2 容量上
//       4 MB     36 GB/s    部分命中
//      64 MB   6.67 GB/s    完全装不下，退化为纯 PCIe
//
//   managed 38.59 GB/s  vs  cudaHostAlloc(Mapped) 40.51 GB/s，且 UVA 下指针值相同
//   → 在这台机器上 managed 基本等价于 zero-copy 锁页映射内存
//
// 编译运行： nvcc -O3 readcache.cu -o readcache && ./readcache
// 注意 64 MB 一档需要约 64 MB 显存

#include <cuda_runtime.h>
#include <cstdio>

__global__ void readk(int n, int passes, const float* x, float* sink)
{
    float acc = 0.0f;
    for (int p = 0; p < passes; ++p) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int s = gridDim.x * blockDim.x;
        for (; i < n; i += s) acc += x[i];
    }
    if (acc == -12345.0f) *sink = acc;   // 防止编译器把整个循环优化掉
}

// 返回：每遍扫描的等效读带宽（GB/s）
static double bandwidth(const float* x, int n, int passes, int reps)
{
    float* sink = nullptr;
    cudaMalloc(&sink, 4);

    // grid 要跟问题规模匹配。固定 1024 个 block 时，64 KB 那种小缓冲区会让
    // 94% 的线程空转，测出来的是调度开销而不是带宽 —— 第一版就栽在这。
    int grid = (n + 255) / 256;
    if (grid > 1024) grid = 1024;
    if (grid < 1)    grid = 1;

    readk<<<grid,256>>>(n, passes, x, sink);      // 预热
    cudaDeviceSynchronize();

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int r = 0; r < reps; ++r) readk<<<grid,256>>>(n, passes, x, sink);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(sink);

    double perPassMs = ms / reps / passes;
    double bytes = (double)n * 4;
    return bytes / 1e9 / (perPassMs / 1e3);
}

__global__ void setk(int n, float v, float* x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = gridDim.x * blockDim.x;
    for (; i < n; i += s) x[i] = v;
}

int main()
{
    const size_t KB = 1024, MB = 1024 * 1024;

    printf("=== 问题 1：读来的数据会不会被 L2 缓存？ ===\n");
    printf("（先查 cudaDeviceProp::l2CacheSize，对照着看分界线在哪）\n\n");
    printf("%-14s %-16s %s\n", "缓冲区大小", "等效读带宽", "判读");
    printf("--------------------------------------------------------------\n");

    size_t sizes[] = { 64 * KB, 256 * KB, 1 * MB, 4 * MB, 64 * MB };
    for (size_t bytes : sizes)
    {
        int n = (int)(bytes / 4);
        float* x = nullptr;
        if (cudaMallocManaged(&x, bytes) != cudaSuccess) { printf("分配失败\n"); continue; }

        setk<<<1024,256>>>(n, 1.0f, x);
        cudaDeviceSynchronize();

        int passes = (int)((32.0 * MB) / bytes);
        if (passes < 1) passes = 1;
        if (passes > 2000) passes = 2000;

        double gb = bandwidth(x, n, passes, 5);
        printf("%-14zu %-16.2f %s\n",
               bytes / KB, gb,
               (gb > 20.0) ? "← 远快于 PCIe，命中 L2" : "→ PCIe 速度，L2 装不下");

        cudaFree(x);
    }

    printf("\n=== 问题 2：managed 和「锁页映射内存」是不是一回事？ ===\n\n");
    printf("%-30s %-16s %s\n", "内存类型", "读带宽", "说明");
    printf("--------------------------------------------------------------\n");

    const size_t sz = 4 * MB;
    const int n2 = (int)(sz / 4);

    float* m = nullptr;
    cudaMallocManaged(&m, sz);
    setk<<<1024,256>>>(n2, 1.0f, m);
    cudaDeviceSynchronize();
    printf("%-30s %-16.2f GB/s\n", "cudaMallocManaged", bandwidth(m, n2, 8, 5));

    float* h = nullptr;
    if (cudaHostAlloc(&h, sz, cudaHostAllocMapped) == cudaSuccess) {
        float* d = nullptr;
        cudaHostGetDevicePointer(&d, h, 0);   // UVA 下 d == h，显式调用更通用
        for (int i = 0; i < n2; ++i) h[i] = 1.0f;
        printf("%-30s %-16.2f GB/s\n", "cudaHostAlloc(Mapped)", bandwidth(d, n2, 8, 5));
        printf("  指针值: host=%p  device=%p  →  %s\n", (void*)h, (void*)d,
               ((void*)h == (void*)d) ? "相同（UVA）" : "不同");
        cudaFreeHost(h);
    } else {
        printf("cudaHostAlloc 失败\n");
    }

    cudaFree(m);
    printf("\n参照：PCIe 3.0 x8 实测可达 ≈ 6.6 GB/s（单向）\n");
    return 0;
}
