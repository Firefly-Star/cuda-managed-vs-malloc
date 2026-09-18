// devquery.cu  —  查设备的 Unified Memory 能力
//
// 对应笔记：docs/cuda-managed-vs-malloc.md §4.3
//
// 关键看 concurrentManagedAccess —— 本机是 0，意味着没有 GPU 侧页错误机制，
// 数据也就没有"搬到显存"的通路。（managedMemory 是 1，说明 managed 本身可用，
// 但只有"基础模型"，没有按需迁移。）
//
// 编译运行： nvcc devquery.cu -o devquery && ./devquery

#include <cuda_runtime.h>
#include <cstdio>

int main()
{
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    printf("=== 设备 ===\n");
    printf("name                        : %s\n", p.name);
    printf("compute capability          : %d.%d\n", p.major, p.minor);
    printf("SM 数量                     : %d\n", p.multiProcessorCount);
    printf("显存                        : %.0f MB\n", p.totalGlobalMem / 1048576.0);
    printf("memoryBusWidth              : %d bit\n", p.memoryBusWidth);
    printf("l2CacheSize                 : %.0f KB\n", p.l2CacheSize / 1024.0);

    // SM 是硬件；block / thread 是软件概念，由 <<<gridDim, blockDim>>> 在启动时指定。
    // 下面这些上限决定了"一个 SM 最多能同时装下多少个 block"。
    printf("\n=== 并行度上限（硬件约束）===\n");
    printf("每块最多线程数              : %d\n", p.maxThreadsPerBlock);
    printf("每个 SM 最多线程数          : %d\n", p.maxThreadsPerMultiProcessor);
    printf("warp 大小                   : %d\n", p.warpSize);
    int mb = 0;
    cudaDeviceGetAttribute(&mb, cudaDevAttrMaxBlocksPerMultiprocessor, 0);
    printf("每个 SM 最多 block 数       : %d\n", mb);

    printf("\n=== Unified Memory 相关能力 ===\n");
    printf("managedMemory                     : %d\n", p.managedMemory);
    printf("concurrentManagedAccess           : %d\n", p.concurrentManagedAccess);
    printf("pageableMemoryAccess              : %d\n", p.pageableMemoryAccess);
    printf("directManagedMemAccessFromHost    : %d\n", p.directManagedMemAccessFromHost);

    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrConcurrentManagedAccess, 0);
    printf("\ncudaDevAttrConcurrentManagedAccess: %d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrPageableMemoryAccess, 0);
    printf("cudaDevAttrPageableMemoryAccess   : %d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrManagedMemory, 0);
    printf("cudaDevAttrManagedMemory          : %d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrDirectManagedMemAccessFromHost, 0);
    printf("cudaDevAttrDirectManagedMemFromHost: %d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrUnifiedAddressing, 0);
    printf("cudaDevAttrUnifiedAddressing      : %d\n", v);
    return 0;
}
