// pcie_bw.cu  —  实测 PCIe 链路的**可达**带宽
//
// 对应笔记：docs/cuda-managed-vs-malloc.md §4.1
//
// 为什么需要这个：说"managed 内存走 PCIe 所以慢"时，得有个分母才知道到底有多慢。
// 拿规格表的理论值当分母是不对的 —— PCIe 3.0 x8 理论 7.88 GB/s 里有一部分是
// 协议本身的开销（TLP 包头、ACK），谁也拿不到，用理论值算会显得"还有余地"。
// 必须实测这条链路实际能跑多快，才是公平的对比基准。
//
// 做法：锁页内存（cudaHostAlloc）+ cudaMemcpy。锁页内存让 DMA 能直读主机内存，
// 避开驱动内部那次"复制到临时缓冲区"的中转拷贝，否则测出来会偏慢一倍。
//
// 本机实测（GTX 1650，PCIe 3.0 x8）：
//     64 MB   HtoD 6.63 GB/s   DtoH 6.42 GB/s
//     256 MB  HtoD 6.66 GB/s   DtoH 6.59 GB/s
//   → 可达带宽 ≈ 6.6 GB/s，约为规格理论值 7.88 GB/s 的 84%（正常效率）
//
// 有了这个分母，managed 内核的读数才有意义：
//     managed 读方向 6.57 GB/s ÷ 实测可达 6.66 GB/s = 98.6% —— 完全打满
//
// 前置查询链路规格： nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv
//
// 编译运行： nvcc -O3 pcie_bw.cu -o pcie_bw && ./pcie_bw
// 注意 256 MB 那档需要约 512 MB 显存 + 256 MB 锁页内存，小显存/小内存机器请调小 sizes[]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

static double bench(void* d, void* h, size_t bytes, int reps, cudaMemcpyKind kind)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) cudaMemcpy(d, h, bytes, kind);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / reps;
}

int main()
{
    const size_t MB = 1024ull * 1024;
    size_t sizes[] = { 64 * MB, 256 * MB };
    const int reps = 10;

    printf("%-10s %-22s %-22s\n", "数据量", "HtoD (写向 GPU)", "DtoH (从 GPU 读)");
    printf("------------------------------------------------------------\n");

    for (size_t bytes : sizes)
    {
        void* h = nullptr;
        void* d = nullptr;
        if (cudaHostAlloc(&h, bytes, cudaHostAllocDefault) != cudaSuccess) {
            printf("cudaHostAlloc 失败（锁页内存不够？）\n"); return 1;
        }
        if (cudaMalloc(&d, bytes) != cudaSuccess) {
            printf("cudaMalloc 失败（显存不够？）\n"); cudaFreeHost(h); return 1;
        }
        memset(h, 1, bytes);

        double gb = bytes / 1e9;

        // 预热一次，把首次建链/缺页的开销排除掉
        cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        double ms_h2d = bench(d, h, bytes, reps, cudaMemcpyHostToDevice);
        double ms_d2h = bench(h, d, bytes, reps, cudaMemcpyDeviceToHost);

        printf("%-10zu %8.3f ms %7.2f GB/s   %8.3f ms %7.2f GB/s\n",
               bytes / MB,
               ms_h2d, gb / (ms_h2d / 1e3),
               ms_d2h, gb / (ms_d2h / 1e3));

        cudaFree(d);
        cudaFreeHost(h);
    }

    printf("\n对照：PCIe 3.0 x8 规格理论值 = 7.88 GB/s（实测通常只有 85~95%%）\n");
    return 0;
}
