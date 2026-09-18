// bwbench.cu  —  带宽基准：managed vs cudaMalloc，扫三个规模
//
// 对应笔记：docs/cuda-managed-vs-malloc.md §4.1
//
// 同一个 add 内核，同一份代码，只换内存来源，N 从 1e6 扫到 64e6。
// 实测（GPU 正常频率下）：
//     N=1e6   12 MB   managed  1.22 ms ( 9.9 GB/s)   cudaMalloc 0.09 ms (138.7 GB/s)
//     N=8e6   96 MB   managed  9.67 ms ( 9.9 GB/s)   cudaMalloc 0.67 ms (142.5 GB/s)
//     N=64e6 768 MB   managed 77.41 ms ( 9.9 GB/s)   cudaMalloc 5.96 ms (128.9 GB/s)
//
// 关键现象：managed 的带宽跨三个数量级恒定不变 —— 这就是"每个字节都必须
// 跨过 PCIe"的证据。如果数据能进显存，规模小的时候必然更快。
//
// 编译运行： nvcc -O3 bwbench.cu -o bwbench && ./bwbench
// 注意 N=64e6 需要 ~512MB 显存，小显存卡请把 Ns[] 调小。

#include <cuda_runtime.h>
#include <cstdio>

__global__ void add(int n, float* x, float* y)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] += x[i];
}

__global__ void setk(int n, float value, float* x)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = value;
}

static double time_add(int N, float* x, float* y, int G, int reps)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int r = 0; r < reps; ++r) add<<<G, 256>>>(N, x, y);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / reps;
}

int main()
{
    int Ns[] = {1000000, 8000000, 64000000};

    printf("%-12s %-10s %-14s %-14s %-10s\n",
           "N", "流量", "managed", "cudaMalloc", "倍数");
    printf("--------------------------------------------------------------\n");

    for (int k = 0; k < 3; ++k)
    {
        int N = Ns[k];
        size_t bytes = (size_t)N * sizeof(float);
        int G = 1024;
        int reps = (N <= 1000000) ? 50 : (N <= 8000000 ? 20 : 10);

        // ---- managed ----
        float *mx = nullptr, *my = nullptr;
        if (cudaMallocManaged(&mx, bytes) != cudaSuccess ||
            cudaMallocManaged(&my, bytes) != cudaSuccess) {
            printf("N=%d managed 分配失败\n", N); continue;
        }
        setk<<<G,256>>>(N, 1.0f, mx);
        setk<<<G,256>>>(N, 1.0f, my);
        cudaDeviceSynchronize();
        double ms_m = time_add(N, mx, my, G, reps);

        // ---- cudaMalloc 显存 ----
        float *dx = nullptr, *dy = nullptr;
        if (cudaMalloc(&dx, bytes) != cudaSuccess ||
            cudaMalloc(&dy, bytes) != cudaSuccess) {
            printf("N=%d 显存分配失败\n", N); cudaFree(mx); cudaFree(my); continue;
        }
        setk<<<G,256>>>(N, 1.0f, dx);
        setk<<<G,256>>>(N, 1.0f, dy);
        cudaDeviceSynchronize();
        double ms_d = time_add(N, dx, dy, G, reps);

        // add 的访存流量 = 读 x + 读 y + 写 y = 3 * bytes
        double gb = 3.0 * bytes / 1e9;
        printf("%-12d %-10.1f %-14s %-14s %.2fx\n",
               N, gb,
               ([](double ms, double g){ static char b[32]; snprintf(b,32,"%.2fms %.1fGB/s",ms,g/(ms/1e3)); return b; })(ms_m, gb),
               ([](double ms, double g){ static char b[32]; snprintf(b,32,"%.2fms %.1fGB/s",ms,g/(ms/1e3)); return b; })(ms_d, gb),
               ms_m / ms_d);

        cudaFree(mx); cudaFree(my); cudaFree(dx); cudaFree(dy);
    }
    return 0;
}
