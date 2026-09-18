// 优化阶梯：同一个 vector add 问题，四个版本，全部 N=1e6，可横向比较
#include <cuda_runtime.h>
#include <cstdio>

// ---- 串行版：256 个线程各自跑完整循环（= vector_add2 的写法）----
__global__ void add_serial(int n, float* x, float* y)
{
    for (int i = 0; i < n; ++i)
        y[i] += x[i];
}

// ---- 跨步版：只用一个 block（= vector_add_p 的写法）----
__global__ void add_block(int n, float* x, float* y)
{
    int index  = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] += x[i];
}

// ---- 网格跨步版：多 block（工业写法）----
__global__ void add_grid(int n, float* x, float* y)
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

static double bench(void (*launch)(int, float*, float*), int N, int reps,
                    float* x, float* y)
{
    launch(N, x, y);                      // 预热
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int r = 0; r < reps; ++r) launch(N, x, y);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / reps;
}

// A 级的基线用 <<<1,1>>>（真的只用 1 个线程），和 kernels/week1/day1/vector_add.cu
// 里 add<<<1,1>>> 的写法一致。别改成 <<<1,256>>>：那会让 256 个线程跑重复的全量循环，
// 除了"没有并行"之外还叠加了 256 倍的冗余工作，两个因素混在一起，就不是干净的基线了。
static void serial1(int n, float* x, float* y) { add_serial<<<1, 1>>>(n, x, y); }
static void blk1   (int n, float* x, float* y) { add_block <<<1, 256>>>(n, x, y); }
static void grid3907(int n, float* x, float* y) { add_grid <<<3907, 256>>>(n, x, y); }

int main()
{
    const int N = 1000000;
    const size_t bytes = (size_t)N * sizeof(float);
    const double GB = 3.0 * bytes / 1e9;      // 读 x + 读 y + 写 y

    struct { const char* name; double ms; } res[4];

    // ===== managed 内存 =====
    float *mx, *my;
    cudaMallocManaged(&mx, bytes);
    cudaMallocManaged(&my, bytes);
    setk<<<3907,256>>>(N, 1.0f, mx);
    setk<<<3907,256>>>(N, 1.0f, my);
    cudaDeviceSynchronize();

    res[0] = {"A 串行  + 1 block    + managed", bench(serial1,  N, 3, mx, my)};
    res[1] = {"B 跨步  + 1 block    + managed", bench(blk1,     N, 20, mx, my)};
    res[2] = {"C 跨步  + 3907 block + managed", bench(grid3907, N, 20, mx, my)};

    // ===== cudaMalloc 显存 =====
    float *dx, *dy;
    cudaMalloc(&dx, bytes);
    cudaMalloc(&dy, bytes);
    setk<<<3907,256>>>(N, 1.0f, dx);
    setk<<<3907,256>>>(N, 1.0f, dy);
    cudaDeviceSynchronize();

    res[3] = {"D 跨步  + 3907 block + cudaMalloc", bench(grid3907, N, 100, dx, dy)};

    printf("\n%-32s %10s %12s %10s\n", "版本", "耗时", "带宽", "相对 A");
    printf("--------------------------------------------------------------------\n");
    for (int i = 0; i < 4; ++i)
        printf("%-32s %8.3f ms %9.1f GB/s %9.0fx\n",
               res[i].name, res[i].ms, GB / (res[i].ms / 1e3), res[0].ms / res[i].ms);

    cudaFree(mx); cudaFree(my); cudaFree(dx); cudaFree(dy);
    return 0;
}
