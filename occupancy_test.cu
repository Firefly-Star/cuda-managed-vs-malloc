// occupancy_test.cu  —  猜想② 验证：是不是占用率不足
//
// 对应笔记：docs/cuda-managed-vs-malloc.md §3「猜想 ②」
//
// 同一个 grid-stride 内核，工作量不变，只改 grid 大小。
// 实测：1 → 3.838 ms，4 → 1.253 ms，然后 14/56/224/3907 全部稳定在 1.214 ms。
// 4 个 block 就饱和 —— 说明主因不是占用率，而是别处有个更高的天花板。
//
// 编译运行： nvcc -O3 occupancy_test.cu -o occupancy_test && ./occupancy_test
// 看明细：   nsys profile --trace=cuda --sample=none -o p ./occupancy_test
//            sqlite3 p.sqlite "select (end-start)/1e6 from
//              CUPTI_ACTIVITY_KIND_KERNEL order by start;"   # 按启动顺序，每 5 个一组

#include <iostream>

// grid-stride 版本：改 grid 大小，工作量不变
__global__
void add(int n, float* x, float* y)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] += x[i];
}

__global__
void set(int n, float value, float* x)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = value;
}

int main()
{
    float *x, *y;
    int N = 1e6;
    cudaMallocManaged(&x, sizeof(float) * N);
    cudaMallocManaged(&y, sizeof(float) * N);

    set<<<3907,256>>>(N, 1.0f, x);
    set<<<3907,256>>>(N, 1.0f, y);
    cudaDeviceSynchronize();

    int grids[] = {1, 4, 14, 56, 224, 3907};
    for (int g : grids)
    {
        for (int r = 0; r < 5; ++r)
        {
            add<<<g, 256>>>(N, x, y);
            cudaDeviceSynchronize();
        }
    }
    std::cout << "done, y[0] = " << y[0] << "\n";
    cudaFree(x);
    cudaFree(y);
}
