// pagefault_test.cu  —  猜想① 验证：是不是 Unified Memory 的缺页开销
//
// 对应笔记：docs/cuda-managed-vs-malloc.md §3「猜想 ①」
//
// 同一个 add 内核、同一块 managed 数据，连跑 10 次。
// 如果缺页是主因，第一次该显著慢于后续。
// 实测：第 1 次 3.929 ms，第 2~10 次 3.65~3.78 ms —— 没有热身效应，猜想排除。
//
// 注意：每次 add 之后都要 cudaDeviceSynchronize()，否则内核会排队重叠，
// 测出来的是启动开销而不是执行时间。
//
// 编译运行： nvcc -O3 pagefault_test.cu -o pagefault_test && ./pagefault_test

#include <iostream>

__global__
void add(int n, float* x, float* y)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] += x[i];
}

__global__
void set(int n, float value, float* x)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = value;
}

int main()
{
    float *x, *y;
    int N = 1e6;
    cudaMallocManaged(&x, sizeof(float) * N);
    cudaMallocManaged(&y, sizeof(float) * N);

    set<<<1,256>>>(N, 1.0f, x);
    set<<<1,256>>>(N, 1.0f, y);
    cudaDeviceSynchronize();

    // 同一个 add 内核连跑 10 次：只改"重复次数"，不改数据大小
    for (int r = 0; r < 10; ++r)
    {
        add<<<1, 256>>>(N, x, y);
        cudaDeviceSynchronize();
    }

    std::cout << "y[0] = " << y[0] << "\n";
    cudaFree(x);
    cudaFree(y);
}
