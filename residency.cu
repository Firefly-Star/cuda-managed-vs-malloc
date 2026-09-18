// residency.cu  —  决定性判据：managed 数据到底有没有进过显存？
//
// 对应笔记：docs/cuda-managed-vs-malloc.md §4.2
//
// 为什么需要这个实验：
//   「按页迁移」和「按次远程取」都会表现为「耗时正比于访问量」，
//   所以 um_model.cu 那个实验分不开它们。两者的根本差别在于——
//
//     按页迁移   → 数据会【住进显存】，空闲显存应该减少
//     按次远程取 → 数据始终留在【主机内存】，空闲显存不变
//
//   所以直接查显存用量。同时用 cudaMalloc 做对照，标定「真的占显存」长什么样。
//   对照很关键：没有它，看到「空闲显存不变」时就没法排除「测量方法本身不灵」。
//
// 本机实测（GTX 1650 4GB，240 MB 缓冲）：
//     cudaMalloc 240MB 之后        空闲 3057 MB   ← 精确少 240 MB，尺子是准的
//     cudaFree 之后                空闲 3297 MB
//     cudaMallocManaged 240MB 之后 空闲 3297 MB   ← 一点没少
//     GPU 内核全量碰过之后         空闲 3297 MB   ← 一点没少
//     CPU 全量碰过之后             空闲 3297 MB
//     GPU 再碰一次之后             空闲 3297 MB
//
//   → 数据从来没进过显存，一次都没有。按页迁移被排除。
//
// 编译运行： nvcc -O3 residency.cu -o residency && ./residency
// 注意 240 MB 那档需要约 240 MB 显存，小显存卡请调小 bytes

#include <cuda_runtime.h>
#include <cstdio>

static void report(const char* label)
{
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    printf("  %-34s 空闲 %6.0f MB\n", label, freeB / 1048576.0);
}

__global__ void setk(int n, float v, float* x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = gridDim.x * blockDim.x;
    for (; i < n; i += s) x[i] = v;
}

int main()
{
    const size_t MB = 1024ull * 1024;
    const size_t bytes = 240 * MB;
    const int n = (int)(bytes / 4);

    printf("=== 基准 ===\n");
    report("起始");

    // 对照：cudaMalloc 肯定占显存。这一组用来验证 cudaMemGetInfo 测得准。
    printf("\n=== A: cudaMalloc（对照）===\n");
    float* d = nullptr;
    cudaMalloc(&d, bytes);
    report("cudaMalloc 240MB 之后");
    cudaFree(d);
    report("cudaFree 之后");

    // 正题：managed 内存会不会占显存？
    printf("\n=== B: cudaMallocManaged ===\n");
    float* m = nullptr;
    cudaMallocManaged(&m, bytes);
    report("cudaMallocManaged 240MB 之后");

    setk<<<1024,256>>>(n, 1.0f, m);
    cudaDeviceSynchronize();
    report("GPU 内核全量碰过之后");

    for (int i = 0; i < n; ++i) m[i] = 2.0f;      // CPU 侧也碰一遍
    report("CPU 全量碰过之后");

    setk<<<1024,256>>>(n, 3.0f, m);
    cudaDeviceSynchronize();
    report("GPU 再碰一次之后");

    printf("\n=== 判读 ===\n");
    printf("  B 段空闲显存明显下降（接近 240MB）→ 数据搬进显存了 = 按页迁移\n");
    printf("  B 段空闲显存基本不变              → 数据留在主机内存 = 按次远程取\n");
    printf("  （先看 A 段：若 cudaMalloc 那行没少 240MB，说明测量方法不灵，结论不可信）\n");

    cudaFree(m);
    return 0;
}
