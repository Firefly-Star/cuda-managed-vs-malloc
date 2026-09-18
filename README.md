# cuda-managed-vs-malloc

`cudaMallocManaged` 与 `cudaMalloc` + `cudaMemcpy` 的实测对比程序。

> 起因是照着 NVIDIA 的 [Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) 做练习，顺手算了下带宽，发现数字对不上，往下挖出的一串实验。

## 一句话结论

**在 Windows 系平台（原生 Windows / WSL2）上，`cudaMallocManaged` 分配的数据永远不会进显存** —— GPU 访问它要跨 PCIe 回主机内存取。实测带宽恒为 9.9 GB/s，而 `cudaMalloc` + `cudaMemcpy` 能跑到 130 GB/s，**13 倍差距**。

⚠️ 所有绝对数字依赖硬件，**有意义的都是比值**。

---

## 文件对照表

| 文件 | 对应笔记章节 | 验证什么 | 关键结论 |
|---|---|---|---|
| `devquery.cu` | §4.3 | 设备的 Unified Memory 能力 | `concurrentManagedAccess = 0` |
| `pagefault_test.cu` | §3 猜想① | 是不是缺页开销 | ❌ 第 1 次 3.929 ms，之后 3.65~3.78 ms，无热身效应 |
| `occupancy_test.cu` | §3 猜想② | 是不是占用率不足 | ⚠️ 1→4 blocks 有 3 倍，但 4 blocks 后就饱和 |
| `memloc_test.cu` | §3 猜想③ | 是不是内存位置问题 | ✅ **12.3 倍差距，命中** |
| `bwbench.cu` | §4.1 | 跨规模扫带宽 | managed 恒为 9.9 GB/s，cudaMalloc 达 131~142 GB/s |
| `um_model.cu` | §4.2 | 按需访问还是整体搬运 | 固定访问量、分配大小涨 30 倍，耗时不变 → 按需 |
| `residency.cu` | §4.2 | **决定性的**：数据进没进过显存 | 240 MB managed 全程占 0 显存 → 不是按页迁移 |
| `readcache.cu` | §4.1 | L2 会不会缓存跨 PCIe 取来的数据 | 分界线精确落在 1 MB（=L2 容量）；managed ≈ zero-copy |
| `pcie_bw.cu` | §4.1 | 实测 PCIe 链路可达带宽 | 6.6 GB/s（HtoD 6.66 / DtoH 6.59）—— managed 读方向打满其 98.6% |
| `ladder.cu` | §7 | 完整优化阶梯 | 串行/managed → grid-stride/显存，**~4700 倍** |

**建议的阅读顺序**：`ladder.cu`（看结论）→ `bwbench.cu`（看量化）→ 三个 `*_test`（看怎么排除其他可能）。

---

## 编译运行

```bash
make            # 编译全部
make ladder     # 只编译某一个
./ladder
```

或者手动：

```bash
nvcc -O3 ladder.cu -o ladder && ./ladder
```

想拿内核级明细，配合 nsys：

```bash
nsys profile --trace=cuda --sample=none -o p ./ladder
nsys stats --force-export=true -r cuda_gpu_kern_sum p.nsys-rep
```

> `-O3` 别省。省略的话 nvcc 虽然默认也是优化编译，但显式写上更保险 —— 用 `-G` 调试编译会让数字完全失真。

---

## 本机复现的参考数字

环境：WSL2 (Ubuntu 22.04) + CUDA 13.3 + GTX 1650 4GB (cc 7.5, 14 SM, L2 1MB)

`ladder.cu` 输出：

```
版本                               耗时       带宽   相对 A
--------------------------------------------------------------------
A 串行  + 1 block    + managed  436.143 ms       0.0 GB/s         1x
B 跨步  + 1 block    + managed    3.647 ms       3.3 GB/s       120x
C 跨步  + 3907 block + managed    1.218 ms       9.9 GB/s       358x
D 跨步  + 3907 block + cudaMalloc    0.092 ms     130.4 GB/s      4738x
```

`bwbench.cu` 输出：

```
N            流量     managed        cudaMalloc     倍数
--------------------------------------------------------------
1000000      0.0        1.22ms 9.9GB/s 0.09ms 138.7GB/s 14.05x
8000000      0.1        9.67ms 9.9GB/s 0.67ms 142.5GB/s 14.36x
64000000     0.8       77.41ms 9.9GB/s 5.96ms 128.9GB/s 12.99x
```

（`流量` 那一列的格式化有点问题，显示成了 0.0 —— 实际分别是 12 / 96 / 768 MB。）

---

## 复现注意事项

1. **确认 GPU 没被降频**。功耗墙或温度限制会把所有数字压扁，而且会把 13 倍的差距伪装成 3 倍。跑之前先看：
   ```bash
   nvidia-smi --query-gpu=pstate,clocks.sm,clocks.mem --format=csv
   ```
   应该是 `P0` 且时钟接近上限；`P8` / 300MHz 说明在降频，测了没意义。

2. **绝对数字依赖硬件**，别照搬。有意义的是**比值**（managed vs cudaMalloc 的倍数、各级台阶的倍数）。

3. **`bwbench.cu` 的 N=64e6 需要约 512 MB 显存**。小显存卡请把 `Ns[]` 和 reps 调小。

4. **`concurrentManagedAccess` 是因平台而异的**。在原生 Linux（cc≥6.0 + 核支持）上它是 1，managed 的行为会完全不同 —— 那时的结论也会不一样。本文件夹的所有结论都限定在 **Windows 系统平台（含 WSL2）**。

5. **CUDA 13 的 API 变更**：`cudaMemPrefetchAsync` / `cudaMemAdvise` 的 location 参数从裸 device id 变成了按值传递的 `cudaMemLocation`。用旧写法编译不过。

---

## 一句话结论

在 Windows 系平台上，`cudaMallocManaged` 的数据永远不会进显存，每次访存都要跨 PCIe 回主机内存取。实测带宽恒为 ~9.9 GB/s，而 `cudaMalloc` + `cudaMemcpy` 能跑到 ~131 GB/s —— **13 倍差距**。所以：**用显式内存管理。**
