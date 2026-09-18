# 配套笔记：docs/cuda-managed-vs-malloc.md
# 用法：make / make ladder / make run / make clean

NVCC  := nvcc
FLAGS := -O3

TARGETS := devquery pagefault_test occupancy_test memloc_test bwbench um_model residency readcache pcie_bw ladder

.PHONY: all run clean $(TARGETS)

all: $(TARGETS)

# 每个 .cu 编译成同名可执行文件
$(TARGETS): %: %.cu
	$(NVCC) $(FLAGS) $< -o $@

# 依次跑一遍（ladder 和 bwbench 会打印汇总表）
run: all
	@for t in $(TARGETS); do \
		echo "========== $$t =========="; \
		./$$t || echo "(退出码 $$?)"; \
		echo; \
	done

clean:
	rm -f $(TARGETS) *.nsys-rep *.sqlite *.qdstrm
