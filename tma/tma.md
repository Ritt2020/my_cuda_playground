# TMA
> 本文档将详细讲解NVIDIA HOPPER架构引入的Tensor Memory Access。包含了从学习到使用的一切。
> 看到这里了，不点个star吗？
## TMA是什么

TMA（Tensor Memory Accelerator）是 NVIDIA 在 Hopper 架构（H100，SM90）中首次引入的专用异步拷贝硬件单元，用于在 **全局内存（Global Memory）** 和 **共享内存（Shared Memory）** 之间进行1D–5D张量的高吞吐批量数据搬运，并支持多播（multicast）以及从SMEM到GMEM的带规约写入（add/min/max/按位与/或）。编程上通过 cuTensorMap（张量映射描述符）与 PTX 指令 `cp.async.bulk.tensor`/`cp.reduce.async.bulk.tensor` 等配合使用，并以异步代理与 mbarrier/fence 机制完成一致性与同步。

- 提出/发布时间：2022 年随 NVIDIA Hopper 架构发布（GTC）
- 支持的架构：Hopper（SM90/SM90a）。前代 Ampere 仅有 `cp.async` 异步拷贝，不包含 TMA 单元
- 对应 CUDA 版本：
  - 能编译/运行 Hopper（`sm_90`）从 CUDA 11.8 起（提供 Hopper 架构支持）
  - 使用 TMA 相关 PTX 指令/工具链（如 `cp.async.bulk.tensor`、cuTensorMap 等）实际需要 CUDA 12.0+（对应 PTX ISA 8.5 及以上）

## TMA过程讲解

TMA 有两条最常用路径：
- TMA Load：GMEM → SMEM（PTX 走 `cp.async.bulk.tensor.*.global.shared`，用 mbarrier 完成通知）
- TMA Store：SMEM → GMEM（PTX 走 `cp.async.bulk.tensor.*.shared.global`，在写前使用 `fence.proxy.async.shared::cta` 保证可见性；可选规约）

### 同步机制讲解

- 异步代理与一致性：TMA 在“异步代理（async proxy）”执行，与常规线程的“通用代理（generic proxy）”不同步，需要显式同步来建立先后关系与可见性。
- GMEM→SMEM（TMA Load）的完成同步：
  - 使用异步事务屏障 mbarrier。典型序列（由单线程发起）：
    1) 初始化 mbarrier（设置到达计数、起始相位）
    2) 通过 `expect_tx/complete_tx::bytes` 指明本次传输字节数
    3) 发起 `cp.async.bulk.tensor`（绑定该 mbarrier）
    4) CTA 内线程在需要使用数据前 `wait` 该 mbarrier，相位匹配后才能读取 SMEM
  - 在 C++ CUDA API 中，可通过 `cuda::barrier` 与 `cuda::memcpy_async(..., barrier)` 完成上述语义：发起拷贝后，线程组调用 `bar.arrive()`/`bar.wait(token)` 等待完成。
- SMEM→GMEM（TMA Store）的顺序保证：
  - 在发起 Store 前，使用 `fence.proxy.async.shared::cta`（或等价 API 封装）建立“通用代理对共享内存的写”先于“异步代理读共享内存并写回全局”的顺序，避免竞争。
  - 若需要在内核内等待 Store 完成再复用 SMEM/依赖结果，可使用“commit/wait（bulk-group）”或借助更高层 `cuda::pipeline`；简单场景也可依赖内核结束/设备同步作为完成点。

要点：
- 只有 1 个线程（通常 `threadIdx.x == 0`）发起 TMA 请求；其余线程通过屏障/同步等待完成。
- 对齐/步长要求：TMA 期望 16B 对齐，且非连续维度步长通常需为 16B 的整数倍。C++ API 提供 `cuda::aligned_size_t<16>(bytes)` 显式声明传输字节对齐。

### 单线程拷贝（结合仓库示例）

示例文件：`my_cuda_playground/tma/test_cp.async.bulk/test.cu` 展示了 GMEM→SMEM 的最小可用骨架，并将 SMEM 内容再写到另一个 GMEM 缓冲区以便主机验证。

关键片段与解释：

```12:34:/sharedir44-48/wanghy/my_cuda_playground/tma/test_cp.async.bulk/test.cu
__global__ void test_tma_copy_to_shared(uint32_t* a, uint32_t* b) {
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;                      // 1) CTA 级 mbarrier（C++ 封装）
    if(threadIdx.x == 0){
        init(&bar, blockDim.x);                  // 设定等待参与者数量
        fence_proxy_async_shared();              // 建立与 async proxy 的顺序关系
    }
    __syncthreads();

    alignas(16) __shared__ uint32_t shared[8];   // 2) 对齐的 SMEM 目标缓冲区

    if(threadIdx.x == 0){                        // 3) 单线程发起 TMA Load
        cuda::memcpy_async(shared, a, cuda::aligned_size_t<16>(sizeof(shared)), bar);
    }
    barrier::arrival_token token = bar.arrive(); // 4) CTA 到达并等待完成
    bar.wait(std::move(token));
    __syncthreads();

    if(threadIdx.x == 0){                        // 5) 使用 SMEM 数据（复制回 GMEM 供主机校验）
        for(uint32_t i = 0; i < 8; i++) {
            b[i] = shared[i];
        }
    }
}
```

- `bar`/`cuda::memcpy_async(..., bar)` 实现了“发起异步传输 + 完成等待”的 mbarrier 语义。
- `alignas(16)` 与 `cuda::aligned_size_t<16>` 明确了 TMA 的 16B 对齐约束。
- 只有 `threadIdx.x == 0` 发起 TMA；`__syncthreads()` 保证其他线程在使用 SMEM 前已经完成等待与可见性。

## 如何使用

### CUDA API实现

下面给出两条常见路径在 CUDA C++ API 下的最小模板。注意：仅单线程发起，其他线程通过屏障等待；并严格满足 16B 对齐与步长要求。

#### 从全局内存拷贝到共享内存

适用于将 GMEM 的一段（或张量切片）搬运到 CTA 的 SMEM，以供后续计算消费。

```cpp
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ __forceinline__ void fence_proxy_async_shared(){
    asm volatile("fence.proxy.async.shared::cta;");
}

template <class T>
__global__ void tma_load_gmem_to_smem(const T* __restrict__ g_src,
                                      size_t count_elems) {
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        fence_proxy_async_shared();                 // 与 async proxy 建立顺序
    }
    __syncthreads();

    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    size_t bytes = count_elems * sizeof(T);

    if (threadIdx.x == 0) {
        cuda::memcpy_async(smem, g_src, cuda::aligned_size_t<16>(bytes), bar);
    }
    auto token = bar.arrive();                      // 等待拷贝完成
    bar.wait(std::move(token));
    __syncthreads();

    // 此后 smem 中数据对 CTA 可见，可安全消费
}
```

要点：
- 若要做 2D+ 张量切片并自动处理越界/步长，可结合 cuTensorMap/CuTe/CUTLASS 生成 TMA 描述符；本文先用线性段演示最小可用形态。
- 若将继续进行“生产者-消费者”流水线，可用 `cuda::pipeline` 管理多阶段 mbarrier；本文示例用最简单的 `cuda::barrier`。

#### 从共享内存拷贝到全局内存

适用于将 CTA 在 SMEM 内产生/规整好的块写回 GMEM。此方向关键在发起前对 SMEM 写入做“代理栅栏”。

```cpp
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ __forceinline__ void fence_proxy_async_shared(){
    asm volatile("fence.proxy.async.shared::cta;");
}

template <class T>
__global__ void tma_store_smem_to_gmem(T* __restrict__ g_dst,
                                       size_t count_elems) {
#pragma nv_diag_suppress static_var_with_dynamic_init
    // 可选：若希望在内核内等待写回完成再复用 SMEM，可同样使用 barrier
    __shared__ barrier bar;
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
    }
    __syncthreads();

    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    size_t bytes = count_elems * sizeof(T);

    // 先由所有线程生产/写入 SMEM
    for (size_t i = threadIdx.x; i < count_elems; i += blockDim.x) {
        smem[i] = static_cast<T>(i);                // 示例：写入某个模式
    }
    __syncthreads();

    // 建立“通用代理写 SMEM”先于“异步代理读 SMEM 并写 GMEM”的顺序
    if (threadIdx.x == 0) {
        fence_proxy_async_shared();
        // 方式 A：不在内核内等待，直接发起（适用于内核即将结束/无需复用 SMEM）
        // cuda::memcpy_async(g_dst, smem, cuda::aligned_size_t<16>(bytes));

        // 方式 B：在内核内等待完成后再继续（例如复用 SMEM）
        cuda::memcpy_async(g_dst, smem, cuda::aligned_size_t<16>(bytes), bar);
    }
    // 若选择方式 B：
    auto token = bar.arrive();
    bar.wait(std::move(token));
    __syncthreads();
}
```

说明：
- TMA Store 的“正确性关键”是发起前的 `fence.proxy.async.shared::cta`，确保 SMEM 中由通用代理产生的数据对异步代理可见。
- 是否在内核内等待（方式 B）取决于是否要立即复用 SMEM 或依赖已写回 GMEM 的结果继续计算；若内核结束即可让运行时/设备同步承担完成点，方式 A 更简洁。
- 需要规约写回（add/min/max/按位与/或）时，可在 PTX/CUTLASS 里选择 `cp.reduce.async.bulk.tensor.*` 或对应 API 封装。

### 编译与运行（基于仓库示例）

- 硬件/软件前提：Hopper（SM90/SM90a），CUDA 12+
- 推荐编译命令（基于示例 `test_cp.async.bulk/test.cu`）：

```bash
nvcc -std=c++17 -arch=sm_90 -O2 \
  /sharedir44-48/wanghy/my_cuda_playground/tma/test_cp.async.bulk/test.cu \
  -o /sharedir44-48/wanghy/my_cuda_playground/tma/test_cp.async.bulk/test && \\
/sharedir44-48/wanghy/my_cuda_playground/tma/test_cp.async.bulk/test
```

- 预期输出：先打印 host 侧 `a[i]`，随后打印由设备拷回的 `b[i]`（应与 `a[i]` 一致），验证 GMEM→SMEM→GMEM 路径正确。

### PTX实现

- GMEM→SMEM 的底层可映射为：
  `cp.async.bulk.tensor.{1d..5d}.shared::cluster.global.mbarrier::complete_tx::bytes`
- SMEM→GMEM 的底层可映射为：
  `cp.async.bulk.tensor.{1d..5d}.global.shared::cta[.bulk_group]`，写前配合 `fence.proxy.async.shared::cta`
- 规约写回使用：
  `cp.reduce.async.bulk.tensor.{op}`（`add|min|max|and|or`）

在高层库（CUTLASS/CuTe）中，上述均由 `cute::copy(tma_*, ...)` 等接口封装。

### CUTLASS实现

- CUTLASS 提供 SM90 TMA Load/Store 的完整封装（含多播、流水线、规约等），并将 mbarrier/fence 细节隐藏在 `Pipeline` 与 `TiledCopy` 等抽象下；适合工程化 GEMM/Conv 等场景。
- 参考 Colfax 教程与 CUTLASS 示例获取 2D/3D 张量切片与集成范式。

参考：
- NVIDIA Hopper Tuning Guide（1.4.1.2 Tensor Memory Accelerator）【`https://docs.nvidia.com/cuda/hopper-tuning-guide/`】
- Hopper Compatibility Guide（CUDA 11.8 起支持 `sm_90`）【`https://docs.nvidia.com/cuda/hopper-compatibility-guide/`】
- PTX ISA（包含 `cp.async.bulk.tensor` 等指令，见 Asynchronous copy/Tensor copy 部分）【`https://docs.nvidia.com/cuda/parallel-thread-execution/`】
- CUTLASS/Colfax：Mastering the NVIDIA Tensor Memory Accelerator (TMA)【`https://research.colfax-intl.com/tutorial-hopper-tma/`】
- PyTorch Blog：Deep Dive on the Hopper TMA Unit（背景与用法示例）【`https://pytorch.org/blog/hopper-tma-unit/`】
