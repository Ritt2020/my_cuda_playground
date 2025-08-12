# TMA: 使用 cp.async.bulk 拷贝到 Shared Memory 示例说明

本目录的 `test.cu` 演示了在 Hopper 架构 GPU 上，使用 CUDA C++ 高层 API `cuda::memcpy_async` 搭配 `cuda::barrier`（mbarrier）驱动 TMA，从全局内存异步拷贝数据到共享内存的最小可运行示例。

### 支持架构与环境要求
- **GPU 架构**: NVIDIA Hopper（SM90 及以上）。例如 H100。
- **CUDA 版本**: CUDA 12+（建议 12.2 及以上）。
- **编译器与标准库**: `nvcc` 搭配 libcu++（包含 `<cuda/barrier>`）。建议使用 `-std=c++17` 或更高标准。

提示：在部分 H100 平台上，如遇到架构相关报错，可将编译选项从 `-arch=sm_90` 切换为 `-arch=sm_90a`。

### 示例亮点（特性）
- **TMA 异步拷贝**: 通过 `cuda::memcpy_async` 触发硬件协助的异步数据传输，减轻线程负担。
- **mbarrier 协同**: 使用 `cuda::barrier` 初始化、`arrive/wait` 协同全体线程，保证拷贝完成后再继续访问共享内存。
- **对齐与一致性**: 使用 `alignas(16)` 的共享内存，并通过 `cuda::aligned_size_t<16>` 指定传输字节数，满足对齐要求；示例中还加入了 `fence.proxy.async.shared::cta` 保证代理内存路径的一致性。
- **简单可验证**: 将全局内存中的 8 个 `uint32_t` 值拷贝到共享内存，再写回到输出数组，便于核对结果。

### 关键代码要点
- 共享内存声明（16B 对齐，示例传输 32B）：
  ```cpp
  alignas(16) __shared__ uint32_t shared[8];
  ```
- 拷贝触发（仅 1 个线程发起，配合 barrier）：
  ```cpp
  if (threadIdx.x == 0) {
      cuda::memcpy_async(shared, a, cuda::aligned_size_t<16>(sizeof(shared)), bar);
  }
  ```
- 同步与一致性：
  ```cpp
  barrier::arrival_token token = bar.arrive();
  bar.wait(std::move(token));
  // 代理路径 fence，确保共享内存可见性
  fence_proxy_async_shared();
  __syncthreads();
  ```

### 编译与运行
假设当前工作目录为仓库根目录：

```bash
# 编译（Hopper/SM90）
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90 \
  my_cuda_playground/tma/test_cp.async.bulk/test.cu \
  -o my_cuda_playground/tma/test_cp.async.bulk/test

# 如果遇到架构报错，可尝试：
# nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a \
#   my_cuda_playground/tma/test_cp.async.bulk/test.cu \
#   -o my_cuda_playground/tma/test_cp.async.bulk/test

# 运行
./my_cuda_playground/tma/test_cp.async.bulk/test
```

预期会打印输入数组 a 与输出数组 b，b 应与 a 一致：
```
a[0] = 0 a[1] = 1 ...
b[0] = 0 b[1] = 1 ...
```

### 常见注意事项
- **对齐要求**: 使用 `cuda::aligned_size_t<16>` 时，源与目的地址及传输尺寸需满足 16B 对齐；示例中通过 `alignas(16)` 和 32B 传输满足要求。
- **参与线程**: `cuda::barrier` 的参与者数量与 `init(&bar, blockDim.x)` 对应；示例中所有线程都会参与 `arrive/wait`。
- **单线程触发拷贝**: 为避免重复传输，通常只需由一个线程（如 `threadIdx.x == 0`）调用 `cuda::memcpy_async`。
- **一致性/fence**: `fence.proxy.async.shared::cta` 有助于确保代理路径访问的一致性；在共享内存读取前进行 `wait + fence + __syncthreads()` 更稳妥。
- **更高维拷贝**: TMA 还支持 2D-5D 的张量搬运（需要张量描述符等更复杂的设置）；本示例聚焦最简 1D 拷贝到 shared 的用法。

### 扩展阅读
- CUDA C++ `libcu++` barrier: `#include <cuda/barrier>`
- Hopper 架构与 TMA（Tensor Memory Accelerator）概览（参见 NVIDIA 官方文档与 CUDA 12+ 新特性介绍） 