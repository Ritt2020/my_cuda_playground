#include <cuda.h>
#include <stdio.h>
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ __forceinline__
void fence_proxy_async_shared(){
    asm volatile("fence.proxy.async.shared::cta;");
}

__global__ void test_tma_copy_to_shared(uint32_t* a, uint32_t* b) {
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar; // mbarrier
    if(threadIdx.x == 0){
        init(&bar, blockDim.x);
        fence_proxy_async_shared();
    }
    __syncthreads();
    // 开同样大小的shared memory
    alignas(16) __shared__ uint32_t shared[8];
    auto tma_load_size = 8 * sizeof(uint32_t);
    // 使用TMA拷贝到shared memory
    if(threadIdx.x == 0){
        cuda::memcpy_async(shared, a, cuda::aligned_size_t<16>(sizeof(shared)), bar);
    }
    barrier::arrival_token token = bar.arrive();
    bar.wait(std::move(token));
    fence_proxy_async_shared();
    // cp_async_mbarrier_arrive(bar);
    // mbarrier_arrive(bar);
    __syncthreads();
    // 把shared中内容拷贝到b数组
    if(threadIdx.x == 0){
        for(uint32_t i = 0; i < 8; i++) {
            b[i] = shared[i];
        }
    }
}

int main() {
    // 生成一个大小为8的数组，每个元素的值为0-7
    uint32_t a[8], b[8];
    for (int i = 0; i < 8; i++) {
        a[i] = i;
        printf("a[%d] = %d ", i, a[i]);
    }
    printf("\n");

    // 分配显存并拷贝数组到global memory
    uint32_t *d_a, *d_b;
    cudaMalloc((void**)&d_a, 8 * sizeof(uint32_t));
    cudaMalloc((void**)&d_b, 8 * sizeof(uint32_t));
    cudaMemcpy(d_a, a, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // 测试 TMA拷贝 到shared memory
    test_tma_copy_to_shared<<<1, 256>>>(d_a, d_b);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(b, d_b, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; i++) {
        printf("b[%d] = %d ", i, b[i]);
    }
    printf("\n");
    return 0;
}