#include <cuda.h>
#include <stdio.h>

__device__ __forceinline__
void async_store_from_shared(const uint32_t* a, uint32_t b, uint32_t len) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(a), "r"(b), "r"(len));
}

__device__ __forceinline__
void fence_proxy_async_shared(){
    asm volatile("fence.proxy.async.shared::cta;");
}

__device__ __forceinline__
void wait_group_bulk() {
    asm volatile(
        "cp.async.bulk.commit_group;\n"
        "cp.async.bulk.wait_group 0;\n"
        ::
    );
}

__global__ void test_async_store_from_shared(uint32_t* a) {
    __shared__ uint32_t shared[8];
    for(int i = 0; i < 8; i++) {
        shared[i] = i;
    }
    uint32_t sPtr = __cvta_generic_to_shared(shared);
    if(threadIdx.x == 0){
        async_store_from_shared(a, sPtr, 8 * sizeof(uint32_t));
    }
    wait_group_bulk();
}

int main() {
    uint32_t *d_a;
    cudaMalloc((void**)&d_a, 8 * sizeof(uint32_t));
    test_async_store_from_shared<<<1, 256>>>(d_a);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    uint32_t a[8];
    cudaMemcpy(a, d_a, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 8; i++) {
        printf("a[%d] = %d\n", i, a[i]);
    }
    cudaFree(d_a);
    return 0;
}