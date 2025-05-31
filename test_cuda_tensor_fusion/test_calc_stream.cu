#include <cuda.h>
#include <stdio.h>
#include <chrono>

__device__ __forceinline__
void convert_fp32_to_tf32(float* data, int count) {
    uint32_t* uint_data = reinterpret_cast<uint32_t*>(data);
    for(int i = 0; i < count; i++) {
        asm volatile(
            "cvt.rna.tf32.f32 %0, %0;\n"
            :"+r"(uint_data[i])
        );
    }
}

__device__ __forceinline__
void tf32_m16n8k8(float* MatA, float* MatB, float* MatC) {
    uint32_t const* A = reinterpret_cast<uint32_t const*>(MatA);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(MatB);
    float* C = reinterpret_cast<float*>(MatC);

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%0, %1, %2, %3};\n"
        :"+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        :"r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
         "r"(B[0]), "r"(B[1])
    );
}

__global__ void cuda_core_kernel() {
    int tid = threadIdx.x;
    float value = (float)(tid + 1);
    
    for(int i = 0; i < 7000000; i++) {
        value = value / 2.0f + 1.0f;
    }
    
    // Prevent optimization
    if(value < 0) printf("%f\n", value);
}

__global__ void tensor_core_kernel() {
    int lane_id = threadIdx.x % 32;
    
    // Initialize matrices with 1.0
    float fragA[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float fragB[2] = {1.0f, 1.0f};
    float fragC[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    
    // Only threads participating in tensor core operations
    if(lane_id < 32) {
        // Convert to TF32 format once at the beginning
        convert_fp32_to_tf32(fragA, 4);
        convert_fp32_to_tf32(fragB, 2);
        
        for(int i = 0; i < 1000000; i++) {
            tf32_m16n8k8(fragA, fragB, fragC);
        }
    }
    
    // Prevent optimization
    if(fragC[0] < 0) printf("%f\n", fragC[0]);
}

int main() {
    cudaSetDevice(0);
    const int num_blocks = 132;
    const int threads_per_block = 32 * 16; // 16 warps * 32 threads
    
    // Create streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    printf("Testing CUDA and Tensor Core parallel execution with streams...\n");
    printf("Blocks: %d, Threads per block: %d\n", num_blocks, threads_per_block);
    
    // Warmup phase
    printf("\n=== Warmup Phase ===\n");
    for(int i = 0; i < 5; i++) {
        cuda_core_kernel<<<num_blocks, threads_per_block, 0, stream1>>>();
        tensor_core_kernel<<<num_blocks, threads_per_block, 0, stream2>>>();
        cudaDeviceSynchronize();
    }
    printf("Warmup completed.\n");
    
    // Test 1: Only CUDA cores
    printf("\n=== CUDA Core Only ===\n");
    cudaEventRecord(start);
    cuda_core_kernel<<<num_blocks, threads_per_block, 0, stream1>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Core Only Time: %.3f ms\n", milliseconds);
    
    // Test 2: Only Tensor cores
    printf("\n=== Tensor Core Only ===\n");
    cudaEventRecord(start);
    tensor_core_kernel<<<num_blocks, threads_per_block, 0, stream1>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tensor Core Only Time: %.3f ms\n", milliseconds);
    
    // Test 3: Both CUDA and Tensor cores in parallel (different streams)
    printf("\n=== Both Cores Parallel (Different Streams) ===\n");
    cudaEventRecord(start);
    cuda_core_kernel<<<num_blocks, threads_per_block, 0, stream1>>>();
    tensor_core_kernel<<<num_blocks, threads_per_block, 0, stream2>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Both Cores Parallel Time: %.3f ms\n", milliseconds);
    
    // Test 4: Two CUDA core kernels in parallel
    printf("\n=== Two CUDA Core Kernels Parallel ===\n");
    cudaEventRecord(start);
    cuda_core_kernel<<<num_blocks, threads_per_block, 0, stream1>>>();
    cuda_core_kernel<<<num_blocks, threads_per_block, 0, stream2>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Two CUDA Cores Parallel Time: %.3f ms\n", milliseconds);
    
    // Test 5: Two Tensor core kernels in parallel
    printf("\n=== Two Tensor Core Kernels Parallel ===\n");
    cudaEventRecord(start);
    tensor_core_kernel<<<num_blocks, threads_per_block, 0, stream1>>>();
    tensor_core_kernel<<<num_blocks, threads_per_block, 0, stream2>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Two Tensor Cores Parallel Time: %.3f ms\n", milliseconds);
    
    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\nTest completed.\n");
    return 0;
}
