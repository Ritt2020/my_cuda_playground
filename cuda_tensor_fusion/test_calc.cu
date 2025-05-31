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

__device__ void cuda_core_compute() {
    int tid = threadIdx.x;
    float value = (float)(tid + 1);
    
    for(int i = 0; i < 7000000; i++) {
        value = value / 2.0f + 1.0f;
    }
    
    // Prevent optimization
    if(value < 0) printf("%f\n", value);
}

__device__ void tensor_core_compute() {
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

__global__ void fusion_kernel(bool enable_cuda, bool enable_tensor) {
    int warp_id = threadIdx.x / 32;
    int total_warps = blockDim.x / 32;
    int half_warps = total_warps / 2;
    
    if(warp_id < half_warps && enable_cuda) {
        // First half warps do CUDA core computation
        cuda_core_compute();
    } else if(warp_id >= half_warps && enable_tensor) {
        // Second half warps do Tensor core computation
        tensor_core_compute();
    }
}

__global__ void all_cuda_kernel() {
    // All threads do CUDA core computation
    cuda_core_compute();
}

__global__ void all_tensor_kernel() {
    // All threads do Tensor core computation
    tensor_core_compute();
}

int main() {
    const int num_blocks = 132;
    const int threads_per_block = 32 * 32; // 16 warps * 32 threads
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    printf("Testing CUDA and Tensor Core fusion...\n");
    printf("Blocks: %d, Threads per block: %d\n", num_blocks, threads_per_block);
    
    // Print kernel function attributes
    printf("\n=== Kernel Function Attributes ===\n");
    cudaFuncAttributes attr;
    cudaError_t err = cudaFuncGetAttributes(&attr, fusion_kernel);
    if (err == cudaSuccess) {
        printf("Max threads per block: %d\n", attr.maxThreadsPerBlock);
        printf("Shared memory size per block: %zu bytes\n", attr.sharedSizeBytes);
        printf("Constant memory size: %zu bytes\n", attr.constSizeBytes);
        printf("Local memory size per thread: %zu bytes\n", attr.localSizeBytes);
        printf("Number of registers per thread: %d\n", attr.numRegs);
        printf("PTX version: %d\n", attr.ptxVersion);
        printf("Binary version: %d\n", attr.binaryVersion);
        printf("Cache mode CA: %d\n", attr.cacheModeCA);
        printf("Max dynamic shared memory per block: %d bytes\n", attr.maxDynamicSharedSizeBytes);
        printf("Preferred shared memory carveout: %d\n", attr.preferredShmemCarveout);
    } else {
        printf("Failed to get kernel attributes: %s\n", cudaGetErrorString(err));
    }
    
    // Warmup phase
    printf("\n=== Warmup Phase ===\n");
    const int warmup_blocks = num_blocks; // Use 10% of blocks for warmup
    printf("Running warmup with %d blocks...\n", warmup_blocks);
    
    // Warmup with all three configurations
    for(int i = 0; i < 3; i++) {
        fusion_kernel<<<warmup_blocks, threads_per_block>>>(true, false);   // CUDA only
        fusion_kernel<<<warmup_blocks, threads_per_block>>>(false, true);   // Tensor only
        fusion_kernel<<<warmup_blocks, threads_per_block>>>(true, true);    // Both
        all_cuda_kernel<<<warmup_blocks, threads_per_block>>>();            // All CUDA
        all_tensor_kernel<<<warmup_blocks, threads_per_block>>>();          // All Tensor
        cudaDeviceSynchronize();
    }
    printf("Warmup completed.\n");
    
    // Test 1: Only CUDA cores (half warps)
    printf("\n=== CUDA Core Only (Half Warps) ===\n");
    cudaEventRecord(start);
    fusion_kernel<<<num_blocks, threads_per_block>>>(true, false);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Core Only (Half Warps) Time: %.3f ms\n", milliseconds);
    
    // Test 2: Only Tensor cores (half warps)
    printf("\n=== Tensor Core Only (Half Warps) ===\n");
    cudaEventRecord(start);
    fusion_kernel<<<num_blocks, threads_per_block>>>(false, true);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tensor Core Only (Half Warps) Time: %.3f ms\n", milliseconds);
    
    // Test 3: Both CUDA and Tensor cores
    printf("\n=== Both CUDA and Tensor Cores ===\n");
    cudaEventRecord(start);
    fusion_kernel<<<num_blocks, threads_per_block>>>(true, true);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Both Cores Time: %.3f ms\n", milliseconds);
    
    // Test 4: All threads do CUDA core computation
    printf("\n=== All Threads CUDA Core ===\n");
    cudaEventRecord(start);
    all_cuda_kernel<<<num_blocks, threads_per_block>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("All Threads CUDA Core Time: %.3f ms\n", milliseconds);
    
    // Test 5: All threads do Tensor core computation
    printf("\n=== All Threads Tensor Core ===\n");
    cudaEventRecord(start);
    all_tensor_kernel<<<num_blocks, threads_per_block>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("All Threads Tensor Core Time: %.3f ms\n", milliseconds);
    
    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
