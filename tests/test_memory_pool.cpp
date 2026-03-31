#include <gtest/gtest.h>
#include <cuda_runtime.h>

// The GpuMemoryPool class is defined in memory_pool.cu
// In a real build, we'd link against neuroswarm_cuda

TEST(MemoryPoolTest, SkipWithoutGPU) {
    int count; cudaGetDeviceCount(&count);
    if (count == 0) GTEST_SKIP() << "No GPU";
}

TEST(MemoryPoolTest, BasicAllocAndFree) {
    int count; cudaGetDeviceCount(&count);
    if (count == 0) GTEST_SKIP();

    void* ptr = nullptr;
    auto err = cudaMalloc(&ptr, 1024 * 1024);
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_NE(ptr, nullptr);
    cudaFree(ptr);
}

TEST(MemoryPoolTest, MultipleAllocs) {
    int count; cudaGetDeviceCount(&count);
    if (count == 0) GTEST_SKIP();

    std::vector<void*> ptrs;
    for (int i = 0; i < 100; i++) {
        void* ptr;
        ASSERT_EQ(cudaMalloc(&ptr, 4096), cudaSuccess);
        ptrs.push_back(ptr);
    }
    for (void* p : ptrs) cudaFree(p);
}

TEST(MemoryPoolTest, LargeAllocation) {
    int count; cudaGetDeviceCount(&count);
    if (count == 0) GTEST_SKIP();

    void* ptr;
    auto err = cudaMalloc(&ptr, 256 * 1024 * 1024); // 256MB
    ASSERT_EQ(err, cudaSuccess);
    cudaFree(ptr);
}
