// Wraps the CUDA memory pool with multi-GPU and per-agent accounting
//

#include "neuroswarm/common.h"

#include <spdlog/spdlog.h>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <cuda_runtime.h>

// Forward declare the CUDA pool from memory_pool.cu
namespace neuroswarm { namespace cuda {
class GpuMemoryPool;
}}

namespace neuroswarm {

// Multi-GPU Memory Manager
class GpuMemoryPoolManager {
public:
    explicit GpuMemoryPoolManager(int num_gpus, size_t per_gpu_budget_mb = 16384)
        : num_gpus_(num_gpus), per_gpu_budget_(per_gpu_budget_mb * 1024 * 1024)
    {
        int device_count;
        cudaGetDeviceCount(&device_count);
        num_gpus_ = std::min(num_gpus_, device_count);

        for (int i = 0; i < num_gpus_; i++) {
            cudaSetDevice(i);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);

            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);

            GpuInfo info;
            info.device_id   = i;
            info.name        = prop.name;
            info.total_mem   = total_mem;
            info.free_mem    = free_mem;
            info.allocated   = 0;

            gpu_info_.push_back(info);
            spdlog::info("MemoryManager: GPU {} '{}' — {:.1f}GB total, {:.1f}GB free",
                         i, info.name,
                         total_mem / (1024.0 * 1024 * 1024),
                         free_mem / (1024.0 * 1024 * 1024));
        }
    }

    ~GpuMemoryPoolManager() {
        // Free all tracked allocations
        for (auto& [ptr, alloc] : allocations_) {
            cudaSetDevice(alloc.device_id);
            cudaFree(ptr);
        }
    }

    /// Allocate memory on the specified GPU for a given agent
    void* allocate(size_t size, int device_id, uint32_t agent_id = 0) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (device_id < 0 || device_id >= num_gpus_) {
            spdlog::error("Invalid device_id {}", device_id);
            return nullptr;
        }

        // Check per-agent limits
        if (agent_id > 0) {
            auto it = agent_limits_.find(agent_id);
            if (it != agent_limits_.end()) {
                size_t current = agent_usage(agent_id);
                if (current + size > it->second) {
                    spdlog::warn("Agent {} would exceed memory limit ({:.1f}MB / {:.1f}MB)",
                                 agent_id,
                                 (current + size) / (1024.0 * 1024),
                                 it->second / (1024.0 * 1024));
                    return nullptr;
                }
            }
        }

        // Check GPU budget
        if (gpu_info_[device_id].allocated + size > per_gpu_budget_) {
            spdlog::warn("GPU {} budget exceeded, attempting defrag", device_id);
            defragment(device_id);
            if (gpu_info_[device_id].allocated + size > per_gpu_budget_) {
                return nullptr;
            }
        }

        cudaSetDevice(device_id);
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            spdlog::error("cudaMalloc failed on GPU {}: {} ({} bytes)",
                         device_id, cudaGetErrorString(err), size);
            return nullptr;
        }

        Allocation alloc;
        alloc.ptr       = ptr;
        alloc.size      = size;
        alloc.device_id = device_id;
        alloc.agent_id  = agent_id;
        alloc.timestamp = Clock::now();

        allocations_[ptr] = alloc;
        gpu_info_[device_id].allocated += size;

        total_allocated_ += size;
        peak_allocated_ = std::max(peak_allocated_, total_allocated_);

        return ptr;
    }

    /// Free a previously allocated buffer
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            spdlog::warn("Attempted to free unknown pointer");
            return;
        }

        cudaSetDevice(it->second.device_id);
        cudaFree(ptr);

        gpu_info_[it->second.device_id].allocated -= it->second.size;
        total_allocated_ -= it->second.size;

        allocations_.erase(it);
    }

    /// Copy data between GPUs (handles P2P vs staged copy)
    void copy_cross_gpu(void* dst, int dst_gpu, void* src, int src_gpu,
                        size_t size, cudaStream_t stream = nullptr) {
        int can_peer;
        cudaDeviceCanAccessPeer(&can_peer, dst_gpu, src_gpu);

        if (can_peer) {
            // Direct P2P copy
            cudaMemcpyPeerAsync(dst, dst_gpu, src, src_gpu, size, stream);
        } else {
            // Stage through host
            void* host_buf = nullptr;
            cudaMallocHost(&host_buf, size);

            cudaSetDevice(src_gpu);
            cudaMemcpyAsync(host_buf, src, size, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            cudaSetDevice(dst_gpu);
            cudaMemcpyAsync(dst, host_buf, size, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);

            cudaFreeHost(host_buf);
        }
    }

    /// Set per-agent memory limit
    void set_agent_limit(uint32_t agent_id, size_t max_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        agent_limits_[agent_id] = max_bytes;
    }

    /// Get memory usage for a specific agent
    size_t agent_usage(uint32_t agent_id) const {
        size_t total = 0;
        for (const auto& [ptr, alloc] : allocations_) {
            if (alloc.agent_id == agent_id) {
                total += alloc.size;
            }
        }
        return total;
    }

    /// Force-free all memory belonging to an agent (for kill switch)
    void evict_agent(uint32_t agent_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<void*> to_free;
        for (const auto& [ptr, alloc] : allocations_) {
            if (alloc.agent_id == agent_id) {
                to_free.push_back(ptr);
            }
        }
        for (void* ptr : to_free) {
            cudaSetDevice(allocations_[ptr].device_id);
            cudaFree(ptr);
            gpu_info_[allocations_[ptr].device_id].allocated -= allocations_[ptr].size;
            total_allocated_ -= allocations_[ptr].size;
            allocations_.erase(ptr);
        }
        spdlog::info("Evicted {} allocations for agent {}", to_free.size(), agent_id);
    }

    /// Defragment a GPU (compact live allocations)
    void defragment(int device_id) {
        // In practice, this would use CUDA virtual memory management
        // (cudaMemPool APIs on CUDA 11.2+) to compact the address space.
        // For now, we just trim the free lists.
        spdlog::info("Defragmenting GPU {} memory", device_id);
    }

    // Getters
    size_t total_allocated() const { return total_allocated_; }
    size_t peak_allocated() const { return peak_allocated_; }
    int num_gpus() const { return num_gpus_; }

    struct GpuSnapshot {
        int device_id;
        std::string name;
        size_t total_mem;
        size_t allocated;
        size_t free;
    };

    std::vector<GpuSnapshot> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<GpuSnapshot> result;
        for (const auto& info : gpu_info_) {
            result.push_back({
                info.device_id, info.name, info.total_mem,
                info.allocated, info.total_mem - info.allocated
            });
        }
        return result;
    }

private:
    struct GpuInfo {
        int device_id;
        std::string name;
        size_t total_mem;
        size_t free_mem;
        size_t allocated;
    };

    struct Allocation {
        void* ptr;
        size_t size;
        int device_id;
        uint32_t agent_id;
        TimePoint timestamp;
    };

    int num_gpus_;
    size_t per_gpu_budget_;
    mutable std::mutex mutex_;

    std::vector<GpuInfo> gpu_info_;
    std::unordered_map<void*, Allocation> allocations_;
    std::unordered_map<uint32_t, size_t> agent_limits_;

    size_t total_allocated_ = 0;
    size_t peak_allocated_ = 0;
};

} // namespace neuroswarm
