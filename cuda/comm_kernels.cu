// Zero-copy inter-agent message passing on GPU
//
// Features:
//   - Lock-free ring buffer on GPU shared memory
//   - CUDA IPC for cross-process communication
//   - Stream-ordered message serialization
//   - Batched scatter/gather for multi-agent broadcast
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>

namespace cg = cooperative_groups;

namespace neuroswarm {
namespace cuda {

// Lock-Free Ring Buffer Header (lives in GPU global memory)
struct GpuRingBufferHeader {
    volatile uint64_t   write_pos;
    volatile uint64_t   read_pos;
    uint64_t            capacity;        // In bytes
    uint32_t            num_producers;
    uint32_t            num_consumers;
    volatile uint32_t   is_active;       // Kill switch
};

// Message Header (packed for GPU transfer)
struct GpuMessageHeader {
    uint32_t    src_agent;
    uint32_t    dst_agent;       // 0xFFFFFFFF = broadcast
    uint32_t    msg_type;
    uint32_t    payload_size;    // Bytes
    uint64_t    sequence_num;
    uint64_t    timestamp;
};

constexpr uint32_t BROADCAST_AGENT = 0xFFFFFFFF;

// Kernel: Enqueue Message into Ring Buffer
__global__ void enqueue_message_kernel(
    GpuRingBufferHeader* __restrict__ header,
    uint8_t*             __restrict__ buffer,
    const GpuMessageHeader*          msg_header,
    const uint8_t*       __restrict__ payload,
    uint32_t                          payload_size
) {
    if (threadIdx.x != 0) return;
    if (!header->is_active) return;

    uint64_t total_size = sizeof(GpuMessageHeader) + payload_size;

    // Atomic reserve space
    uint64_t write_pos = atomicAdd((unsigned long long*)&header->write_pos, total_size);
    uint64_t ring_pos  = write_pos % header->capacity;

    // Copy header
    uint8_t* dst = buffer + ring_pos;
    const uint8_t* src_hdr = reinterpret_cast<const uint8_t*>(msg_header);
    for (uint32_t i = 0; i < sizeof(GpuMessageHeader); i++) {
        dst[i] = src_hdr[i];
    }

    // Copy payload
    dst += sizeof(GpuMessageHeader);
    for (uint32_t i = 0; i < payload_size; i++) {
        dst[i] = payload[i];
    }

    __threadfence(); // Ensure writes are visible
}

// Kernel: Scatter — Copy data from source to multiple agents
// Copies `count` elements from `src` to each destination in `dst_ptrs`
__global__ void scatter_kernel(
    const float*        __restrict__ src,
    float**             __restrict__ dst_ptrs,    // Array of destination pointers
    int                              num_dsts,
    size_t                           count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < count; i += stride) {
        float val = src[i];
        for (int d = 0; d < num_dsts; d++) {
            dst_ptrs[d][i] = val;
        }
    }
}

// Kernel: Gather — Aggregate results from multiple agents
// Averages `count` elements from each source in `src_ptrs` into `dst`
__global__ void gather_average_kernel(
    const float**       __restrict__ src_ptrs,    // Array of source pointers
    float*              __restrict__ dst,
    int                              num_srcs,
    size_t                           count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    float inv_n = 1.0f / (float)num_srcs;

    for (size_t i = idx; i < count; i += stride) {
        float sum = 0.0f;
        for (int s = 0; s < num_srcs; s++) {
            sum += src_ptrs[s][i];
        }
        dst[i] = sum * inv_n;
    }
}

// Kernel: Gather with Voting (for self-consistency checks)
// Each position votes across agents. Output is the majority value.
__global__ void gather_vote_kernel(
    const int32_t**     __restrict__ src_ptrs,
    int32_t*            __restrict__ dst,
    float*              __restrict__ confidence,  // Agreement ratio per position
    int                              num_srcs,
    size_t                           count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < count; i += stride) {
        // Simple majority vote (works for token IDs)
        // For production, use a histogram approach
        int32_t best_val = src_ptrs[0][i];
        int best_count = 0;

        for (int s = 0; s < num_srcs; s++) {
            int32_t val = src_ptrs[s][i];
            int cnt = 0;
            for (int t = 0; t < num_srcs; t++) {
                if (src_ptrs[t][i] == val) cnt++;
            }
            if (cnt > best_count) {
                best_count = cnt;
                best_val = val;
            }
        }

        dst[i] = best_val;
        if (confidence) {
            confidence[i] = (float)best_count / (float)num_srcs;
        }
    }
}

// Kernel: All-Reduce Sum (for gradient aggregation across agents)
__global__ void allreduce_sum_kernel(
    float**             __restrict__ buffers,     // Each agent's buffer (in-place)
    int                              num_agents,
    size_t                           count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < count; i += stride) {
        float sum = 0.0f;
        for (int a = 0; a < num_agents; a++) {
            sum += buffers[a][i];
        }
        // Write result to all agents
        for (int a = 0; a < num_agents; a++) {
            buffers[a][i] = sum;
        }
    }
}

// Kernel: Initialize Ring Buffer
__global__ void init_ring_buffer_kernel(
    GpuRingBufferHeader* header,
    uint64_t capacity,
    uint32_t num_producers,
    uint32_t num_consumers
) {
    if (threadIdx.x == 0) {
        header->write_pos     = 0;
        header->read_pos      = 0;
        header->capacity      = capacity;
        header->num_producers = num_producers;
        header->num_consumers = num_consumers;
        header->is_active     = 1;
    }
}

// Host Launch Functions

void launch_scatter(const float* src, float** dst_ptrs, int num_dsts,
                     size_t count, cudaStream_t stream) {
    int bs = 256;
    int nb = std::min((int)((count + bs - 1) / bs), 65535);
    scatter_kernel<<<nb, bs, 0, stream>>>(src, dst_ptrs, num_dsts, count);
}

void launch_gather_average(const float** src_ptrs, float* dst, int num_srcs,
                            size_t count, cudaStream_t stream) {
    int bs = 256;
    int nb = std::min((int)((count + bs - 1) / bs), 65535);
    gather_average_kernel<<<nb, bs, 0, stream>>>(src_ptrs, dst, num_srcs, count);
}

void launch_gather_vote(const int32_t** src_ptrs, int32_t* dst, float* confidence,
                         int num_srcs, size_t count, cudaStream_t stream) {
    int bs = 256;
    int nb = std::min((int)((count + bs - 1) / bs), 65535);
    gather_vote_kernel<<<nb, bs, 0, stream>>>(src_ptrs, dst, confidence, num_srcs, count);
}

void launch_allreduce_sum(float** buffers, int num_agents, size_t count,
                           cudaStream_t stream) {
    int bs = 256;
    int nb = std::min((int)((count + bs - 1) / bs), 65535);
    allreduce_sum_kernel<<<nb, bs, 0, stream>>>(buffers, num_agents, count);
}

void launch_init_ring_buffer(GpuRingBufferHeader* header, uint64_t capacity,
                              uint32_t num_producers, uint32_t num_consumers,
                              cudaStream_t stream) {
    init_ring_buffer_kernel<<<1, 1, 0, stream>>>(
        header, capacity, num_producers, num_consumers);
}

} // namespace cuda
} // namespace neuroswarm
