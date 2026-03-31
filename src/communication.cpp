// GPU-native IPC, pub/sub routing, shared memory regions
//

#include "neuroswarm/communication.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>

namespace neuroswarm {

// Construction / Destruction

CommunicationBus::CommunicationBus(int num_gpus, size_t ring_buffer_size)
    : num_gpus_(num_gpus), ring_buffer_size_(ring_buffer_size)
{
    // Create CUDA streams for async communication
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        comm_streams_.push_back(stream);
    }

    // Allocate GPU ring buffers for each GPU
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);

        void* buffer = nullptr;
        void* header = nullptr;
        cudaMalloc(&buffer, ring_buffer_size);
        cudaMalloc(&header, sizeof(uint64_t) * 8); // Ring buffer header
        cudaMemset(buffer, 0, ring_buffer_size);
        cudaMemset(header, 0, sizeof(uint64_t) * 8);

        gpu_ring_buffers_.push_back(buffer);
        gpu_ring_headers_.push_back(header);
    }

    // Enable P2P access between GPUs if available
    for (int i = 0; i < num_gpus; i++) {
        for (int j = 0; j < num_gpus; j++) {
            if (i != j) {
                int can_access;
                cudaDeviceCanAccessPeer(&can_access, i, j);
                if (can_access) {
                    cudaSetDevice(i);
                    cudaDeviceEnablePeerAccess(j, 0);
                    spdlog::info("P2P access enabled: GPU {} → GPU {}", i, j);
                }
            }
        }
    }

    spdlog::info("CommunicationBus initialized: {} GPUs, {}MB ring buffer per GPU",
                 num_gpus, ring_buffer_size / (1024 * 1024));
}

CommunicationBus::~CommunicationBus() {
    // Free GPU resources
    for (int i = 0; i < num_gpus_; i++) {
        cudaSetDevice(i);
        if (gpu_ring_buffers_[i]) cudaFree(gpu_ring_buffers_[i]);
        if (gpu_ring_headers_[i]) cudaFree(gpu_ring_headers_[i]);
        if (comm_streams_[i]) cudaStreamDestroy(comm_streams_[i]);
    }

    // Free shared regions
    for (auto& [name, ptr] : shared_regions_) {
        cudaFree(ptr);
    }
}

// Message Passing (CPU-side queues)

void CommunicationBus::send(const Message& msg) {
    if (!active_.load()) return;

    total_messages_.fetch_add(1, std::memory_order_relaxed);
    total_bytes_.fetch_add(msg.payload_size + sizeof(Message),
                           std::memory_order_relaxed);

    // Route to destination agent's queue
    {
        std::lock_guard<std::mutex> lock(queues_mutex_);
        agent_queues_[msg.dst_agent_id].push(msg);
    }

    // Notify subscribers
    dispatch_message(msg);
}

void CommunicationBus::send_gpu(uint32_t src, uint32_t dst, void* gpu_data,
                                  size_t size, cudaStream_t stream) {
    if (!active_.load()) return;

    // Zero-copy: just pass the device pointer directly
    Message msg;
    msg.type = MessageType::STATE_UPDATE;
    msg.src_agent_id = src;
    msg.dst_agent_id = dst;
    msg.payload_size = size;
    msg.gpu_payload_ptr = gpu_data;
    msg.timestamp_ns = now_ns();

    total_messages_.fetch_add(1);
    total_bytes_.fetch_add(size);

    std::lock_guard<std::mutex> lock(queues_mutex_);
    agent_queues_[dst].push(msg);
}

void CommunicationBus::broadcast(uint32_t src, const Message& msg) {
    if (!active_.load()) return;

    std::lock_guard<std::mutex> lock(queues_mutex_);
    for (auto& [agent_id, queue] : agent_queues_) {
        if (agent_id != src) {
            Message copy = msg;
            copy.dst_agent_id = agent_id;
            queue.push(copy);
            total_messages_.fetch_add(1);
        }
    }
}

bool CommunicationBus::poll(uint32_t agent_id, Message& out_msg) {
    std::lock_guard<std::mutex> lock(queues_mutex_);
    auto it = agent_queues_.find(agent_id);
    if (it == agent_queues_.end() || it->second.empty()) {
        return false;
    }

    out_msg = it->second.front();
    it->second.pop();
    return true;
}

// Pub/Sub

uint64_t CommunicationBus::subscribe(uint32_t agent_id, MessageType type,
                                      MessageHandler handler) {
    std::lock_guard<std::mutex> lock(subs_mutex_);
    uint64_t sub_id = next_sub_id_++;
    subscriptions_.push_back({agent_id, type, handler});

    // Ensure agent has a queue
    {
        std::lock_guard<std::mutex> qlock(queues_mutex_);
        agent_queues_[agent_id]; // Create if doesn't exist
    }

    spdlog::debug("Agent {} subscribed to message type {} (sub_id={})",
                  agent_id, static_cast<int>(type), sub_id);
    return sub_id;
}

void CommunicationBus::unsubscribe(uint64_t subscription_id) {
    std::lock_guard<std::mutex> lock(subs_mutex_);
    // In production, subscriptions would be keyed by ID
    spdlog::debug("Unsubscribed {}", subscription_id);
}

void CommunicationBus::dispatch_message(const Message& msg) {
    std::lock_guard<std::mutex> lock(subs_mutex_);
    for (const auto& sub : subscriptions_) {
        if (sub.msg_type == msg.type &&
            (sub.subscriber_id == msg.dst_agent_id ||
             msg.dst_agent_id == 0xFFFFFFFF)) // Broadcast
        {
            try {
                sub.handler(msg);
            } catch (const std::exception& e) {
                spdlog::error("Subscription handler error: {}", e.what());
            }
        }
    }
}

// Scatter / Gather (GPU-side)

void CommunicationBus::scatter(void* src, const std::vector<void*>& dst_ptrs,
                                size_t size, cudaStream_t stream) {
    for (auto* dst : dst_ptrs) {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    }
    total_bytes_.fetch_add(size * dst_ptrs.size());
}

void CommunicationBus::gather(const std::vector<void*>& src_ptrs, void* dst,
                               size_t size, cudaStream_t stream) {
    // Copy each source — in production, use the fused gather kernel
    for (size_t i = 0; i < src_ptrs.size(); i++) {
        cudaMemcpyAsync(static_cast<char*>(dst) + i * size,
                        src_ptrs[i], size,
                        cudaMemcpyDeviceToDevice, stream);
    }
    total_bytes_.fetch_add(size * src_ptrs.size());
}

// Shared Memory Regions

void* CommunicationBus::create_shared_region(const std::string& name,
                                               size_t size, int gpu_id) {
    std::lock_guard<std::mutex> lock(regions_mutex_);

    if (shared_regions_.count(name)) {
        spdlog::warn("Shared region '{}' already exists", name);
        return shared_regions_[name];
    }

    cudaSetDevice(gpu_id);
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        spdlog::error("Failed to create shared region '{}': {}",
                     name, cudaGetErrorString(err));
        return nullptr;
    }

    cudaMemset(ptr, 0, size);
    shared_regions_[name] = ptr;

    spdlog::info("Created shared region '{}': {}MB on GPU {}",
                 name, size / (1024 * 1024), gpu_id);
    return ptr;
}

void* CommunicationBus::open_shared_region(const std::string& name) {
    std::lock_guard<std::mutex> lock(regions_mutex_);
    auto it = shared_regions_.find(name);
    return (it != shared_regions_.end()) ? it->second : nullptr;
}

void CommunicationBus::close_shared_region(const std::string& name) {
    std::lock_guard<std::mutex> lock(regions_mutex_);
    auto it = shared_regions_.find(name);
    if (it != shared_regions_.end()) {
        cudaFree(it->second);
        shared_regions_.erase(it);
        spdlog::info("Closed shared region '{}'", name);
    }
}

// Control

void CommunicationBus::flush() {
    for (auto& stream : comm_streams_) {
        cudaStreamSynchronize(stream);
    }
}

void CommunicationBus::drain(uint32_t agent_id) {
    std::lock_guard<std::mutex> lock(queues_mutex_);
    auto it = agent_queues_.find(agent_id);
    if (it != agent_queues_.end()) {
        while (!it->second.empty()) it->second.pop();
    }
}

void CommunicationBus::kill_switch() {
    spdlog::warn("Communication bus KILL SWITCH activated");
    active_.store(false);

    // Clear all queues
    std::lock_guard<std::mutex> lock(queues_mutex_);
    for (auto& [id, queue] : agent_queues_) {
        while (!queue.empty()) queue.pop();
    }
}

} // namespace neuroswarm
