// GPU-native inter-agent communication with zero-copy transfers
//
#pragma once

#include "neuroswarm/common.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>
#include <unordered_map>
#include <queue>
#include <atomic>

namespace neuroswarm {

// Subscription — for pub/sub message routing
using MessageHandler = std::function<void(const Message&)>;

struct Subscription {
    uint32_t    subscriber_id;
    MessageType msg_type;
    MessageHandler handler;
};

// Communication Bus — Zero-copy GPU IPC
class CommunicationBus {
public:
    explicit CommunicationBus(int num_gpus = 1, size_t ring_buffer_size = 64 * 1024 * 1024);
    ~CommunicationBus();

    // Message passing
    void send(const Message& msg);
    void send_gpu(uint32_t src, uint32_t dst, void* gpu_data,
                  size_t size, cudaStream_t stream);
    void broadcast(uint32_t src, const Message& msg);
    bool poll(uint32_t agent_id, Message& out_msg);

    // Pub/Sub
    uint64_t subscribe(uint32_t agent_id, MessageType type, MessageHandler handler);
    void unsubscribe(uint64_t subscription_id);

    // Scatter/Gather
    void scatter(void* src, const std::vector<void*>& dst_ptrs,
                 size_t size, cudaStream_t stream);
    void gather(const std::vector<void*>& src_ptrs, void* dst,
                size_t size, cudaStream_t stream);

    // Zero-copy shared memory
    void* create_shared_region(const std::string& name, size_t size, int gpu_id);
    void* open_shared_region(const std::string& name);
    void close_shared_region(const std::string& name);

    // Control
    void flush();
    void drain(uint32_t agent_id);
    void kill_switch();

    // Metrics
    uint64_t total_messages() const { return total_messages_.load(); }
    uint64_t total_bytes() const { return total_bytes_.load(); }

private:
    void dispatch_message(const Message& msg);

    int num_gpus_;
    size_t ring_buffer_size_;

    // Per-agent message queues
    mutable std::mutex queues_mutex_;
    std::unordered_map<uint32_t, std::queue<Message>> agent_queues_;

    // Subscriptions
    mutable std::mutex subs_mutex_;
    std::vector<Subscription> subscriptions_;
    uint64_t next_sub_id_ = 1;

    // Shared memory regions
    mutable std::mutex regions_mutex_;
    std::unordered_map<std::string, void*> shared_regions_;

    // GPU ring buffers (per GPU)
    std::vector<void*> gpu_ring_buffers_;
    std::vector<void*> gpu_ring_headers_;

    // CUDA streams for async comm
    std::vector<cudaStream_t> comm_streams_;

    // Stats
    std::atomic<uint64_t> total_messages_{0};
    std::atomic<uint64_t> total_bytes_{0};
    std::atomic<bool> active_{true};
};

} // namespace neuroswarm
