// NeuroSwarm - GPU-Native Multi-Agent Reasoning System
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <optional>
#include <variant>
#include <chrono>
#include <atomic>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUDA Error Checking Macros
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            throw std::runtime_error(                                          \
                std::string("CUDA error: ") + cudaGetErrorString(err));        \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA kernel error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            throw std::runtime_error(                                          \
                std::string("CUDA kernel error: ") + cudaGetErrorString(err)); \
        }                                                                      \
    } while (0)

namespace neuroswarm {

// Type Aliases
using TimePoint = std::chrono::steady_clock::time_point;
using Duration  = std::chrono::steady_clock::duration;
using Clock     = std::chrono::steady_clock;

// Enums
enum class AgentRole : uint8_t {
    PLANNER   = 0,
    EXECUTOR  = 1,
    VALIDATOR = 2,
    MONITOR   = 3,
};

enum class AgentState : uint8_t {
    IDLE       = 0,
    RUNNING    = 1,
    WAITING    = 2,
    COMPLETED  = 3,
    FAILED     = 4,
    KILLED     = 5,
};

enum class QuantMode : uint8_t {
    FP32  = 0,
    FP16  = 1,
    INT8  = 2,
    INT4  = 3,
};

enum class MessageType : uint8_t {
    TASK_ASSIGN    = 0,
    TASK_RESULT    = 1,
    STATE_UPDATE   = 2,
    HEARTBEAT      = 3,
    KILL_SIGNAL    = 4,
    ROLLBACK       = 5,
    CHECKPOINT     = 6,
};

enum class CircuitState : uint8_t {
    CLOSED    = 0, // Operational
    OPEN      = 1, // Tripped — blocking calls
    HALF_OPEN = 2, // Testing recovery
};

// Core Data Structures
struct AgentConfig {
    std::string     name;
    AgentRole       role;
    QuantMode       quant_mode      = QuantMode::FP16;
    int             gpu_id          = 0;
    size_t          max_memory_mb   = 8192;
    int             max_tokens      = 4096;
    float           temperature     = 0.7f;
    float           top_p           = 0.9f;
    int             top_k           = 50;
    size_t          timeout_ms      = 30000;
    bool            enable_sandbox  = true;
};

struct Message {
    MessageType     type;
    uint32_t        src_agent_id;
    uint32_t        dst_agent_id;
    uint64_t        sequence_num;
    uint64_t        timestamp_ns;
    size_t          payload_size;
    void*           gpu_payload_ptr;  // Device pointer for zero-copy
    std::vector<uint8_t> cpu_payload; // Fallback CPU payload
};

struct InferenceRequest {
    uint32_t        agent_id;
    std::vector<int32_t> input_tokens;
    int             max_new_tokens  = 512;
    float           temperature     = 0.7f;
    float           top_p           = 0.9f;
    int             top_k           = 50;
    QuantMode       quant_mode      = QuantMode::FP16;
    cudaStream_t    stream          = nullptr;
};

struct InferenceResult {
    uint32_t        agent_id;
    std::vector<int32_t> output_tokens;
    float           latency_ms;
    size_t          tokens_per_second;
    bool            truncated;
};

struct PerformanceMetrics {
    double          total_throughput_tps;      // tokens/sec across all agents
    double          avg_latency_ms;
    double          p50_latency_ms;
    double          p99_latency_ms;
    size_t          gpu_memory_used_mb;
    size_t          gpu_memory_total_mb;
    float           gpu_utilization_pct;
    uint64_t        total_inferences;
    uint64_t        total_messages;
    uint64_t        failed_inferences;
    double          uptime_seconds;
};

struct ResourceLimits {
    size_t          max_gpu_memory_bytes = 0;     // 0 = unlimited
    size_t          max_cpu_memory_bytes = 0;
    double          max_gpu_utilization  = 1.0;
    uint32_t        max_concurrent_ops   = 64;
    uint32_t        max_tokens_per_min   = 100000;
    Duration        max_execution_time   = std::chrono::seconds(300);
};

// GPU Tensor Descriptor
struct TensorDesc {
    void*           data        = nullptr;
    size_t          num_elements = 0;
    std::vector<int64_t> shape;
    QuantMode       dtype       = QuantMode::FP16;
    int             device_id   = 0;

    size_t byte_size() const {
        size_t elem_size = 0;
        switch (dtype) {
            case QuantMode::FP32: elem_size = 4; break;
            case QuantMode::FP16: elem_size = 2; break;
            case QuantMode::INT8: elem_size = 1; break;
            case QuantMode::INT4: elem_size = 1; break; // packed
        }
        return num_elements * elem_size;
    }
};

// Utility Functions
inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()
    ).count();
}

inline double elapsed_ms(TimePoint start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

inline const char* agent_role_str(AgentRole role) {
    switch (role) {
        case AgentRole::PLANNER:   return "Planner";
        case AgentRole::EXECUTOR:  return "Executor";
        case AgentRole::VALIDATOR: return "Validator";
        case AgentRole::MONITOR:   return "Monitor";
        default:                   return "Unknown";
    }
}

inline const char* agent_state_str(AgentState state) {
    switch (state) {
        case AgentState::IDLE:      return "Idle";
        case AgentState::RUNNING:   return "Running";
        case AgentState::WAITING:   return "Waiting";
        case AgentState::COMPLETED: return "Completed";
        case AgentState::FAILED:    return "Failed";
        case AgentState::KILLED:    return "Killed";
        default:                    return "Unknown";
    }
}

} // namespace neuroswarm
