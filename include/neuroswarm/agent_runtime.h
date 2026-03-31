// Execution engine for individual agents with sandboxing and lifecycle management
//
#pragma once

#include "neuroswarm/common.h"
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <functional>
#include <queue>
#include <unordered_map>

namespace neuroswarm {

class InferenceEngine;
class CommunicationBus;

// Tool Interface — Agents can call tools
struct ToolCall {
    std::string name;
    std::string arguments_json;
    std::string result_json;
    bool        success;
    double      latency_ms;
};

using ToolFunction = std::function<std::string(const std::string& args_json)>;

// Agent Thought/Action (for CoT reasoning)
struct ThoughtAction {
    enum Type { THOUGHT, ACTION, OBSERVATION, FINAL_ANSWER };
    Type        type;
    std::string content;
    TimePoint   timestamp;
};

// Agent Runtime — Manages a single agent's lifecycle
class AgentRuntime {
public:
    AgentRuntime(uint32_t id, const AgentConfig& config,
                 InferenceEngine* engine, CommunicationBus* bus);
    ~AgentRuntime();

    // Lifecycle
    void start();
    void stop();
    void kill();

    // Task execution
    std::string execute(const std::string& prompt, int max_steps = 10);
    void execute_async(const std::string& prompt,
                       std::function<void(const std::string&)> callback);

    // Tool management
    void register_tool(const std::string& name, const std::string& description,
                       ToolFunction func);
    void unregister_tool(const std::string& name);

    // State
    uint32_t id() const { return id_; }
    AgentState state() const { return state_.load(); }
    const AgentConfig& config() const { return config_; }
    const std::vector<ThoughtAction>& trace() const { return trace_; }
    size_t gpu_memory_used() const { return gpu_memory_used_; }

    // Safety
    void set_resource_limits(const ResourceLimits& limits);
    bool is_within_limits() const;

    // Metrics
    uint64_t total_tokens_generated() const { return total_tokens_; }
    double average_latency_ms() const;

private:
    std::string run_inference(const std::string& prompt);
    std::string parse_action(const std::string& response);
    std::string execute_tool(const std::string& tool_name, const std::string& args);
    bool check_hallucination(const std::string& response,
                             const std::vector<std::string>& context);
    void enforce_resource_limits();
    void append_trace(ThoughtAction::Type type, const std::string& content);

    uint32_t id_;
    AgentConfig config_;
    std::atomic<AgentState> state_{AgentState::IDLE};

    InferenceEngine* engine_;   // Not owned
    CommunicationBus* bus_;     // Not owned

    // Tools
    std::mutex tools_mutex_;
    std::unordered_map<std::string, ToolFunction> tools_;
    std::unordered_map<std::string, std::string> tool_descriptions_;

    // Execution trace
    std::mutex trace_mutex_;
    std::vector<ThoughtAction> trace_;

    // Resource tracking
    ResourceLimits resource_limits_;
    size_t gpu_memory_used_ = 0;
    uint64_t total_tokens_ = 0;
    uint64_t total_inferences_ = 0;
    double total_latency_ms_ = 0;
    TimePoint execution_start_;

    // Async execution
    std::thread exec_thread_;
    std::atomic<bool> kill_requested_{false};

    // CUDA resources
    cudaStream_t stream_ = nullptr;
    int gpu_id_ = 0;
};

} // namespace neuroswarm
