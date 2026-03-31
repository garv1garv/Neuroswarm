// Central coordinator for multi-agent GPU-native reasoning
//
#pragma once

#include "neuroswarm/common.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>

namespace neuroswarm {

// Forward declarations
class AgentRuntime;
class CommunicationBus;
class InferenceEngine;
class GpuMemoryPoolManager;
class CircuitBreaker;
class MetricsCollector;

// Task Definition
struct Task {
    uint64_t            id;
    std::string         description;
    std::string         prompt;
    AgentRole           target_role;
    uint32_t            assigned_agent_id = 0;
    AgentState          state = AgentState::IDLE;
    std::vector<uint64_t> dependencies;  // Task IDs that must complete first
    std::string         result;
    TimePoint           created_at;
    TimePoint           started_at;
    TimePoint           completed_at;
    int                 retry_count = 0;
    int                 max_retries = 3;
};

// Orchestrator Configuration
struct OrchestratorConfig {
    std::string         name = "NeuroSwarm";
    int                 num_gpus = 1;
    int                 max_agents = 16;
    int                 max_concurrent_tasks = 32;
    size_t              global_memory_budget_mb = 65536; // 64GB
    bool                enable_profiling = false;
    bool                enable_safety_checks = true;
    bool                enable_hallucination_detection = true;
    int                 metrics_port = 9090;
    std::string         redis_url = "redis://localhost:6379";

    // Circuit breaker settings
    int                 cb_failure_threshold = 5;
    int                 cb_recovery_timeout_ms = 30000;
    float               cb_half_open_success_rate = 0.5f;
};

// Agent Orchestrator
class Orchestrator {
public:
    explicit Orchestrator(const OrchestratorConfig& config);
    ~Orchestrator();

    // Lifecycle
    void initialize();
    void start();
    void stop();
    void shutdown();

    // Agent management
    uint32_t register_agent(const AgentConfig& config);
    void unregister_agent(uint32_t agent_id);
    AgentState get_agent_state(uint32_t agent_id) const;
    std::vector<uint32_t> get_agents_by_role(AgentRole role) const;

    // Task submission
    uint64_t submit_task(Task task);
    void cancel_task(uint64_t task_id);
    Task get_task_status(uint64_t task_id) const;

    // High-level reasoning pipeline
    std::string execute_reasoning_pipeline(
        const std::string& query,
        int max_iterations = 5
    );

    // Control
    void kill_agent(uint32_t agent_id);
    void kill_all_agents();
    void rollback_agent(uint32_t agent_id);
    void checkpoint();
    void restore_checkpoint(const std::string& checkpoint_id);

    // Metrics
    PerformanceMetrics get_metrics() const;
    void reset_metrics();

    // Accessors
    const OrchestratorConfig& config() const { return config_; }
    CommunicationBus* comm_bus() { return comm_bus_.get(); }
    InferenceEngine* inference_engine() { return inference_engine_.get(); }
    GpuMemoryPoolManager* memory_manager() { return memory_manager_.get(); }

private:
    void scheduler_loop();
    void monitor_loop();
    void dispatch_task(Task& task);
    uint32_t select_agent_for_task(const Task& task);
    bool check_dependencies(const Task& task) const;
    void handle_task_completion(uint64_t task_id, const std::string& result);
    void handle_task_failure(uint64_t task_id, const std::string& error);

    OrchestratorConfig config_;

    // Subsystems
    std::unique_ptr<CommunicationBus>      comm_bus_;
    std::unique_ptr<InferenceEngine>        inference_engine_;
    std::unique_ptr<GpuMemoryPoolManager>   memory_manager_;
    std::unique_ptr<MetricsCollector>        metrics_;

    // Agent registry
    mutable std::mutex agents_mutex_;
    std::unordered_map<uint32_t, std::unique_ptr<AgentRuntime>> agents_;
    std::unordered_map<uint32_t, std::unique_ptr<CircuitBreaker>> circuit_breakers_;
    uint32_t next_agent_id_ = 1;

    // Task queue
    mutable std::mutex tasks_mutex_;
    std::condition_variable task_cv_;
    std::queue<uint64_t> pending_tasks_;
    std::unordered_map<uint64_t, Task> all_tasks_;
    uint64_t next_task_id_ = 1;

    // Threads
    std::thread scheduler_thread_;
    std::thread monitor_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> initialized_{false};
};

} // namespace neuroswarm
