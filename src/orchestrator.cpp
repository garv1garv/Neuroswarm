// Central coordinator: agent lifecycle, task scheduling, reasoning pipeline
//

#include "neuroswarm/orchestrator.h"
#include "neuroswarm/agent_runtime.h"
#include "neuroswarm/communication.h"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <sstream>

using json = nlohmann::json;

namespace neuroswarm {

// Construction / Destruction

Orchestrator::Orchestrator(const OrchestratorConfig& config)
    : config_(config)
{
    spdlog::info("NeuroSwarm Orchestrator created: {}", config.name);
}

Orchestrator::~Orchestrator() {
    if (running_.load()) {
        shutdown();
    }
}

// Lifecycle

void Orchestrator::initialize() {
    spdlog::info("Initializing NeuroSwarm with {} GPUs", config_.num_gpus);

    // Initialize CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        spdlog::error("No CUDA devices found!");
        throw std::runtime_error("No CUDA devices available");
    }
    spdlog::info("Found {} CUDA device(s)", device_count);

    for (int i = 0; i < std::min(device_count, config_.num_gpus); i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        spdlog::info("  GPU {}: {} ({}MB, SM {}{})",
                     i, prop.name,
                     prop.totalGlobalMem / (1024 * 1024),
                     prop.major, prop.minor);
    }

    // Initialize subsystems
    comm_bus_ = std::make_unique<CommunicationBus>(config_.num_gpus);
    // inference_engine_ and memory_manager_ would be initialized here
    // with TensorRT models and pool configuration

    initialized_.store(true);
    spdlog::info("NeuroSwarm initialization complete");
}

void Orchestrator::start() {
    if (!initialized_.load()) {
        throw std::runtime_error("Orchestrator not initialized. Call initialize() first.");
    }

    running_.store(true);

    // Start scheduler thread
    scheduler_thread_ = std::thread(&Orchestrator::scheduler_loop, this);

    // Start monitor thread
    monitor_thread_ = std::thread(&Orchestrator::monitor_loop, this);

    spdlog::info("NeuroSwarm started — scheduler and monitor threads running");
}

void Orchestrator::stop() {
    spdlog::info("Stopping NeuroSwarm...");
    running_.store(false);
    task_cv_.notify_all();

    if (scheduler_thread_.joinable()) scheduler_thread_.join();
    if (monitor_thread_.joinable()) monitor_thread_.join();

    spdlog::info("NeuroSwarm stopped");
}

void Orchestrator::shutdown() {
    stop();
    kill_all_agents();

    comm_bus_.reset();
    inference_engine_.reset();
    memory_manager_.reset();
    metrics_.reset();

    initialized_.store(false);
    spdlog::info("NeuroSwarm shutdown complete");
}

// Agent Management

uint32_t Orchestrator::register_agent(const AgentConfig& config) {
    std::lock_guard<std::mutex> lock(agents_mutex_);

    if (agents_.size() >= static_cast<size_t>(config_.max_agents)) {
        throw std::runtime_error("Maximum agent limit reached");
    }

    uint32_t id = next_agent_id_++;
    auto runtime = std::make_unique<AgentRuntime>(
        id, config, inference_engine_.get(), comm_bus_.get());

    // Create circuit breaker for this agent
    circuit_breakers_[id] = std::make_unique<CircuitBreaker>();

    agents_[id] = std::move(runtime);

    spdlog::info("Registered agent '{}' (id={}, role={}, gpu={})",
                 config.name, id, agent_role_str(config.role), config.gpu_id);

    return id;
}

void Orchestrator::unregister_agent(uint32_t agent_id) {
    std::lock_guard<std::mutex> lock(agents_mutex_);

    auto it = agents_.find(agent_id);
    if (it != agents_.end()) {
        it->second->stop();
        agents_.erase(it);
        circuit_breakers_.erase(agent_id);
        spdlog::info("Unregistered agent {}", agent_id);
    }
}

AgentState Orchestrator::get_agent_state(uint32_t agent_id) const {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    auto it = agents_.find(agent_id);
    return (it != agents_.end()) ? it->second->state() : AgentState::KILLED;
}

std::vector<uint32_t> Orchestrator::get_agents_by_role(AgentRole role) const {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    std::vector<uint32_t> result;
    for (const auto& [id, agent] : agents_) {
        if (agent->config().role == role) {
            result.push_back(id);
        }
    }
    return result;
}

// Task Management

uint64_t Orchestrator::submit_task(Task task) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);

    task.id = next_task_id_++;
    task.state = AgentState::IDLE;
    task.created_at = Clock::now();

    uint64_t id = task.id;
    all_tasks_[id] = std::move(task);
    pending_tasks_.push(id);

    task_cv_.notify_one();
    spdlog::debug("Task {} submitted: {}", id, all_tasks_[id].description);

    return id;
}

void Orchestrator::cancel_task(uint64_t task_id) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    auto it = all_tasks_.find(task_id);
    if (it != all_tasks_.end()) {
        it->second.state = AgentState::KILLED;
        spdlog::info("Task {} cancelled", task_id);
    }
}

Task Orchestrator::get_task_status(uint64_t task_id) const {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    auto it = all_tasks_.find(task_id);
    if (it != all_tasks_.end()) return it->second;
    throw std::runtime_error("Task not found: " + std::to_string(task_id));
}

// Reasoning Pipeline

std::string Orchestrator::execute_reasoning_pipeline(
    const std::string& query, int max_iterations
) {
    spdlog::info("Starting reasoning pipeline: '{}'", query.substr(0, 100));
    auto pipeline_start = Clock::now();

    // Step 1: Planner creates a plan
    Task plan_task;
    plan_task.description = "Create execution plan";
    plan_task.prompt = "You are a planning agent. Given the following query, create "
                       "a detailed step-by-step plan.\n\nQuery: " + query +
                       "\n\nPlan:";
    plan_task.target_role = AgentRole::PLANNER;
    uint64_t plan_id = submit_task(std::move(plan_task));

    // Wait for plan
    std::string plan_result;
    {
        std::unique_lock<std::mutex> lock(tasks_mutex_);
        task_cv_.wait(lock, [&]() {
            auto& t = all_tasks_[plan_id];
            return t.state == AgentState::COMPLETED || t.state == AgentState::FAILED;
        });
        plan_result = all_tasks_[plan_id].result;
    }

    if (plan_result.empty()) {
        return "Error: Planning agent failed to produce a plan.";
    }

    // Step 2: Parse plan into executor sub-tasks
    std::vector<uint64_t> executor_task_ids;
    // In a real system, we'd parse the plan into discrete steps
    // For now, submit a single executor task
    Task exec_task;
    exec_task.description = "Execute plan step";
    exec_task.prompt = "You are an executor agent. Execute the following plan:\n\n" +
                       plan_result + "\n\nProvide the results:";
    exec_task.target_role = AgentRole::EXECUTOR;
    exec_task.dependencies = {plan_id};
    uint64_t exec_id = submit_task(std::move(exec_task));
    executor_task_ids.push_back(exec_id);

    // Wait for executors
    std::string exec_results;
    {
        std::unique_lock<std::mutex> lock(tasks_mutex_);
        for (auto eid : executor_task_ids) {
            task_cv_.wait(lock, [&]() {
                auto& t = all_tasks_[eid];
                return t.state == AgentState::COMPLETED || t.state == AgentState::FAILED;
            });
            exec_results += all_tasks_[eid].result + "\n";
        }
    }

    // Step 3: Validator checks results
    Task val_task;
    val_task.description = "Validate results";
    val_task.prompt = "You are a validator agent. Verify the following results for "
                      "correctness and completeness.\n\nOriginal Query: " + query +
                      "\n\nPlan: " + plan_result +
                      "\n\nResults: " + exec_results +
                      "\n\nValidation:";
    val_task.target_role = AgentRole::VALIDATOR;
    val_task.dependencies = executor_task_ids;
    uint64_t val_id = submit_task(std::move(val_task));

    // Wait for validation
    std::string final_result;
    {
        std::unique_lock<std::mutex> lock(tasks_mutex_);
        task_cv_.wait(lock, [&]() {
            auto& t = all_tasks_[val_id];
            return t.state == AgentState::COMPLETED || t.state == AgentState::FAILED;
        });
        final_result = all_tasks_[val_id].result;
    }

    double total_ms = elapsed_ms(pipeline_start);
    spdlog::info("Reasoning pipeline complete in {:.1f}ms", total_ms);

    return final_result;
}

// Control

void Orchestrator::kill_agent(uint32_t agent_id) {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    auto it = agents_.find(agent_id);
    if (it != agents_.end()) {
        it->second->kill();
        spdlog::warn("Agent {} killed", agent_id);
    }
}

void Orchestrator::kill_all_agents() {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    for (auto& [id, agent] : agents_) {
        agent->kill();
    }
    spdlog::warn("All agents killed");
}

void Orchestrator::rollback_agent(uint32_t agent_id) {
    // Send rollback message via comm bus
    Message msg;
    msg.type = MessageType::ROLLBACK;
    msg.src_agent_id = 0; // Orchestrator
    msg.dst_agent_id = agent_id;
    msg.timestamp_ns = now_ns();
    comm_bus_->send(msg);

    spdlog::info("Rollback initiated for agent {}", agent_id);
}

// Metrics

PerformanceMetrics Orchestrator::get_metrics() const {
    PerformanceMetrics pm{};

    std::lock_guard<std::mutex> lock(agents_mutex_);
    for (const auto& [id, agent] : agents_) {
        pm.total_inferences += agent->total_tokens_generated();
        pm.gpu_memory_used_mb += agent->gpu_memory_used() / (1024 * 1024);
    }

    pm.total_messages = comm_bus_ ? comm_bus_->total_messages() : 0;

    // Calculate throughput
    // In production, this would use sliding window counters

    return pm;
}

// Internal: Scheduler Loop

void Orchestrator::scheduler_loop() {
    spdlog::debug("Scheduler thread started");

    while (running_.load()) {
        uint64_t task_id = 0;

        {
            std::unique_lock<std::mutex> lock(tasks_mutex_);
            task_cv_.wait_for(lock, std::chrono::milliseconds(100), [&]() {
                return !pending_tasks_.empty() || !running_.load();
            });

            if (!running_.load()) break;
            if (pending_tasks_.empty()) continue;

            task_id = pending_tasks_.front();
        }

        // Check dependencies
        {
            std::lock_guard<std::mutex> lock(tasks_mutex_);
            auto& task = all_tasks_[task_id];
            if (!check_dependencies(task)) {
                continue; // Dependencies not met — retry later
            }
            pending_tasks_.pop();
            dispatch_task(task);
        }
    }

    spdlog::debug("Scheduler thread exiting");
}

void Orchestrator::dispatch_task(Task& task) {
    uint32_t agent_id = select_agent_for_task(task);
    if (agent_id == 0) {
        spdlog::warn("No available agent for task {} (role={})",
                     task.id, agent_role_str(task.target_role));
        // Re-queue
        pending_tasks_.push(task.id);
        return;
    }

    task.assigned_agent_id = agent_id;
    task.state = AgentState::RUNNING;
    task.started_at = Clock::now();

    // Execute asynchronously
    std::lock_guard<std::mutex> lock(agents_mutex_);
    auto it = agents_.find(agent_id);
    if (it != agents_.end()) {
        uint64_t tid = task.id;
        it->second->execute_async(task.prompt,
            [this, tid](const std::string& result) {
                handle_task_completion(tid, result);
            });
    }
}

uint32_t Orchestrator::select_agent_for_task(const Task& task) {
    std::lock_guard<std::mutex> lock(agents_mutex_);

    // Find idle agents with matching role
    uint32_t best_agent = 0;
    for (const auto& [id, agent] : agents_) {
        if (agent->config().role == task.target_role &&
            agent->state() == AgentState::IDLE)
        {
            // Check circuit breaker
            auto cb_it = circuit_breakers_.find(id);
            if (cb_it != circuit_breakers_.end()) {
                // Circuit breaker check would go here
            }
            best_agent = id;
            break; // Take first available
        }
    }

    return best_agent;
}

bool Orchestrator::check_dependencies(const Task& task) const {
    for (uint64_t dep_id : task.dependencies) {
        auto it = all_tasks_.find(dep_id);
        if (it == all_tasks_.end() || it->second.state != AgentState::COMPLETED) {
            return false;
        }
    }
    return true;
}

void Orchestrator::handle_task_completion(uint64_t task_id, const std::string& result) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    auto it = all_tasks_.find(task_id);
    if (it != all_tasks_.end()) {
        it->second.state = AgentState::COMPLETED;
        it->second.result = result;
        it->second.completed_at = Clock::now();

        double ms = elapsed_ms(it->second.started_at);
        spdlog::info("Task {} completed in {:.1f}ms by agent {}",
                     task_id, ms, it->second.assigned_agent_id);
    }
    task_cv_.notify_all();
}

void Orchestrator::handle_task_failure(uint64_t task_id, const std::string& error) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    auto it = all_tasks_.find(task_id);
    if (it != all_tasks_.end()) {
        it->second.retry_count++;
        if (it->second.retry_count < it->second.max_retries) {
            spdlog::warn("Task {} failed (attempt {}/{}): {}",
                        task_id, it->second.retry_count, it->second.max_retries, error);
            it->second.state = AgentState::IDLE;
            pending_tasks_.push(task_id);
        } else {
            spdlog::error("Task {} permanently failed after {} retries: {}",
                         task_id, it->second.max_retries, error);
            it->second.state = AgentState::FAILED;
            it->second.result = "ERROR: " + error;
        }
    }
    task_cv_.notify_all();
}

// Internal: Monitor Loop

void Orchestrator::monitor_loop() {
    spdlog::debug("Monitor thread started");

    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        if (!running_.load()) break;

        // Check agent health
        std::lock_guard<std::mutex> lock(agents_mutex_);
        for (const auto& [id, agent] : agents_) {
            if (agent->state() == AgentState::RUNNING) {
                // Check for timeout
                if (!agent->is_within_limits()) {
                    spdlog::warn("Agent {} exceeded resource limits — killing", id);
                    agent->kill();
                }
            }
        }

        // Log metrics
        auto metrics = get_metrics();
        spdlog::debug("Metrics: agents={}, tasks={}, messages={}, gpu_mem={}MB",
                      agents_.size(), all_tasks_.size(),
                      metrics.total_messages, metrics.gpu_memory_used_mb);
    }

    spdlog::debug("Monitor thread exiting");
}

} // namespace neuroswarm
