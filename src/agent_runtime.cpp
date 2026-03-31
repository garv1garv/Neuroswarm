// Individual agent execution, tool calling, and hallucination detection
//

#include "neuroswarm/agent_runtime.h"
#include "neuroswarm/communication.h"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <regex>
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

namespace neuroswarm {

// Construction / Destruction

AgentRuntime::AgentRuntime(uint32_t id, const AgentConfig& config,
                           InferenceEngine* engine, CommunicationBus* bus)
    : id_(id), config_(config), engine_(engine), bus_(bus),
      gpu_id_(config.gpu_id)
{
    // Create a dedicated CUDA stream for this agent
    cudaSetDevice(gpu_id_);
    cudaStreamCreate(&stream_);

    spdlog::debug("AgentRuntime {} created (role={}, gpu={})",
                  id, agent_role_str(config.role), gpu_id_);
}

AgentRuntime::~AgentRuntime() {
    stop();
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

// Lifecycle

void AgentRuntime::start() {
    state_.store(AgentState::IDLE);
    kill_requested_.store(false);
    spdlog::info("Agent {} started", id_);
}

void AgentRuntime::stop() {
    kill_requested_.store(true);
    if (exec_thread_.joinable()) {
        exec_thread_.join();
    }
    state_.store(AgentState::IDLE);
}

void AgentRuntime::kill() {
    spdlog::warn("Agent {} kill requested", id_);
    kill_requested_.store(true);
    state_.store(AgentState::KILLED);

    // Abort any pending CUDA work
    if (stream_) {
        cudaStreamSynchronize(stream_);
    }

    if (exec_thread_.joinable()) {
        exec_thread_.detach(); // Don't wait — force kill
    }
}

// Synchronous Execution — ReAct-style loop

std::string AgentRuntime::execute(const std::string& prompt, int max_steps) {
    state_.store(AgentState::RUNNING);
    execution_start_ = Clock::now();
    trace_.clear();

    spdlog::info("Agent {} executing: '{}'", id_, prompt.substr(0, 80));

    // Build system prompt with available tools
    std::string system_prompt = build_system_prompt();

    // Conversation context
    std::vector<std::string> context;
    context.push_back("System: " + system_prompt);
    context.push_back("User: " + prompt);

    std::string final_answer;

    for (int step = 0; step < max_steps && !kill_requested_.load(); step++) {
        // Check resource limits
        enforce_resource_limits();
        if (state_.load() == AgentState::KILLED) break;

        // Run inference
        std::string full_context;
        for (const auto& c : context) full_context += c + "\n";
        std::string response = run_inference(full_context);

        if (response.empty()) {
            spdlog::warn("Agent {} got empty response at step {}", id_, step);
            break;
        }

        // Parse response for actions or final answer
        if (response.find("FINAL ANSWER:") != std::string::npos) {
            // Extract final answer
            auto pos = response.find("FINAL ANSWER:");
            final_answer = response.substr(pos + 14);
            // Trim whitespace
            final_answer.erase(0, final_answer.find_first_not_of(" \n\r\t"));
            append_trace(ThoughtAction::FINAL_ANSWER, final_answer);
            break;
        }

        // Check for tool calls
        std::string action = parse_action(response);
        if (!action.empty()) {
            append_trace(ThoughtAction::ACTION, action);

            // Parse tool name and arguments
            auto sep = action.find('(');
            if (sep != std::string::npos) {
                std::string tool_name = action.substr(0, sep);
                std::string tool_args = action.substr(sep + 1);
                if (!tool_args.empty() && tool_args.back() == ')') {
                    tool_args.pop_back();
                }

                std::string tool_result = execute_tool(tool_name, tool_args);
                append_trace(ThoughtAction::OBSERVATION, tool_result);
                context.push_back("Observation: " + tool_result);
            }
        } else {
            // Treat as thought
            append_trace(ThoughtAction::THOUGHT, response);
            context.push_back("Thought: " + response);
        }
    }

    // Hallucination check
    if (!final_answer.empty() && config_.enable_sandbox) {
        std::vector<std::string> obs;
        for (const auto& t : trace_) {
            if (t.type == ThoughtAction::OBSERVATION) {
                obs.push_back(t.content);
            }
        }
        if (!check_hallucination(final_answer, obs)) {
            spdlog::warn("Agent {} hallucination detected — flagging result", id_);
            final_answer = "[HALLUCINATION WARNING] " + final_answer;
        }
    }

    state_.store(AgentState::COMPLETED);
    double total_ms = elapsed_ms(execution_start_);
    spdlog::info("Agent {} completed in {:.1f}ms ({} steps, {} tokens)",
                 id_, total_ms, trace_.size(), total_tokens_);

    return final_answer;
}

// Async Execution

void AgentRuntime::execute_async(const std::string& prompt,
                                  std::function<void(const std::string&)> callback) {
    if (exec_thread_.joinable()) {
        exec_thread_.join();
    }

    exec_thread_ = std::thread([this, prompt, callback]() {
        try {
            std::string result = execute(prompt);
            if (callback) callback(result);
        } catch (const std::exception& e) {
            spdlog::error("Agent {} async execution failed: {}", id_, e.what());
            if (callback) callback("ERROR: " + std::string(e.what()));
        }
    });
}

// Tool Management

void AgentRuntime::register_tool(const std::string& name,
                                  const std::string& description,
                                  ToolFunction func) {
    std::lock_guard<std::mutex> lock(tools_mutex_);
    tools_[name] = func;
    tool_descriptions_[name] = description;
    spdlog::debug("Agent {} registered tool: {}", id_, name);
}

void AgentRuntime::unregister_tool(const std::string& name) {
    std::lock_guard<std::mutex> lock(tools_mutex_);
    tools_.erase(name);
    tool_descriptions_.erase(name);
}

// Resource Limits

void AgentRuntime::set_resource_limits(const ResourceLimits& limits) {
    resource_limits_ = limits;
}

bool AgentRuntime::is_within_limits() const {
    // Check execution time
    if (state_.load() == AgentState::RUNNING) {
        auto elapsed = Clock::now() - execution_start_;
        if (resource_limits_.max_execution_time.count() > 0 &&
            elapsed > resource_limits_.max_execution_time) {
            return false;
        }
    }

    // Check GPU memory
    if (resource_limits_.max_gpu_memory_bytes > 0 &&
        gpu_memory_used_ > resource_limits_.max_gpu_memory_bytes) {
        return false;
    }

    return true;
}

double AgentRuntime::average_latency_ms() const {
    if (total_inferences_ == 0) return 0.0;
    return total_latency_ms_ / total_inferences_;
}

// Private Implementation

std::string AgentRuntime::run_inference(const std::string& prompt) {
    auto start = Clock::now();

    // In production, this calls the InferenceEngine with TensorRT
    // For now, simulate the interface
    std::string response;

    if (engine_ != nullptr) {
        InferenceRequest req;
        req.agent_id = id_;
        req.max_new_tokens = config_.max_tokens;
        req.temperature = config_.temperature;
        req.top_p = config_.top_p;
        req.top_k = config_.top_k;
        req.quant_mode = config_.quant_mode;
        req.stream = stream_;

        // Tokenize prompt → req.input_tokens
        // engine_->infer(req) → InferenceResult
        // Detokenize result.output_tokens → response
    }

    double ms = elapsed_ms(start);
    total_latency_ms_ += ms;
    total_inferences_++;

    return response;
}

std::string AgentRuntime::parse_action(const std::string& response) {
    // Look for action patterns:
    //   Action: tool_name(args)
    //   ```tool_call\ntool_name(args)\n```
    std::regex action_re(R"(Action:\s*(\w+\([^)]*\)))");
    std::smatch match;
    if (std::regex_search(response, match, action_re)) {
        return match[1].str();
    }

    // Alternative format
    std::regex tool_call_re(R"(```tool_call\s*\n(\w+\([^)]*\))\s*\n```)");
    if (std::regex_search(response, match, tool_call_re)) {
        return match[1].str();
    }

    return "";
}

std::string AgentRuntime::execute_tool(const std::string& tool_name,
                                        const std::string& args) {
    std::lock_guard<std::mutex> lock(tools_mutex_);
    auto it = tools_.find(tool_name);
    if (it == tools_.end()) {
        return "Error: Unknown tool '" + tool_name + "'";
    }

    auto start = Clock::now();
    try {
        std::string result = it->second(args);
        double ms = elapsed_ms(start);
        spdlog::debug("Agent {} tool '{}' completed in {:.1f}ms", id_, tool_name, ms);
        return result;
    } catch (const std::exception& e) {
        spdlog::error("Agent {} tool '{}' failed: {}", id_, tool_name, e.what());
        return "Error executing tool: " + std::string(e.what());
    }
}

bool AgentRuntime::check_hallucination(const std::string& response,
                                        const std::vector<std::string>& context) {
    // Self-consistency check:
    // Run the same prompt N times and check if outputs are consistent
    // This is a simplified version — production would use
    // embedding similarity and factual grounding

    if (context.empty()) return true; // No observations to ground against

    // Check if response contains facts not present in any observation
    // Simple heuristic: if response contains numbers/dates/names,
    // they should appear in at least one observation
    std::regex fact_re(R"(\b\d{2,}\b)"); // Numbers with 2+ digits
    auto facts_begin = std::sregex_iterator(response.begin(), response.end(), fact_re);
    auto facts_end = std::sregex_iterator();

    for (auto it = facts_begin; it != facts_end; ++it) {
        std::string fact = it->str();
        bool grounded = false;
        for (const auto& obs : context) {
            if (obs.find(fact) != std::string::npos) {
                grounded = true;
                break;
            }
        }
        if (!grounded) {
            spdlog::debug("Ungrounded fact '{}' in agent {} response", fact, id_);
            return false;
        }
    }

    return true;
}

void AgentRuntime::enforce_resource_limits() {
    if (!is_within_limits()) {
        spdlog::warn("Agent {} exceeded resource limits", id_);
        state_.store(AgentState::KILLED);
        kill_requested_.store(true);
    }
}

void AgentRuntime::append_trace(ThoughtAction::Type type, const std::string& content) {
    std::lock_guard<std::mutex> lock(trace_mutex_);
    trace_.push_back({type, content, Clock::now()});
}

// Build system prompt that includes tool descriptions
std::string build_system_prompt() {
    // This would be populated from tool_descriptions_ in a real implementation
    return R"(You are an AI agent in the NeuroSwarm multi-agent system.

You can use tools by responding with: Action: tool_name(arguments)
After each action, you will receive an Observation with the result.

When you have a final answer, respond with: FINAL ANSWER: <your answer>

Available tools will be listed below. Think step-by-step before acting.
Be precise and verify your work.)";
}

} // namespace neuroswarm
