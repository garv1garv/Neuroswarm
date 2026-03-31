// Resource isolation and safety constraints for agent execution
//

#include "neuroswarm/common.h"

#include <spdlog/spdlog.h>
#include <vector>
#include <string>
#include <mutex>
#include <unordered_set>
#include <regex>
#include <functional>

namespace neuroswarm {

// Sandbox Policy
struct SandboxPolicy {
    ResourceLimits  resource_limits;

    // File system restrictions
    std::vector<std::string> allowed_paths;
    std::vector<std::string> blocked_paths;
    bool allow_file_write = false;
    bool allow_file_delete = false;

    // Network restrictions
    bool allow_network = false;
    std::vector<std::string> allowed_hosts;
    std::vector<int> allowed_ports;

    // Command restrictions
    bool allow_shell_commands = false;
    std::unordered_set<std::string> allowed_commands;
    std::unordered_set<std::string> blocked_commands;

    // Content restrictions
    std::vector<std::string> blocked_patterns;  // Regex patterns
    int max_output_length = 100000;             // Characters

    // GPU restrictions
    bool allow_gpu_allocation = true;
    size_t max_gpu_alloc_per_call = 1024 * 1024 * 1024; // 1GB
};

// Sandbox — Enforces policies on agent actions
class Sandbox {
public:
    explicit Sandbox(uint32_t agent_id, const SandboxPolicy& policy = {})
        : agent_id_(agent_id), policy_(policy)
    {
        // Compile blocked patterns
        for (const auto& pattern : policy_.blocked_patterns) {
            try {
                blocked_regexes_.emplace_back(pattern, std::regex::optimize);
            } catch (const std::regex_error& e) {
                spdlog::warn("Invalid blocked pattern '{}': {}", pattern, e.what());
            }
        }
    }

    /// Check if a file path is allowed
    bool check_file_access(const std::string& path, bool write) const {
        if (write && !policy_.allow_file_write) {
            log_violation("File write blocked: " + path);
            return false;
        }

        // Check blocked paths
        for (const auto& blocked : policy_.blocked_paths) {
            if (path.find(blocked) == 0) {
                log_violation("Blocked path access: " + path);
                return false;
            }
        }

        // Check allowed paths (if specified, whitelist mode)
        if (!policy_.allowed_paths.empty()) {
            bool allowed = false;
            for (const auto& ap : policy_.allowed_paths) {
                if (path.find(ap) == 0) {
                    allowed = true;
                    break;
                }
            }
            if (!allowed) {
                log_violation("Path not in allowed list: " + path);
                return false;
            }
        }

        return true;
    }

    /// Check if a network request is allowed
    bool check_network(const std::string& host, int port) const {
        if (!policy_.allow_network) {
            log_violation("Network access blocked: " + host + ":" + std::to_string(port));
            return false;
        }

        if (!policy_.allowed_hosts.empty()) {
            bool found = false;
            for (const auto& h : policy_.allowed_hosts) {
                if (host == h || host.find(h) != std::string::npos) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                log_violation("Host not allowed: " + host);
                return false;
            }
        }

        if (!policy_.allowed_ports.empty()) {
            bool found = std::find(policy_.allowed_ports.begin(),
                                   policy_.allowed_ports.end(), port)
                         != policy_.allowed_ports.end();
            if (!found) {
                log_violation("Port not allowed: " + std::to_string(port));
                return false;
            }
        }

        return true;
    }

    /// Check if a shell command is allowed
    bool check_command(const std::string& command) const {
        if (!policy_.allow_shell_commands) {
            log_violation("Shell command blocked: " + command);
            return false;
        }

        // Extract base command
        std::string base_cmd = command.substr(0, command.find(' '));

        if (policy_.blocked_commands.count(base_cmd)) {
            log_violation("Blocked command: " + base_cmd);
            return false;
        }

        if (!policy_.allowed_commands.empty() &&
            !policy_.allowed_commands.count(base_cmd)) {
            log_violation("Command not in allowed list: " + base_cmd);
            return false;
        }

        return true;
    }

    /// Check output content for safety
    bool check_output(const std::string& content) const {
        if (content.length() > static_cast<size_t>(policy_.max_output_length)) {
            log_violation("Output exceeds maximum length");
            return false;
        }

        for (const auto& regex : blocked_regexes_) {
            if (std::regex_search(content, regex)) {
                log_violation("Output contains blocked pattern");
                return false;
            }
        }

        return true;
    }

    /// Check GPU memory allocation
    bool check_gpu_alloc(size_t size) const {
        if (!policy_.allow_gpu_allocation) {
            log_violation("GPU allocation blocked");
            return false;
        }

        if (size > policy_.max_gpu_alloc_per_call) {
            log_violation("GPU allocation exceeds limit: " +
                         std::to_string(size / (1024*1024)) + "MB");
            return false;
        }

        return true;
    }

    /// Get violation log
    std::vector<std::string> violations() const {
        std::lock_guard<std::mutex> lock(log_mutex_);
        return violations_;
    }

    uint32_t agent_id() const { return agent_id_; }
    const SandboxPolicy& policy() const { return policy_; }

private:
    void log_violation(const std::string& msg) const {
        std::lock_guard<std::mutex> lock(log_mutex_);
        violations_.push_back(msg);
        spdlog::warn("[Sandbox:Agent{}] {}", agent_id_, msg);
    }

    uint32_t agent_id_;
    SandboxPolicy policy_;
    std::vector<std::regex> blocked_regexes_;

    mutable std::mutex log_mutex_;
    mutable std::vector<std::string> violations_;
};

// Default Sandbox Policies

inline SandboxPolicy strict_policy() {
    SandboxPolicy p;
    p.allow_file_write = false;
    p.allow_file_delete = false;
    p.allow_network = false;
    p.allow_shell_commands = false;
    p.resource_limits.max_execution_time = std::chrono::seconds(60);
    p.resource_limits.max_gpu_memory_bytes = 4ULL * 1024 * 1024 * 1024; // 4GB
    p.blocked_patterns = {
        R"(password|secret|token|api_key)",  // Sensitive data patterns
        R"(rm\s+-rf|format\s+[A-Z]:)",       // Dangerous commands
    };
    return p;
}

inline SandboxPolicy permissive_policy() {
    SandboxPolicy p;
    p.allow_file_write = true;
    p.allow_network = true;
    p.allow_shell_commands = true;
    p.resource_limits.max_execution_time = std::chrono::seconds(300);
    p.resource_limits.max_gpu_memory_bytes = 32ULL * 1024 * 1024 * 1024; // 32GB
    p.blocked_commands = {"rm", "format", "shutdown", "reboot"};
    return p;
}

inline SandboxPolicy driver_testing_policy() {
    SandboxPolicy p;
    p.allow_file_write = true;
    p.allow_network = true;
    p.allow_shell_commands = true;
    p.allowed_commands = {
        "nvidia-smi", "nsys", "ncu", "cuda-gdb",
        "python", "python3", "pytest",
        "git", "make", "cmake"
    };
    p.allowed_paths = {"/tmp/neuroswarm/", "/var/log/neuroswarm/"};
    p.resource_limits.max_execution_time = std::chrono::minutes(30);
    p.resource_limits.max_gpu_memory_bytes = 16ULL * 1024 * 1024 * 1024;
    return p;
}

} // namespace neuroswarm
