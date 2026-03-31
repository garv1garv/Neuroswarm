// Prevents cascading failures in multi-agent systems
//
// States: CLOSED → OPEN (on failure threshold) → HALF_OPEN → CLOSED/OPEN
//

#include "neuroswarm/common.h"

#include <spdlog/spdlog.h>
#include <atomic>
#include <mutex>
#include <chrono>
#include <functional>
#include <deque>

namespace neuroswarm {

class CircuitBreaker {
public:
    struct Config {
        int     failure_threshold     = 5;       // Failures before tripping
        int     success_threshold     = 3;       // Successes in half-open to close
        int     recovery_timeout_ms   = 30000;   // Time before half-open attempt
        float   failure_rate_threshold = 0.5f;   // Alternative: rate-based tripping
        int     sliding_window_size   = 20;      // Window for rate calculation
        bool    log_transitions       = true;
    };

    explicit CircuitBreaker(Config cfg = {})
        : config_(cfg), state_(CircuitState::CLOSED)
    {}

    /// Execute a function through the circuit breaker
    template<typename Func>
    auto execute(Func&& fn) -> decltype(fn()) {
        if (!allow_request()) {
            throw std::runtime_error("Circuit breaker is OPEN — request rejected");
        }

        try {
            auto result = fn();
            record_success();
            return result;
        } catch (...) {
            record_failure();
            throw;
        }
    }

    /// Check if a request should be allowed
    bool allow_request() {
        std::lock_guard<std::mutex> lock(mutex_);

        switch (state_) {
            case CircuitState::CLOSED:
                return true;

            case CircuitState::OPEN: {
                // Check if recovery timeout has elapsed
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    Clock::now() - last_failure_time_
                ).count();

                if (elapsed >= config_.recovery_timeout_ms) {
                    transition_to(CircuitState::HALF_OPEN);
                    return true; // Allow one probe request
                }
                return false;
            }

            case CircuitState::HALF_OPEN:
                return true; // Allow probe requests
        }

        return false;
    }

    /// Record a successful operation
    void record_success() {
        std::lock_guard<std::mutex> lock(mutex_);
        total_successes_++;

        results_.push_back(true);
        trim_window();

        switch (state_) {
            case CircuitState::HALF_OPEN:
                consecutive_successes_++;
                if (consecutive_successes_ >= config_.success_threshold) {
                    transition_to(CircuitState::CLOSED);
                }
                break;

            case CircuitState::CLOSED:
                consecutive_failures_ = 0;
                break;

            default:
                break;
        }
    }

    /// Record a failed operation
    void record_failure() {
        std::lock_guard<std::mutex> lock(mutex_);
        total_failures_++;
        last_failure_time_ = Clock::now();

        results_.push_back(false);
        trim_window();

        switch (state_) {
            case CircuitState::CLOSED: {
                consecutive_failures_++;

                // Check count-based threshold
                if (consecutive_failures_ >= config_.failure_threshold) {
                    transition_to(CircuitState::OPEN);
                    break;
                }

                // Check rate-based threshold
                if (results_.size() >= static_cast<size_t>(config_.sliding_window_size)) {
                    float rate = failure_rate();
                    if (rate >= config_.failure_rate_threshold) {
                        transition_to(CircuitState::OPEN);
                    }
                }
                break;
            }

            case CircuitState::HALF_OPEN:
                // Failed during probe — go back to OPEN
                consecutive_successes_ = 0;
                transition_to(CircuitState::OPEN);
                break;

            default:
                break;
        }
    }

    /// Force the circuit breaker to a specific state
    void force_state(CircuitState state) {
        std::lock_guard<std::mutex> lock(mutex_);
        transition_to(state);
    }

    /// Reset all counters
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = CircuitState::CLOSED;
        consecutive_failures_ = 0;
        consecutive_successes_ = 0;
        total_failures_ = 0;
        total_successes_ = 0;
        results_.clear();
    }

    // Getters
    CircuitState state() const { return state_; }
    int consecutive_failures() const { return consecutive_failures_; }
    uint64_t total_failures() const { return total_failures_; }
    uint64_t total_successes() const { return total_successes_; }

    float failure_rate() const {
        if (results_.empty()) return 0.0f;
        int failures = 0;
        for (bool success : results_) {
            if (!success) failures++;
        }
        return static_cast<float>(failures) / results_.size();
    }

    const char* state_str() const {
        switch (state_) {
            case CircuitState::CLOSED:    return "CLOSED";
            case CircuitState::OPEN:      return "OPEN";
            case CircuitState::HALF_OPEN: return "HALF_OPEN";
            default:                      return "UNKNOWN";
        }
    }

private:
    void transition_to(CircuitState new_state) {
        if (state_ == new_state) return;

        if (config_.log_transitions) {
            spdlog::warn("Circuit breaker: {} → {}",
                         state_str_for(state_), state_str_for(new_state));
        }

        state_ = new_state;

        switch (new_state) {
            case CircuitState::CLOSED:
                consecutive_failures_ = 0;
                consecutive_successes_ = 0;
                break;
            case CircuitState::OPEN:
                consecutive_successes_ = 0;
                break;
            case CircuitState::HALF_OPEN:
                consecutive_successes_ = 0;
                break;
        }
    }

    void trim_window() {
        while (results_.size() > static_cast<size_t>(config_.sliding_window_size)) {
            results_.pop_front();
        }
    }

    static const char* state_str_for(CircuitState s) {
        switch (s) {
            case CircuitState::CLOSED:    return "CLOSED";
            case CircuitState::OPEN:      return "OPEN";
            case CircuitState::HALF_OPEN: return "HALF_OPEN";
            default:                      return "UNKNOWN";
        }
    }

    Config config_;
    CircuitState state_;
    std::mutex mutex_;

    int consecutive_failures_ = 0;
    int consecutive_successes_ = 0;
    uint64_t total_failures_ = 0;
    uint64_t total_successes_ = 0;
    TimePoint last_failure_time_;

    std::deque<bool> results_; // Sliding window: true=success, false=failure
};

} // namespace neuroswarm
