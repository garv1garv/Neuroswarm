#include "neuroswarm/common.h"

#include <spdlog/spdlog.h>
#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/exposer.h>
#include <prometheus/registry.h>

#include <memory>
#include <string>

namespace neuroswarm {

class MetricsCollector {
public:
    explicit MetricsCollector(int port = 9090)
        : exposer_("0.0.0.0:" + std::to_string(port)),
          registry_(std::make_shared<prometheus::Registry>())
    {
        // ── Counters ─────────────────────────────────
        auto& counter_family = prometheus::BuildCounter()
            .Name("neuroswarm_inferences_total")
            .Help("Total inference requests processed")
            .Register(*registry_);
        inference_counter_ = &counter_family.Add({});

        auto& token_family = prometheus::BuildCounter()
            .Name("neuroswarm_tokens_generated_total")
            .Help("Total tokens generated across all agents")
            .Register(*registry_);
        token_counter_ = &token_family.Add({});

        auto& msg_family = prometheus::BuildCounter()
            .Name("neuroswarm_messages_total")
            .Help("Total inter-agent messages")
            .Register(*registry_);
        message_counter_ = &msg_family.Add({});

        auto& error_family = prometheus::BuildCounter()
            .Name("neuroswarm_errors_total")
            .Help("Total errors across all agents")
            .Register(*registry_);
        error_counter_ = &error_family.Add({});

        // ── Gauges ───────────────────────────────────
        auto& gpu_mem_family = prometheus::BuildGauge()
            .Name("neuroswarm_gpu_memory_bytes")
            .Help("Current GPU memory usage in bytes")
            .Register(*registry_);
        gpu_memory_gauge_ = &gpu_mem_family.Add({});

        auto& active_agents_family = prometheus::BuildGauge()
            .Name("neuroswarm_active_agents")
            .Help("Number of currently active agents")
            .Register(*registry_);
        active_agents_gauge_ = &active_agents_family.Add({});

        auto& pending_tasks_family = prometheus::BuildGauge()
            .Name("neuroswarm_pending_tasks")
            .Help("Number of pending tasks in queue")
            .Register(*registry_);
        pending_tasks_gauge_ = &pending_tasks_family.Add({});

        auto& throughput_family = prometheus::BuildGauge()
            .Name("neuroswarm_throughput_tps")
            .Help("Current throughput in tokens per second")
            .Register(*registry_);
        throughput_gauge_ = &throughput_family.Add({});

        // ── Histograms ───────────────────────────────
        auto& latency_family = prometheus::BuildHistogram()
            .Name("neuroswarm_inference_latency_ms")
            .Help("Inference latency in milliseconds")
            .Register(*registry_);
        latency_histogram_ = &latency_family.Add(
            {}, prometheus::Histogram::BucketBoundaries{
                1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000
            });

        auto& ttft_family = prometheus::BuildHistogram()
            .Name("neuroswarm_time_to_first_token_ms")
            .Help("Time to first token in milliseconds")
            .Register(*registry_);
        ttft_histogram_ = &ttft_family.Add(
            {}, prometheus::Histogram::BucketBoundaries{
                5, 10, 25, 50, 100, 250, 500, 1000, 2000
            });

        exposer_.RegisterCollectable(registry_);
        spdlog::info("Metrics server started on port {}", port);
    }

    // Record methods
    void record_inference(double latency_ms, int tokens) {
        inference_counter_->Increment();
        token_counter_->Increment(tokens);
        latency_histogram_->Observe(latency_ms);
    }

    void record_ttft(double ms) {
        ttft_histogram_->Observe(ms);
    }

    void record_message() {
        message_counter_->Increment();
    }

    void record_error() {
        error_counter_->Increment();
    }

    // Update gauges
    void set_gpu_memory(size_t bytes) {
        gpu_memory_gauge_->Set(static_cast<double>(bytes));
    }

    void set_active_agents(int count) {
        active_agents_gauge_->Set(count);
    }

    void set_pending_tasks(int count) {
        pending_tasks_gauge_->Set(count);
    }

    void set_throughput(double tps) {
        throughput_gauge_->Set(tps);
    }

private:
    prometheus::Exposer exposer_;
    std::shared_ptr<prometheus::Registry> registry_;

    prometheus::Counter* inference_counter_;
    prometheus::Counter* token_counter_;
    prometheus::Counter* message_counter_;
    prometheus::Counter* error_counter_;

    prometheus::Gauge* gpu_memory_gauge_;
    prometheus::Gauge* active_agents_gauge_;
    prometheus::Gauge* pending_tasks_gauge_;
    prometheus::Gauge* throughput_gauge_;

    prometheus::Histogram* latency_histogram_;
    prometheus::Histogram* ttft_histogram_;
};

} // namespace neuroswarm
