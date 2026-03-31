// Launches the orchestrator with a configurable agent swarm
//
// Usage: ./neuroswarm [--config config.json] [--gpus N] [--agents N]
//

#include "neuroswarm/orchestrator.h"
#include "neuroswarm/agent_runtime.h"
#include "neuroswarm/communication.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>
#include <csignal>
#include <string>
#include <memory>

using json = nlohmann::json;

static std::unique_ptr<neuroswarm::Orchestrator> g_orchestrator;

void signal_handler(int signum) {
    spdlog::warn("Received signal {} — initiating graceful shutdown", signum);
    if (g_orchestrator) {
        g_orchestrator->stop();
    }
}

void setup_logging() {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::info);
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");

    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        "neuroswarm.log", 50 * 1024 * 1024, 5);
    file_sink->set_level(spdlog::level::debug);

    auto logger = std::make_shared<spdlog::logger>("neuroswarm",
        spdlog::sinks_init_list{console_sink, file_sink});
    logger->set_level(spdlog::level::debug);
    spdlog::set_default_logger(logger);
}

neuroswarm::OrchestratorConfig load_config(const std::string& path) {
    neuroswarm::OrchestratorConfig config;

    if (!path.empty()) {
        std::ifstream file(path);
        if (file.is_open()) {
            json j;
            file >> j;

            if (j.contains("name"))       config.name = j["name"];
            if (j.contains("num_gpus"))    config.num_gpus = j["num_gpus"];
            if (j.contains("max_agents"))  config.max_agents = j["max_agents"];
            if (j.contains("redis_url"))   config.redis_url = j["redis_url"];
            if (j.contains("metrics_port")) config.metrics_port = j["metrics_port"];
            if (j.contains("enable_profiling"))
                config.enable_profiling = j["enable_profiling"];
            if (j.contains("enable_safety_checks"))
                config.enable_safety_checks = j["enable_safety_checks"];

            spdlog::info("Loaded config from {}", path);
        } else {
            spdlog::warn("Config file not found: {} — using defaults", path);
        }
    }

    return config;
}

void print_banner() {
    std::cout << R"(
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗               ║
    ║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗              ║
    ║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║              ║
    ║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║              ║
    ║   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝              ║
    ║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝              ║
    ║                                                              ║
    ║   ███████╗██╗    ██╗ █████╗ ██████╗ ███╗   ███╗              ║
    ║   ██╔════╝██║    ██║██╔══██╗██╔══██╗████╗ ████║              ║
    ║   ███████╗██║ █╗ ██║███████║██████╔╝██╔████╔██║              ║
    ║   ╚════██║██║███╗██║██╔══██║██╔══██╗██║╚██╔╝██║              ║
    ║   ███████║╚███╔███╔╝██║  ██║██║  ██║██║ ╚═╝ ██║              ║
    ║   ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝              ║
    ║                                                              ║
    ║   GPU-Native Multi-Agent Reasoning System    v1.0.0          ║
    ║   CUDA-Accelerated | Zero-Copy IPC | TensorRT-LLM            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    )" << std::endl;
}

int main(int argc, char** argv) {
    print_banner();
    setup_logging();

    // Parse command-line arguments
    std::string config_path;
    int num_gpus = 1;
    int num_executors = 5;
    std::string query;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc)     config_path = argv[++i];
        else if (arg == "--gpus" && i + 1 < argc)   num_gpus = std::atoi(argv[++i]);
        else if (arg == "--executors" && i + 1 < argc) num_executors = std::atoi(argv[++i]);
        else if (arg == "--query" && i + 1 < argc)  query = argv[++i];
        else if (arg == "--help") {
            std::cout << "Usage: neuroswarm [options]\n"
                      << "  --config <path>     Config file (JSON)\n"
                      << "  --gpus <N>          Number of GPUs to use\n"
                      << "  --executors <N>     Number of executor agents\n"
                      << "  --query <text>      Query to run through reasoning pipeline\n"
                      << "  --help              Show this help\n";
            return 0;
        }
    }

    // Setup signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        // Load configuration
        auto config = load_config(config_path);
        config.num_gpus = num_gpus;

        // Create and initialize orchestrator
        g_orchestrator = std::make_unique<neuroswarm::Orchestrator>(config);
        g_orchestrator->initialize();

        // Register agents
        spdlog::info("Registering agent swarm...");

        // Planner agent (Llama 3.1 70B equivalent — needs more GPU memory)
        neuroswarm::AgentConfig planner_cfg;
        planner_cfg.name = "Planner-Alpha";
        planner_cfg.role = neuroswarm::AgentRole::PLANNER;
        planner_cfg.quant_mode = neuroswarm::QuantMode::INT8;
        planner_cfg.gpu_id = 0;
        planner_cfg.max_memory_mb = 40960;
        planner_cfg.max_tokens = 4096;
        planner_cfg.temperature = 0.3f; // Lower temp for strategic planning
        auto planner_id = g_orchestrator->register_agent(planner_cfg);

        // Executor agents (Llama 3.1 8B × N — lightweight workers)
        std::vector<uint32_t> executor_ids;
        for (int i = 0; i < num_executors; i++) {
            neuroswarm::AgentConfig exec_cfg;
            exec_cfg.name = "Executor-" + std::to_string(i + 1);
            exec_cfg.role = neuroswarm::AgentRole::EXECUTOR;
            exec_cfg.quant_mode = neuroswarm::QuantMode::FP16;
            exec_cfg.gpu_id = i % num_gpus;
            exec_cfg.max_memory_mb = 8192;
            exec_cfg.max_tokens = 2048;
            exec_cfg.temperature = 0.7f;
            executor_ids.push_back(g_orchestrator->register_agent(exec_cfg));
        }

        // Validator agent (Llama 3.1 8B — critical evaluation)
        neuroswarm::AgentConfig validator_cfg;
        validator_cfg.name = "Validator-Omega";
        validator_cfg.role = neuroswarm::AgentRole::VALIDATOR;
        validator_cfg.quant_mode = neuroswarm::QuantMode::FP16;
        validator_cfg.gpu_id = 0;
        validator_cfg.max_memory_mb = 8192;
        validator_cfg.max_tokens = 2048;
        validator_cfg.temperature = 0.1f; // Very low temp for validation
        auto validator_id = g_orchestrator->register_agent(validator_cfg);

        spdlog::info("Agent swarm ready: 1 planner + {} executors + 1 validator",
                     num_executors);

        // Start orchestrator
        g_orchestrator->start();

        // If a query was provided, run it
        if (!query.empty()) {
            spdlog::info("Executing query through reasoning pipeline...");
            std::string result = g_orchestrator->execute_reasoning_pipeline(query);
            std::cout << "\n═══ RESULT ════════════════════════════════\n"
                      << result
                      << "\n═══════════════════════════════════════════\n" << std::endl;
        } else {
            // Interactive REPL mode
            spdlog::info("Entering interactive mode. Type 'quit' to exit.");
            std::string input;
            while (true) {
                std::cout << "\n[NeuroSwarm] > ";
                std::getline(std::cin, input);

                if (input == "quit" || input == "exit") break;
                if (input == "status") {
                    auto metrics = g_orchestrator->get_metrics();
                    std::cout << "  Inferences: " << metrics.total_inferences << "\n"
                              << "  Messages:   " << metrics.total_messages << "\n"
                              << "  GPU Memory: " << metrics.gpu_memory_used_mb << " MB\n";
                    continue;
                }
                if (input.empty()) continue;

                std::string result = g_orchestrator->execute_reasoning_pipeline(input);
                std::cout << "\n" << result << std::endl;
            }
        }

        // Shutdown
        g_orchestrator->shutdown();
        g_orchestrator.reset();

    } catch (const std::exception& e) {
        spdlog::critical("Fatal error: {}", e.what());
        return 1;
    }

    spdlog::info("NeuroSwarm exited cleanly");
    return 0;
}
