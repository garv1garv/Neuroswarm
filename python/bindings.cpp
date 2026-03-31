// Exposes the C++ orchestrator, agents, and GPU operations to Python
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "neuroswarm/common.h"
#include "neuroswarm/orchestrator.h"
#include "neuroswarm/agent_runtime.h"
#include "neuroswarm/communication.h"

namespace py = pybind11;

PYBIND11_MODULE(_neuroswarm_cpp, m) {
    m.doc() = "NeuroSwarm — GPU-Native Multi-Agent Reasoning System (C++ bindings)";

    // ── Enums ────────────────────────────────────────────
    py::enum_<neuroswarm::AgentRole>(m, "AgentRole")
        .value("PLANNER",   neuroswarm::AgentRole::PLANNER)
        .value("EXECUTOR",  neuroswarm::AgentRole::EXECUTOR)
        .value("VALIDATOR", neuroswarm::AgentRole::VALIDATOR)
        .value("MONITOR",   neuroswarm::AgentRole::MONITOR)
        .export_values();

    py::enum_<neuroswarm::AgentState>(m, "AgentState")
        .value("IDLE",      neuroswarm::AgentState::IDLE)
        .value("RUNNING",   neuroswarm::AgentState::RUNNING)
        .value("WAITING",   neuroswarm::AgentState::WAITING)
        .value("COMPLETED", neuroswarm::AgentState::COMPLETED)
        .value("FAILED",    neuroswarm::AgentState::FAILED)
        .value("KILLED",    neuroswarm::AgentState::KILLED)
        .export_values();

    py::enum_<neuroswarm::QuantMode>(m, "QuantMode")
        .value("FP32", neuroswarm::QuantMode::FP32)
        .value("FP16", neuroswarm::QuantMode::FP16)
        .value("INT8", neuroswarm::QuantMode::INT8)
        .value("INT4", neuroswarm::QuantMode::INT4)
        .export_values();

    // ── AgentConfig ──────────────────────────────────────
    py::class_<neuroswarm::AgentConfig>(m, "AgentConfig")
        .def(py::init<>())
        .def_readwrite("name",          &neuroswarm::AgentConfig::name)
        .def_readwrite("role",          &neuroswarm::AgentConfig::role)
        .def_readwrite("quant_mode",    &neuroswarm::AgentConfig::quant_mode)
        .def_readwrite("gpu_id",        &neuroswarm::AgentConfig::gpu_id)
        .def_readwrite("max_memory_mb", &neuroswarm::AgentConfig::max_memory_mb)
        .def_readwrite("max_tokens",    &neuroswarm::AgentConfig::max_tokens)
        .def_readwrite("temperature",   &neuroswarm::AgentConfig::temperature)
        .def_readwrite("top_p",         &neuroswarm::AgentConfig::top_p)
        .def_readwrite("top_k",         &neuroswarm::AgentConfig::top_k)
        .def_readwrite("timeout_ms",    &neuroswarm::AgentConfig::timeout_ms)
        .def_readwrite("enable_sandbox",&neuroswarm::AgentConfig::enable_sandbox);

    // ── OrchestratorConfig ───────────────────────────────
    py::class_<neuroswarm::OrchestratorConfig>(m, "OrchestratorConfig")
        .def(py::init<>())
        .def_readwrite("name",                      &neuroswarm::OrchestratorConfig::name)
        .def_readwrite("num_gpus",                   &neuroswarm::OrchestratorConfig::num_gpus)
        .def_readwrite("max_agents",                 &neuroswarm::OrchestratorConfig::max_agents)
        .def_readwrite("max_concurrent_tasks",       &neuroswarm::OrchestratorConfig::max_concurrent_tasks)
        .def_readwrite("global_memory_budget_mb",    &neuroswarm::OrchestratorConfig::global_memory_budget_mb)
        .def_readwrite("enable_profiling",           &neuroswarm::OrchestratorConfig::enable_profiling)
        .def_readwrite("enable_safety_checks",       &neuroswarm::OrchestratorConfig::enable_safety_checks)
        .def_readwrite("enable_hallucination_detection",
                       &neuroswarm::OrchestratorConfig::enable_hallucination_detection)
        .def_readwrite("metrics_port",               &neuroswarm::OrchestratorConfig::metrics_port)
        .def_readwrite("redis_url",                  &neuroswarm::OrchestratorConfig::redis_url);

    // ── PerformanceMetrics ───────────────────────────────
    py::class_<neuroswarm::PerformanceMetrics>(m, "PerformanceMetrics")
        .def(py::init<>())
        .def_readonly("total_throughput_tps",  &neuroswarm::PerformanceMetrics::total_throughput_tps)
        .def_readonly("avg_latency_ms",        &neuroswarm::PerformanceMetrics::avg_latency_ms)
        .def_readonly("p50_latency_ms",        &neuroswarm::PerformanceMetrics::p50_latency_ms)
        .def_readonly("p99_latency_ms",        &neuroswarm::PerformanceMetrics::p99_latency_ms)
        .def_readonly("gpu_memory_used_mb",    &neuroswarm::PerformanceMetrics::gpu_memory_used_mb)
        .def_readonly("total_inferences",      &neuroswarm::PerformanceMetrics::total_inferences)
        .def_readonly("total_messages",        &neuroswarm::PerformanceMetrics::total_messages);

    // ── Task ─────────────────────────────────────────────
    py::class_<neuroswarm::Task>(m, "Task")
        .def(py::init<>())
        .def_readwrite("id",           &neuroswarm::Task::id)
        .def_readwrite("description",  &neuroswarm::Task::description)
        .def_readwrite("prompt",       &neuroswarm::Task::prompt)
        .def_readwrite("target_role",  &neuroswarm::Task::target_role)
        .def_readwrite("result",       &neuroswarm::Task::result)
        .def_readwrite("max_retries",  &neuroswarm::Task::max_retries);

    // ── Orchestrator ─────────────────────────────────────
    py::class_<neuroswarm::Orchestrator>(m, "Orchestrator")
        .def(py::init<const neuroswarm::OrchestratorConfig&>())
        .def("initialize",              &neuroswarm::Orchestrator::initialize)
        .def("start",                   &neuroswarm::Orchestrator::start)
        .def("stop",                    &neuroswarm::Orchestrator::stop)
        .def("shutdown",                &neuroswarm::Orchestrator::shutdown)
        .def("register_agent",          &neuroswarm::Orchestrator::register_agent)
        .def("unregister_agent",         &neuroswarm::Orchestrator::unregister_agent)
        .def("get_agent_state",          &neuroswarm::Orchestrator::get_agent_state)
        .def("get_agents_by_role",       &neuroswarm::Orchestrator::get_agents_by_role)
        .def("submit_task",              &neuroswarm::Orchestrator::submit_task)
        .def("cancel_task",              &neuroswarm::Orchestrator::cancel_task)
        .def("execute_reasoning_pipeline",
             &neuroswarm::Orchestrator::execute_reasoning_pipeline,
             py::arg("query"), py::arg("max_iterations") = 5)
        .def("kill_agent",              &neuroswarm::Orchestrator::kill_agent)
        .def("kill_all_agents",          &neuroswarm::Orchestrator::kill_all_agents)
        .def("get_metrics",              &neuroswarm::Orchestrator::get_metrics);

    // ── Utility functions ────────────────────────────────
    m.def("agent_role_str", &neuroswarm::agent_role_str, "Convert AgentRole to string");
    m.def("agent_state_str", &neuroswarm::agent_state_str, "Convert AgentState to string");

    // ── Version info ─────────────────────────────────────
    m.attr("__version__") = "1.0.0";
    m.attr("__cuda_support__") = true;
}
