from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class SwarmConfig:
    name: str = "NeuroSwarm"
    num_gpus: int = 1
    max_agents: int = 16
    max_parallel_tasks: int = 8
    planner_model: str = "llama-3.1-70b"
    executor_model: str = "llama-3.1-8b"
    validator_model: str = "llama-3.1-8b"
    num_executors: int = 5
    quant_mode: str = "fp16"
    enable_profiling: bool = False
    enable_safety: bool = True
    enable_hallucination_detection: bool = True
    metrics_port: int = 9090
    redis_url: str = "redis://localhost:6379"
    log_level: str = "INFO"
    output_dir: str = "/tmp/neuroswarm"
    baselines_path: str = "baselines.json"

    # Resource limits
    max_gpu_memory_per_agent_mb: int = 8192
    max_tokens_per_request: int = 4096
    timeout_per_step_s: int = 300

    # Circuit breaker
    cb_failure_threshold: int = 5
    cb_recovery_timeout_s: int = 30
