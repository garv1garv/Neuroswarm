__version__ = "1.0.0"
__author__ = "NeuroSwarm Contributors"

from .swarm import NeuroSwarm
from .config import SwarmConfig

# Re-export agent classes
from agents.planner import PlannerAgent, ExecutionPlan, PlanStep
from agents.executor import ExecutorAgent, ExecutionResult
from agents.validator import ValidatorAgent, ValidationReport

# Re-export tools
from tools.gpu_profiler import GpuProfiler
from tools.cuda_analyzer import CudaAnalyzer
from tools.regression_detector import RegressionDetector
from tools.bug_reporter import BugReporter
from tools.driver_api import DriverAPI

__all__ = [
    "NeuroSwarm",
    "SwarmConfig",
    "PlannerAgent",
    "ExecutorAgent",
    "ValidatorAgent",
    "ExecutionPlan",
    "PlanStep",
    "ExecutionResult",
    "ValidationReport",
    "GpuProfiler",
    "CudaAnalyzer",
    "RegressionDetector",
    "BugReporter",
    "DriverAPI",
]
