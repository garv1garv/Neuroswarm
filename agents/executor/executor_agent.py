import json
import time
import subprocess
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path

@dataclass
class ExecutionResult:
    """Result of a single plan step execution."""
    step_id: str
    success: bool
    output: str
    error: Optional[str] = None
    duration_s: float = 0.0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)  # File paths produced
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "success": self.success,
            "output": self.output[:2000],  # Truncate for logging
            "error": self.error,
            "duration_s": self.duration_s,
            "num_tool_calls": len(self.tool_calls),
            "artifacts": self.artifacts,
            "metrics": self.metrics,
        }

class ExecutorAgent:
    """
    Worker agent in NeuroSwarm.

    Responsibilities:
    - Execute individual plan steps
    - Manage tool invocations
    - Handle errors and retries
    - Report structured results
    - Operate within sandbox constraints
    """

    SYSTEM_PROMPT = """You are an executor agent in NeuroSwarm. You receive a specific
task step from a plan and must execute it precisely.

Rules:
1. Execute the task step as described — do not deviate
2. Use available tools when needed
3. Report results clearly
4. If you encounter an error, report it with full details
5. Do not attempt to modify the plan — that's the planner's job
6. Provide structured output with metrics when possible

Available tools: {tools}

When you need a tool, respond with:
Action: tool_name(arguments)

When you have the final result:
FINAL ANSWER: <your structured result>"""

    def __init__(self, agent_id: int, model_name: str = "llama-3.1-8b"):
        self.agent_id = agent_id
        self.model_name = model_name
        self.tools: Dict[str, Callable] = {}
        self.tool_descriptions: Dict[str, str] = {}
        self.execution_history: List[ExecutionResult] = []

        # Register default tools
        self._register_default_tools()

    def register_tool(self, name: str, description: str, func: Callable):
        """Register a tool that this agent can use."""
        self.tools[name] = func
        self.tool_descriptions[name] = description

    def execute_step(self, step_id: str, description: str,
                     context: Optional[Dict] = None) -> ExecutionResult:
        """
        Execute a single plan step.

        Args:
            step_id: Unique identifier for this step
            description: Detailed description of what to do
            context: Optional context from previous steps

        Returns:
            ExecutionResult with outcome details
        """
        start_time = time.time()
        tool_calls = []

        try:
            # Build execution prompt
            tools_str = "\n".join(
                f"  - {name}: {desc}"
                for name, desc in self.tool_descriptions.items()
            )
            prompt = self.SYSTEM_PROMPT.format(tools=tools_str)
            prompt += f"\n\n## Task Step: {step_id}\n{description}"

            if context:
                prompt += f"\n\n## Context from previous steps:\n{json.dumps(context, indent=2)}"

            # In production, this would run through the inference engine
            # Simulate execution based on step type
            output = self._simulate_execution(step_id, description, tool_calls)

            result = ExecutionResult(
                step_id=step_id,
                success=True,
                output=output,
                duration_s=time.time() - start_time,
                tool_calls=tool_calls,
            )

        except Exception as e:
            result = ExecutionResult(
                step_id=step_id,
                success=False,
                output="",
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                duration_s=time.time() - start_time,
                tool_calls=tool_calls,
            )

        self.execution_history.append(result)
        return result

    def _simulate_execution(self, step_id: str, description: str,
                            tool_calls: list) -> str:
        """Simulate step execution (used when inference engine is not available)."""
        desc_lower = description.lower()

        if "test" in desc_lower or "run" in desc_lower:
            # Simulate test execution
            call = {"tool": "run_tests", "args": description, "success": True}
            tool_calls.append(call)
            return json.dumps({
                "tests_run": 42,
                "passed": 40,
                "failed": 2,
                "skipped": 0,
                "duration_s": 125.3,
                "failures": [
                    {"test": "test_memory_leak_async", "reason": "Timeout after 30s"},
                    {"test": "test_multi_gpu_sync", "reason": "P2P access denied on GPU 3"},
                ],
            })

        elif "benchmark" in desc_lower or "performance" in desc_lower:
            return json.dumps({
                "throughput_tps": 15420,
                "latency_p50_ms": 12.3,
                "latency_p99_ms": 45.7,
                "gpu_utilization": 0.94,
                "memory_peak_gb": 14.2,
            })

        elif "analyze" in desc_lower:
            return json.dumps({
                "summary": "Analysis complete",
                "findings": [
                    "Performance regression of 8% in attention kernel on Hopper",
                    "Memory leak in async copy path (2MB/hour)",
                    "P2P access failure on 4-GPU topology",
                ],
                "severity": {"critical": 1, "warning": 2, "info": 5},
            })

        elif "report" in desc_lower:
            return json.dumps({
                "report_generated": True,
                "path": "/tmp/neuroswarm/reports/driver_test_report.md",
                "sections": ["summary", "test_results", "regressions", "recommendations"],
            })

        else:
            return f"Step '{step_id}' executed successfully: {description}"

    def _register_default_tools(self):
        """Register built-in tools."""
        self.register_tool(
            "run_command",
            "Execute a shell command and return output",
            self._tool_run_command
        )
        self.register_tool(
            "read_file",
            "Read contents of a file",
            self._tool_read_file
        )
        self.register_tool(
            "write_file",
            "Write content to a file",
            self._tool_write_file
        )
        self.register_tool(
            "gpu_info",
            "Get GPU information via nvidia-smi",
            self._tool_gpu_info
        )

    def _tool_run_command(self, command: str) -> str:
        """Execute a shell command with timeout."""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True,
                text=True, timeout=60
            )
            return json.dumps({
                "returncode": result.returncode,
                "stdout": result.stdout[:5000],
                "stderr": result.stderr[:2000],
            })
        except subprocess.TimeoutExpired:
            return json.dumps({"error": "Command timed out after 60s"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_read_file(self, path: str) -> str:
        """Read a file's contents."""
        try:
            content = Path(path).read_text()
            return content[:10000]
        except Exception as e:
            return f"Error reading file: {e}"

    def _tool_write_file(self, args: str) -> str:
        """Write content to a file. Args: 'path|content'"""
        try:
            parts = args.split("|", 1)
            if len(parts) != 2:
                return "Error: Expected 'path|content'"
            path, content = parts
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content)
            return f"Written {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def _tool_gpu_info(self, _: str = "") -> str:
        """Get GPU information."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else "nvidia-smi not available"
        except Exception:
            return "nvidia-smi not available"
