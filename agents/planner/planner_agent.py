import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

class PlanNodeType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"

@dataclass
class PlanStep:
    """A single step in an execution plan."""
    id: str
    description: str
    agent_role: str = "executor"
    dependencies: List[str] = field(default_factory=list)
    estimated_duration_s: float = 30.0
    priority: int = 5  # 1 = highest
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "backoff_ms": 1000,
        "timeout_ms": 60000
    })
    validation_criteria: str = ""
    rollback_action: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionPlan:
    """Complete execution plan with steps and metadata."""
    id: str
    title: str
    objective: str
    steps: List[PlanStep] = field(default_factory=list)
    node_type: PlanNodeType = PlanNodeType.SEQUENTIAL
    created_at: float = field(default_factory=time.time)
    estimated_total_s: float = 0.0
    confidence: float = 0.0
    alternative_plans: List['ExecutionPlan'] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "objective": self.objective,
            "node_type": self.node_type.value,
            "steps": [
                {
                    "id": s.id,
                    "description": s.description,
                    "agent_role": s.agent_role,
                    "dependencies": s.dependencies,
                    "estimated_duration_s": s.estimated_duration_s,
                    "priority": s.priority,
                    "validation_criteria": s.validation_criteria,
                }
                for s in self.steps
            ],
            "estimated_total_s": self.estimated_total_s,
            "confidence": self.confidence,
            "num_alternatives": len(self.alternative_plans),
        }

    def to_prompt(self) -> str:
        """Convert plan to a formatted prompt for executor agents."""
        lines = [f"# Execution Plan: {self.title}", f"Objective: {self.objective}", ""]
        for i, step in enumerate(self.steps, 1):
            deps = f" (after: {', '.join(step.dependencies)})" if step.dependencies else ""
            lines.append(f"Step {i} [{step.id}]{deps}:")
            lines.append(f"  {step.description}")
            if step.validation_criteria:
                lines.append(f"  Validate: {step.validation_criteria}")
            lines.append("")
        return "\n".join(lines)

class PlannerAgent:
    """
    Strategic planning agent for NeuroSwarm.

    Responsibilities:
    - Decompose complex queries into step-by-step plans
    - Identify parallelizable steps
    - Estimate resource requirements
    - Generate validation criteria for each step
    - Produce rollback strategies
    """

    SYSTEM_PROMPT = """You are a strategic planning agent in NeuroSwarm, a GPU-native
multi-agent reasoning system. Your role is to decompose complex tasks into precise,
actionable execution plans.

For each plan, you must:
1. Break the task into discrete, verifiable steps
2. Identify dependencies between steps
3. Mark steps that can execute in parallel
4. Assign appropriate agent roles (executor, validator)
5. Define validation criteria for each step
6. Estimate execution time per step
7. Specify rollback actions for critical steps

Output format: JSON with the following structure:
{
  "title": "Plan title",
  "objective": "What this plan achieves",
  "steps": [
    {
      "id": "step_1",
      "description": "Detailed action description",
      "agent_role": "executor",
      "dependencies": [],
      "estimated_duration_s": 30,
      "priority": 1,
      "validation_criteria": "How to verify success",
      "rollback_action": "What to do on failure"
    }
  ],
  "confidence": 0.85
}

Be thorough but practical. Prefer parallelism where safe."""

    def __init__(self, model_name: str = "llama-3.1-70b"):
        self.model_name = model_name
        self.plan_history: List[ExecutionPlan] = []

    def create_plan(self, query: str, context: Optional[Dict] = None) -> ExecutionPlan:
        """
        Generate an execution plan for a given query.

        Args:
            query: The task description
            context: Optional context (previous results, constraints, etc.)

        Returns:
            An ExecutionPlan ready for execution
        """
        # Build prompt
        prompt_parts = [self.SYSTEM_PROMPT, f"\n## Task\n{query}"]

        if context:
            prompt_parts.append(f"\n## Context\n{json.dumps(context, indent=2)}")

        prompt = "\n".join(prompt_parts)

        # In production, this calls the inference engine
        # For now, return a structured plan based on the query
        plan = self._generate_plan_structure(query)
        self.plan_history.append(plan)

        return plan

    def refine_plan(self, plan: ExecutionPlan, feedback: str) -> ExecutionPlan:
        """Refine an existing plan based on feedback."""
        prompt = f"""Previous plan:
{json.dumps(plan.to_dict(), indent=2)}

Feedback: {feedback}

Generate an improved plan addressing the feedback."""

        refined = self._generate_plan_structure(
            f"{plan.objective} (refined: {feedback})")
        refined.id = f"{plan.id}_refined"
        plan.alternative_plans.append(refined)

        return refined

    def _generate_plan_structure(self, query: str) -> ExecutionPlan:
        """Generate a structured plan (simulation for when no LLM is available)."""
        plan_id = f"plan_{int(time.time())}"

        # Analyze query to determine plan structure
        if "test" in query.lower() or "driver" in query.lower():
            return self._driver_testing_plan(plan_id, query)
        elif "benchmark" in query.lower() or "performance" in query.lower():
            return self._benchmark_plan(plan_id, query)
        else:
            return self._generic_plan(plan_id, query)

    def _driver_testing_plan(self, plan_id: str, query: str) -> ExecutionPlan:
        """Generate a plan for GPU driver testing."""
        plan = ExecutionPlan(
            id=plan_id,
            title="GPU Driver Testing Pipeline",
            objective=query,
            confidence=0.92,
        )

        plan.steps = [
            PlanStep(
                id="setup_env",
                description="Set up isolated test environment with target driver version",
                priority=1,
                estimated_duration_s=120,
                validation_criteria="Test VM is running with correct driver version",
                rollback_action="Restore previous driver version",
            ),
            PlanStep(
                id="run_unit_tests",
                description="Execute CUDA unit test suite across target GPU architectures",
                agent_role="executor",
                dependencies=["setup_env"],
                priority=2,
                estimated_duration_s=300,
                validation_criteria="All unit tests pass or failures are documented",
            ),
            PlanStep(
                id="run_stress_tests",
                description="Run GPU stress tests (memory, compute, mixed workloads)",
                agent_role="executor",
                dependencies=["setup_env"],
                priority=2,
                estimated_duration_s=600,
                validation_criteria="No GPU hangs, OOM errors, or crashes detected",
            ),
            PlanStep(
                id="run_regression_suite",
                description="Execute performance regression benchmarks",
                agent_role="executor",
                dependencies=["setup_env"],
                priority=3,
                estimated_duration_s=900,
                validation_criteria="Performance within 5% of baseline",
            ),
            PlanStep(
                id="analyze_results",
                description="Aggregate test results and identify regressions",
                agent_role="validator",
                dependencies=["run_unit_tests", "run_stress_tests", "run_regression_suite"],
                priority=4,
                estimated_duration_s=60,
                validation_criteria="All results analyzed and categorized",
            ),
            PlanStep(
                id="generate_report",
                description="Generate comprehensive test report with recommendations",
                agent_role="validator",
                dependencies=["analyze_results"],
                priority=5,
                estimated_duration_s=30,
                validation_criteria="Report contains pass/fail summary and regression list",
            ),
        ]

        plan.node_type = PlanNodeType.SEQUENTIAL
        plan.estimated_total_s = sum(s.estimated_duration_s for s in plan.steps)

        return plan

    def _benchmark_plan(self, plan_id: str, query: str) -> ExecutionPlan:
        """Generate a plan for performance benchmarking."""
        plan = ExecutionPlan(
            id=plan_id,
            title="Performance Benchmark Suite",
            objective=query,
            confidence=0.88,
        )

        plan.steps = [
            PlanStep(id="warmup", description="GPU warmup and baseline measurement",
                     priority=1, estimated_duration_s=30),
            PlanStep(id="bench_attention", description="Benchmark attention kernels",
                     dependencies=["warmup"], priority=2, estimated_duration_s=120),
            PlanStep(id="bench_quant", description="Benchmark quantization throughput",
                     dependencies=["warmup"], priority=2, estimated_duration_s=120),
            PlanStep(id="bench_comm", description="Benchmark inter-agent communication",
                     dependencies=["warmup"], priority=2, estimated_duration_s=60),
            PlanStep(id="bench_e2e", description="End-to-end inference benchmark",
                     dependencies=["bench_attention", "bench_quant"],
                     priority=3, estimated_duration_s=300),
            PlanStep(id="report", description="Compile benchmark results and graphs",
                     agent_role="validator",
                     dependencies=["bench_e2e", "bench_comm"],
                     priority=4, estimated_duration_s=30),
        ]

        plan.estimated_total_s = sum(s.estimated_duration_s for s in plan.steps)
        return plan

    def _generic_plan(self, plan_id: str, query: str) -> ExecutionPlan:
        """Generate a generic execution plan."""
        plan = ExecutionPlan(
            id=plan_id,
            title=f"Execute: {query[:60]}",
            objective=query,
            confidence=0.75,
        )

        plan.steps = [
            PlanStep(id="analyze", description=f"Analyze requirements: {query}",
                     priority=1, estimated_duration_s=30),
            PlanStep(id="execute", description="Execute primary task",
                     dependencies=["analyze"], priority=2,
                     estimated_duration_s=120),
            PlanStep(id="validate", description="Validate results",
                     agent_role="validator",
                     dependencies=["execute"], priority=3,
                     estimated_duration_s=30),
        ]

        plan.estimated_total_s = sum(s.estimated_duration_s for s in plan.steps)
        return plan
