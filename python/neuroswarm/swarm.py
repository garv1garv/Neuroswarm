import json
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.planner import PlannerAgent, ExecutionPlan
from agents.executor import ExecutorAgent, ExecutionResult
from agents.validator import ValidatorAgent, ValidationReport
from tools.regression_detector import RegressionDetector
from tools.bug_reporter import BugReporter

logger = logging.getLogger("neuroswarm")

@dataclass
class SwarmResult:
    """Complete result of a swarm reasoning pipeline."""
    query: str
    plan: Optional[ExecutionPlan] = None
    execution_results: List[ExecutionResult] = field(default_factory=list)
    validation: Optional[ValidationReport] = None
    final_answer: str = ""
    total_time_s: float = 0.0
    tokens_generated: int = 0
    num_agents_used: int = 0

    @property
    def success(self) -> bool:
        return self.validation.overall_pass if self.validation else False

    @property
    def report(self) -> str:
        lines = [
            "=" * 60,
            "  NeuroSwarm Reasoning Pipeline Report",
            "=" * 60,
            f"  Query: {self.query[:80]}",
            f"  Status: {'✅ SUCCESS' if self.success else '❌ FAILED'}",
            f"  Time: {self.total_time_s:.2f}s",
            f"  Agents used: {self.num_agents_used}",
            f"  Tokens: {self.tokens_generated}",
            "=" * 60,
        ]

        if self.plan:
            lines.extend([
                "",
                f"📋 Plan: {self.plan.title}",
                f"   Steps: {len(self.plan.steps)}",
                f"   Confidence: {self.plan.confidence:.0%}",
            ])

        if self.execution_results:
            passed = sum(1 for r in self.execution_results if r.success)
            lines.extend([
                "",
                f"⚡ Execution: {passed}/{len(self.execution_results)} steps passed",
            ])

        if self.validation:
            lines.extend([
                "",
                f"🔍 Validation: {self.validation.summary}",
            ])

        if self.final_answer:
            lines.extend([
                "",
                "📝 Answer:",
                self.final_answer[:500],
            ])

        return "\n".join(lines)

class NeuroSwarm:
    """
    High-level interface to the NeuroSwarm multi-agent system.

    Orchestrates planning, execution, and validation through
    GPU-accelerated agents.
    """

    def __init__(self, num_gpus: int = 1, max_parallel: int = 5):
        self.num_gpus = num_gpus
        self.max_parallel = max_parallel

        self.planner: Optional[PlannerAgent] = None
        self.executors: List[ExecutorAgent] = []
        self.validator: Optional[ValidatorAgent] = None

        self.regression_detector = RegressionDetector()
        self.bug_reporter = BugReporter()

        self._use_cpp_backend = False
        self._cpp_orchestrator = None

        logger.info(f"NeuroSwarm initialized: {num_gpus} GPUs, max_parallel={max_parallel}")

        # Try to load C++ backend
        try:
            import _neuroswarm_cpp as cpp
            self._use_cpp_backend = True
            logger.info("C++ backend loaded successfully")
        except ImportError:
            logger.info("C++ backend not available — using Python-only mode")

    def add_planner(self, name: str = "Planner", model: str = "llama-3.1-70b"):
        """Add a planner agent."""
        self.planner = PlannerAgent(model_name=model)
        logger.info(f"Planner agent '{name}' added (model={model})")

    def add_executors(self, count: int = 5, model: str = "llama-3.1-8b"):
        """Add multiple executor agents."""
        for i in range(count):
            executor = ExecutorAgent(agent_id=i + 1, model_name=model)
            self.executors.append(executor)
        logger.info(f"{count} executor agents added (model={model})")

    def add_validator(self, name: str = "Validator", model: str = "llama-3.1-8b"):
        """Add a validator agent."""
        self.validator = ValidatorAgent(model_name=model)
        logger.info(f"Validator agent '{name}' added (model={model})")

    def run(self, query: str, max_iterations: int = 5) -> SwarmResult:
        """
        Execute the full reasoning pipeline.

        1. Planner creates an execution plan
        2. Executors run plan steps (in parallel where possible)
        3. Validator checks all results

        Args:
            query: The task to accomplish
            max_iterations: Max planning iterations

        Returns:
            SwarmResult with plan, results, validation, and final answer
        """
        start_time = time.time()
        result = SwarmResult(query=query)

        logger.info(f"Starting pipeline: '{query[:80]}'")

        # Ensure we have agents
        if not self.planner:
            self.add_planner()
        if not self.executors:
            self.add_executors()
        if not self.validator:
            self.add_validator()

        # ── Step 1: Planning ──────────────────────────────
        logger.info("Phase 1: Planning")
        plan = self.planner.create_plan(query)
        result.plan = plan
        result.num_agents_used = 1
        logger.info(f"Plan created: {plan.title} ({len(plan.steps)} steps, "
                    f"confidence={plan.confidence:.0%})")

        # ── Step 2: Execution ─────────────────────────────
        logger.info("Phase 2: Execution")
        execution_results = self._execute_plan(plan)
        result.execution_results = execution_results
        result.num_agents_used += min(len(plan.steps), len(self.executors))

        # ── Step 3: Validation ────────────────────────────
        logger.info("Phase 3: Validation")
        result_dicts = [r.to_dict() for r in execution_results]
        validation = self.validator.validate(
            plan_id=plan.id,
            results=result_dicts,
            plan=plan.to_dict()
        )
        result.validation = validation
        result.num_agents_used += 1

        # ── Generate final answer ─────────────────────────
        result.final_answer = self._synthesize_answer(query, plan, execution_results, validation)

        # ── Check for regressions ─────────────────────────
        metrics = self._extract_metrics(execution_results)
        if metrics:
            alerts = self.regression_detector.check(metrics)
            for alert in alerts:
                self.bug_reporter.create_from_regression(
                    alert.metric, alert.baseline_value,
                    alert.current_value, alert.change_pct
                )

        result.total_time_s = time.time() - start_time
        logger.info(f"Pipeline complete in {result.total_time_s:.2f}s — "
                    f"{'SUCCESS' if result.success else 'FAILED'}")

        return result

    def _execute_plan(self, plan: ExecutionPlan) -> List[ExecutionResult]:
        """Execute plan steps, parallelizing where possible."""
        results = []
        completed_steps = set()

        # Build dependency graph → execution order
        steps_by_id = {s.id: s for s in plan.steps}
        remaining = list(plan.steps)

        while remaining:
            # Find steps whose dependencies are all met
            ready = [
                s for s in remaining
                if all(d in completed_steps for d in s.dependencies)
            ]

            if not ready:
                logger.warning("Deadlock detected in plan — forcing remaining steps")
                ready = remaining[:1]

            # Execute ready steps in parallel
            with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                futures = {}
                for step in ready:
                    executor = self.executors[len(results) % len(self.executors)]

                    # Build context from completed step results
                    context = {}
                    for r in results:
                        if r.step_id in step.dependencies:
                            context[r.step_id] = r.output

                    future = pool.submit(
                        executor.execute_step,
                        step.id, step.description, context
                    )
                    futures[future] = step

                for future in as_completed(futures):
                    step = futures[future]
                    try:
                        exec_result = future.result()
                        results.append(exec_result)
                        completed_steps.add(step.id)
                        status = "✅" if exec_result.success else "❌"
                        logger.info(f"  {status} Step '{step.id}' completed "
                                   f"in {exec_result.duration_s:.2f}s")
                    except Exception as e:
                        logger.error(f"  ❌ Step '{step.id}' exception: {e}")
                        results.append(ExecutionResult(
                            step_id=step.id, success=False,
                            output="", error=str(e)
                        ))
                        completed_steps.add(step.id)

            remaining = [s for s in remaining if s.id not in completed_steps]

        return results

    def _synthesize_answer(self, query: str, plan: ExecutionPlan,
                            results: List[ExecutionResult],
                            validation: ValidationReport) -> str:
        """Synthesize a final answer from all pipeline outputs."""
        parts = [f"Task: {query}", "", f"Plan: {plan.title}", ""]

        for r in results:
            status = "✓" if r.success else "✗"
            parts.append(f"[{status}] {r.step_id}: {r.output[:200]}")

        parts.extend(["", f"Validation: {validation.summary}"])

        if validation.recommendations:
            parts.extend(["", "Recommendations:"])
            for rec in validation.recommendations:
                parts.append(f"  • {rec}")

        return "\n".join(parts)

    def _extract_metrics(self, results: List[ExecutionResult]) -> Dict[str, float]:
        """Extract numeric metrics from execution results."""
        metrics = {}
        for r in results:
            try:
                data = json.loads(r.output) if isinstance(r.output, str) else r.output
                if isinstance(data, dict):
                    for key, val in data.items():
                        if isinstance(val, (int, float)):
                            metrics[key] = val
            except (json.JSONDecodeError, TypeError):
                pass
        return metrics

    def status(self) -> Dict[str, Any]:
        """Get current swarm status."""
        return {
            "planner": self.planner is not None,
            "executors": len(self.executors),
            "validator": self.validator is not None,
            "cpp_backend": self._use_cpp_backend,
            "num_gpus": self.num_gpus,
            "bugs_filed": len(self.bug_reporter.reports),
        }
