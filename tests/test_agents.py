import json
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.planner import PlannerAgent, ExecutionPlan, PlanStep, PlanNodeType
from agents.executor import ExecutorAgent, ExecutionResult
from agents.validator import ValidatorAgent, ValidationReport, ValidationSeverity

class TestPlannerAgent(unittest.TestCase):
    """Tests for the Planner agent."""

    def setUp(self):
        self.planner = PlannerAgent()

    def test_create_plan_basic(self):
        plan = self.planner.create_plan("Summarize the project status")
        self.assertIsInstance(plan, ExecutionPlan)
        self.assertTrue(len(plan.steps) > 0)
        self.assertTrue(len(plan.id) > 0)

    def test_create_plan_driver_testing(self):
        plan = self.planner.create_plan("Test NVIDIA driver 560.35")
        self.assertEqual(plan.title, "GPU Driver Testing Pipeline")
        self.assertTrue(len(plan.steps) >= 5)
        self.assertGreater(plan.confidence, 0.8)

    def test_create_plan_benchmark(self):
        plan = self.planner.create_plan("Run performance benchmark suite")
        self.assertIn("Benchmark", plan.title)
        self.assertTrue(any(s.id == "bench_attention" for s in plan.steps))

    def test_plan_dependencies(self):
        plan = self.planner.create_plan("Test GPU driver")
        # Validate dependency graph — no forward references
        completed = set()
        for step in plan.steps:
            for dep in step.dependencies:
                self.assertIn(dep, completed,
                              f"Step {step.id} depends on {dep} which hasn't completed yet")
            completed.add(step.id)

    def test_plan_to_dict(self):
        plan = self.planner.create_plan("test")
        d = plan.to_dict()
        self.assertIn("id", d)
        self.assertIn("steps", d)
        self.assertIn("confidence", d)
        self.assertIsInstance(d["steps"], list)

    def test_plan_to_prompt(self):
        plan = self.planner.create_plan("test something")
        prompt = plan.to_prompt()
        self.assertIn("Execution Plan", prompt)
        self.assertIn("Step 1", prompt)

    def test_plan_history(self):
        self.planner.create_plan("task 1")
        self.planner.create_plan("task 2")
        self.assertEqual(len(self.planner.plan_history), 2)

    def test_refine_plan(self):
        plan = self.planner.create_plan("test something")
        refined = self.planner.refine_plan(plan, "Add more stress tests")
        self.assertIn("refined", refined.id)
        self.assertEqual(len(plan.alternative_plans), 1)

class TestExecutorAgent(unittest.TestCase):
    """Tests for the Executor agent."""

    def setUp(self):
        self.executor = ExecutorAgent(agent_id=1)

    def test_execute_test_step(self):
        result = self.executor.execute_step(
            "run_tests", "Run the unit test suite"
        )
        self.assertIsInstance(result, ExecutionResult)
        self.assertTrue(result.success)
        self.assertGreater(result.duration_s, 0)

        # Parse output
        data = json.loads(result.output)
        self.assertIn("tests_run", data)
        self.assertIn("passed", data)

    def test_execute_benchmark_step(self):
        result = self.executor.execute_step(
            "bench", "Run performance benchmark"
        )
        self.assertTrue(result.success)
        data = json.loads(result.output)
        self.assertIn("throughput_tps", data)

    def test_execute_analysis_step(self):
        result = self.executor.execute_step(
            "analyze", "Analyze the results"
        )
        self.assertTrue(result.success)
        data = json.loads(result.output)
        self.assertIn("findings", data)

    def test_tool_registration(self):
        self.executor.register_tool(
            "custom_tool", "A custom tool",
            lambda args: f"result: {args}"
        )
        self.assertIn("custom_tool", self.executor.tools)
        self.executor.unregister_tool("custom_tool")

    def test_default_tools_registered(self):
        self.assertIn("run_command", self.executor.tools)
        self.assertIn("read_file", self.executor.tools)
        self.assertIn("gpu_info", self.executor.tools)

    def test_execution_history(self):
        self.executor.execute_step("s1", "step 1")
        self.executor.execute_step("s2", "step 2")
        self.assertEqual(len(self.executor.execution_history), 2)

    def test_result_to_dict(self):
        result = self.executor.execute_step("s1", "test step")
        d = result.to_dict()
        self.assertIn("step_id", d)
        self.assertIn("success", d)
        self.assertIn("duration_s", d)

class TestValidatorAgent(unittest.TestCase):
    """Tests for the Validator agent."""

    def setUp(self):
        self.validator = ValidatorAgent()

    def test_validate_success(self):
        results = [
            {"step_id": "s1", "success": True, "output": '{"key": "value"}'},
            {"step_id": "s2", "success": True, "output": '{"result": 42}'},
        ]
        report = self.validator.validate("plan_1", results)
        self.assertIsInstance(report, ValidationReport)
        self.assertTrue(report.overall_pass)
        self.assertGreater(report.confidence_score, 0)

    def test_validate_failure(self):
        results = [
            {"step_id": "s1", "success": False, "output": "",
             "error": "CUDA OOM: out of memory"},
        ]
        report = self.validator.validate("plan_2", results)
        self.assertFalse(report.overall_pass)
        self.assertTrue(report.num_failures > 0)

    def test_validate_empty(self):
        report = self.validator.validate("plan_3", [])
        self.assertFalse(report.overall_pass)
        self.assertTrue(report.num_critical > 0)

    def test_regression_detection(self):
        self.validator.set_baselines({"throughput_tps": 15000})
        results = [
            {"step_id": "bench", "success": True,
             "output": json.dumps({"throughput_tps": 10000})},
        ]
        report = self.validator.validate("plan_4", results)
        self.assertTrue(report.regression_detected)
        self.assertTrue(len(report.regressions) > 0)

    def test_safety_check(self):
        results = [
            {"step_id": "s1", "success": True,
             "output": "Executed: rm -rf /important/data"},
        ]
        report = self.validator.validate("plan_5", results)
        critical = [c for c in report.checks
                    if c.severity == ValidationSeverity.CRITICAL]
        self.assertTrue(len(critical) > 0)

    def test_self_consistency(self):
        responses = [
            '{"result": 42, "status": "ok"}',
            '{"result": 43, "status": "ok"}',
            '{"result": 42, "status": "ok"}',
        ]
        is_consistent, ratio = self.validator.self_consistency_check(responses)
        self.assertIsInstance(is_consistent, bool)
        self.assertGreater(ratio, 0)

    def test_report_to_markdown(self):
        results = [{"step_id": "s1", "success": True, "output": "{}"}]
        report = self.validator.validate("plan_6", results)
        md = report.to_markdown()
        self.assertIn("Validation Report", md)
        self.assertIn("plan_6", md)

    def test_plan_coverage(self):
        plan = {
            "steps": [
                {"id": "s1"}, {"id": "s2"}, {"id": "s3"},
            ]
        }
        results = [
            {"step_id": "s1", "success": True, "output": "ok"},
            # Missing s2 and s3
        ]
        report = self.validator.validate("plan_7", results, plan)
        coverage_checks = [c for c in report.checks if "coverage" in c.name]
        self.assertTrue(len(coverage_checks) > 0)

class TestEndToEnd(unittest.TestCase):
    """Integration test: planner → executor → validator pipeline."""

    def test_full_pipeline(self):
        planner = PlannerAgent()
        executors = [ExecutorAgent(i + 1) for i in range(3)]
        validator = ValidatorAgent()
        validator.set_baselines({"throughput_tps": 15000})

        # Plan
        plan = planner.create_plan("Run performance tests on GPU driver")
        self.assertTrue(len(plan.steps) > 0)

        # Execute
        results = []
        for i, step in enumerate(plan.steps):
            executor = executors[i % len(executors)]
            result = executor.execute_step(step.id, step.description)
            results.append(result)

        # Validate
        report = validator.validate(
            plan.id,
            [r.to_dict() for r in results],
            plan.to_dict()
        )

        self.assertIsInstance(report, ValidationReport)
        self.assertTrue(len(report.checks) > 0)
        self.assertGreater(report.confidence_score, 0)

if __name__ == "__main__":
    unittest.main()
