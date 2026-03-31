import argparse
import json
import time
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from agents.validator import ValidatorAgent
from tools.driver_api import DriverAPI
from tools.gpu_profiler import GpuProfiler
from tools.regression_detector import RegressionDetector
from tools.bug_reporter import BugReporter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("driver_testing")

class DriverTestingPipeline:
    """End-to-end automated GPU driver testing pipeline."""

    # Baselines for regression detection
    KNOWN_BASELINES = {
        "throughput_tps": 15000,
        "latency_p50_ms": 15.0,
        "latency_p99_ms": 50.0,
        "gpu_utilization": 0.90,
        "memory_peak_gb": 14.0,
    }

    def __init__(self, driver_version: str, gpu_ids: list):
        self.driver_version = driver_version
        self.gpu_ids = gpu_ids

        # Initialize agents
        self.planner = PlannerAgent()
        self.executors = [
            ExecutorAgent(agent_id=i + 1) for i in range(len(gpu_ids))
        ]
        self.validator = ValidatorAgent()

        # Initialize tools
        self.driver_api = DriverAPI()
        self.profiler = GpuProfiler()
        self.regression_detector = RegressionDetector()
        self.bug_reporter = BugReporter()

        # Set baselines
        self.regression_detector.set_baseline(self.KNOWN_BASELINES)
        self.validator.set_baselines(self.KNOWN_BASELINES)

    def run(self) -> dict:
        """Execute the full driver testing pipeline."""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"  NeuroSwarm Driver Testing Pipeline")
        logger.info(f"  Driver: {self.driver_version}")
        logger.info(f"  GPUs: {self.gpu_ids}")
        logger.info("=" * 60)

        results = {
            "driver_version": self.driver_version,
            "gpus": self.gpu_ids,
            "start_time": time.time(),
            "phases": {},
        }

        # ── Phase 1: Environment Check ───────────────────
        logger.info("\n[INFO] Phase 1: Environment Check")
        driver_info = self.driver_api.get_info()
        results["driver_info"] = driver_info
        logger.info(f"  Driver: {driver_info.get('driver_version', 'N/A')}")
        logger.info(f"  GPU: {driver_info.get('gpu', 'N/A')}")
        logger.info(f"  CUDA: {driver_info.get('cuda_version', 'N/A')}")

        # ── Phase 2: Planning ─────────────────────────────
        logger.info("\n[INFO] Phase 2: Creating Test Plan")
        query = (f"Test NVIDIA GPU driver version {self.driver_version} "
                 f"across {len(self.gpu_ids)} GPUs. "
                 f"Run unit tests, stress tests, and performance benchmarks.")
        plan = self.planner.create_plan(query)
        results["phases"]["planning"] = plan.to_dict()
        logger.info(f"  Plan: {plan.title}")
        logger.info(f"  Steps: {len(plan.steps)}")
        logger.info(f"  Confidence: {plan.confidence:.0%}")

        # ── Phase 3: Execution ────────────────────────────
        logger.info("\n[EXEC] Phase 3: Executing Test Plan")
        exec_results = []
        for i, step in enumerate(plan.steps):
            executor = self.executors[i % len(self.executors)]

            # Build context from previous steps
            context = {r.step_id: r.output for r in exec_results if r.success}

            logger.info(f"  [{i+1}/{len(plan.steps)}] {step.id}: {step.description[:60]}")
            result = executor.execute_step(step.id, step.description, context)
            exec_results.append(result)

            status = "[PASS]" if result.success else "[FAIL]"
            logger.info(f"    {status} Completed in {result.duration_s:.2f}s")

        results["phases"]["execution"] = [r.to_dict() for r in exec_results]

        # ── Phase 4: Validation ───────────────────────────
        logger.info("\n[VAL] Phase 4: Validating Results")
        validation = self.validator.validate(
            plan_id=plan.id,
            results=[r.to_dict() for r in exec_results],
            plan=plan.to_dict()
        )
        results["phases"]["validation"] = validation.to_dict()
        logger.info(f"  {validation.summary}")

        # ── Phase 5: Regression Detection ─────────────────
        logger.info("\n[REG] Phase 5: Regression Analysis")
        metrics = self._extract_metrics(exec_results)
        if metrics:
            alerts = self.regression_detector.check(metrics)
            results["regressions"] = [a.to_dict() for a in alerts]

            if alerts:
                logger.warning(f"  [WARN] {len(alerts)} regression(s) detected!")
                for alert in alerts:
                    logger.warning(f"    {alert.metric}: {alert.change_pct:+.1f}% "
                                  f"({alert.severity})")

                    # Auto-file bugs for significant regressions
                    self.bug_reporter.create_from_regression(
                        alert.metric, alert.baseline_value,
                        alert.current_value, alert.change_pct
                    )
            else:
                logger.info("  [PASS] No regressions detected")

        # ── Phase 6: Report Generation ────────────────────
        logger.info("\n[DOC] Phase 6: Generating Report")
        report_md = validation.to_markdown()
        report_path = f"/tmp/neuroswarm/reports/driver_{self.driver_version}_{int(time.time())}.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_md)
        logger.info(f"  Report saved to: {report_path}")

        # Bug summary
        if self.bug_reporter.reports:
            logger.info(f"\n[BUG] Bugs Filed: {len(self.bug_reporter.reports)}")
            logger.info(self.bug_reporter.summary())

        results["total_time_s"] = time.time() - start_time
        results["overall_pass"] = validation.overall_pass
        results["bugs_filed"] = len(self.bug_reporter.reports)

        # Final summary
        logger.info("\n" + "=" * 60)
        status = "PASSED" if validation.overall_pass else "FAILED"
        logger.info(f"  Driver {self.driver_version}: {status}")
        logger.info(f"  Time: {results['total_time_s']:.1f}s")
        logger.info(f"  Steps: {sum(1 for r in exec_results if r.success)}/{len(exec_results)} passed")
        logger.info(f"  Bugs: {results['bugs_filed']}")
        logger.info("=" * 60)

        return results

    def _extract_metrics(self, results):
        metrics = {}
        for r in results:
            try:
                data = json.loads(r.output) if isinstance(r.output, str) else r.output
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, (int, float)):
                            metrics[k] = v
            except (json.JSONDecodeError, TypeError):
                pass
        return metrics

def main():
    parser = argparse.ArgumentParser(description="NeuroSwarm GPU Driver Testing")
    parser.add_argument("--driver-version", default="560.35",
                        help="Driver version to test")
    parser.add_argument("--gpus", default="0",
                        help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    parser.add_argument("--output", default="/tmp/neuroswarm/results",
                        help="Output directory for results")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]

    pipeline = DriverTestingPipeline(
        driver_version=args.driver_version,
        gpu_ids=gpu_ids
    )

    results = pipeline.run()

    # Save full results JSON
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(
        args.output,
        f"driver_test_{args.driver_version}_{int(time.time())}.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    sys.exit(0 if results["overall_pass"] else 1)

if __name__ == "__main__":
    main()
