import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

@dataclass
class BugReport:
    """Structured bug report."""
    id: str
    title: str
    severity: str           # P0, P1, P2, P3
    component: str
    description: str
    reproduction_steps: List[str] = field(default_factory=list)
    expected_behavior: str = ""
    actual_behavior: str = ""
    environment: Dict[str, str] = field(default_factory=dict)
    logs: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    assignee: str = ""
    labels: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    attachments: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity,
            "component": self.component,
            "description": self.description,
            "reproduction_steps": self.reproduction_steps,
            "expected_behavior": self.expected_behavior,
            "actual_behavior": self.actual_behavior,
            "environment": self.environment,
            "metrics": self.metrics,
            "labels": self.labels,
            "created_at": self.created_at,
        }

    def to_markdown(self) -> str:
        """Generate markdown-formatted bug report."""
        lines = [
            f"# 🐛 {self.title}",
            "",
            f"**Severity:** {self.severity} | **Component:** {self.component}",
            f"**Bug ID:** `{self.id}` | **Created:** {time.strftime('%Y-%m-%d %H:%M', time.localtime(self.created_at))}",
            f"**Labels:** {', '.join(self.labels)}" if self.labels else "",
            "",
            "## Description",
            self.description,
            "",
        ]

        if self.reproduction_steps:
            lines.extend(["## Steps to Reproduce", ""])
            for i, step in enumerate(self.reproduction_steps, 1):
                lines.append(f"{i}. {step}")
            lines.append("")

        if self.expected_behavior:
            lines.extend(["## Expected Behavior", self.expected_behavior, ""])

        if self.actual_behavior:
            lines.extend(["## Actual Behavior", self.actual_behavior, ""])

        if self.environment:
            lines.extend(["## Environment", ""])
            for key, val in self.environment.items():
                lines.append(f"- **{key}:** {val}")
            lines.append("")

        if self.metrics:
            lines.extend(["## Metrics", "```json",
                          json.dumps(self.metrics, indent=2), "```", ""])

        if self.logs:
            lines.extend(["## Logs", "```", self.logs[:3000], "```", ""])

        return "\n".join(lines)

class BugReporter:
    """
    Automated bug report generator for NeuroSwarm.

    Features:
    - Generates structured reports from test failures and regressions
    - Deduplicates identical issues
    - Categorizes by severity and component
    - Provides reproduction steps
    - Exports to JSON, Markdown, and issue tracker formats
    """

    COMPONENT_MAP = {
        "attention": "cuda/attention_kernels",
        "memory": "cuda/memory_pool",
        "quantization": "cuda/quantization_kernels",
        "communication": "src/communication",
        "orchestrator": "src/orchestrator",
        "inference": "src/inference_engine",
        "agent": "agents/",
        "driver": "experiments/driver_testing",
    }

    def __init__(self, output_dir: str = "/tmp/neuroswarm/bugs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports: List[BugReport] = []
        self.known_hashes: set = set()

    def create_from_test_failure(self, test_name: str, error: str,
                                  logs: str = "",
                                  metrics: Optional[Dict] = None) -> Optional[BugReport]:
        """Create a bug report from a test failure."""
        # Generate dedup hash
        bug_hash = hashlib.md5(f"{test_name}:{error[:100]}".encode()).hexdigest()[:12]
        if bug_hash in self.known_hashes:
            return None  # Duplicate
        self.known_hashes.add(bug_hash)

        # Determine component and severity
        component = self._detect_component(test_name, error)
        severity = self._classify_severity(error, test_name)

        report = BugReport(
            id=f"NS-{bug_hash}",
            title=f"Test failure: {test_name}",
            severity=severity,
            component=component,
            description=f"Automated test `{test_name}` failed during NeuroSwarm CI pipeline.",
            reproduction_steps=[
                f"Build NeuroSwarm: `cmake --build build/`",
                f"Run test: `./build/neuroswarm_tests --gtest_filter={test_name}`",
                f"Observe failure: {error[:200]}",
            ],
            expected_behavior="Test should pass without errors",
            actual_behavior=error[:1000],
            environment=self._get_environment(),
            logs=logs,
            metrics=metrics or {},
            labels=["auto-filed", "test-failure", component.split("/")[0]],
        )

        self.reports.append(report)
        self._save_report(report)
        return report

    def create_from_regression(self, metric: str, baseline: float,
                                current: float, change_pct: float,
                                context: str = "") -> Optional[BugReport]:
        """Create a bug report from a performance regression."""
        bug_hash = hashlib.md5(f"regression:{metric}:{change_pct:.0f}".encode()).hexdigest()[:12]
        if bug_hash in self.known_hashes:
            return None
        self.known_hashes.add(bug_hash)

        severity = "P1" if abs(change_pct) > 20 else "P2" if abs(change_pct) > 10 else "P3"
        component = self._detect_component(metric, context)

        report = BugReport(
            id=f"NS-{bug_hash}",
            title=f"Performance regression: {metric} ({change_pct:+.1f}%)",
            severity=severity,
            component=component,
            description=(
                f"Performance regression detected in `{metric}`.\n\n"
                f"- **Baseline:** {baseline:.4f}\n"
                f"- **Current:** {current:.4f}\n"
                f"- **Change:** {change_pct:+.1f}%\n\n"
                f"{context}"
            ),
            reproduction_steps=[
                "Run benchmark suite: `./build/bench_attention`",
                f"Compare {metric} against baseline",
                f"Observed {change_pct:+.1f}% regression",
            ],
            expected_behavior=f"{metric} should be within 5% of baseline ({baseline:.4f})",
            actual_behavior=f"{metric} = {current:.4f} ({change_pct:+.1f}% from baseline)",
            environment=self._get_environment(),
            metrics={"baseline": baseline, "current": current, "change_pct": change_pct},
            labels=["auto-filed", "regression", "performance"],
        )

        self.reports.append(report)
        self._save_report(report)
        return report

    def summary(self) -> str:
        """Generate summary of all filed bugs."""
        if not self.reports:
            return "No bugs filed."

        lines = [
            f"Bug Report Summary ({len(self.reports)} total)",
            "=" * 50,
            "",
            "| ID | Severity | Component | Title |",
            "|-----|-------|-----------|-------|",
        ]
        for r in self.reports:
            lines.append(f"| {r.id} | {r.severity} | {r.component} | {r.title[:50]} |")

        by_severity = {}
        for r in self.reports:
            by_severity.setdefault(r.severity, 0)
            by_severity[r.severity] += 1

        lines.extend(["", "By severity:"])
        for sev in ["P0", "P1", "P2", "P3"]:
            count = by_severity.get(sev, 0)
            if count > 0:
                lines.append(f"  {sev}: {count}")

        return "\n".join(lines)

    def _detect_component(self, name: str, context: str = "") -> str:
        combined = f"{name} {context}".lower()
        for keyword, component in self.COMPONENT_MAP.items():
            if keyword in combined:
                return component
        return "neuroswarm/core"

    def _classify_severity(self, error: str, test_name: str) -> str:
        error_lower = error.lower()
        if any(w in error_lower for w in ["crash", "segfault", "abort", "oom", "corruption"]):
            return "P0"
        if any(w in error_lower for w in ["hang", "deadlock", "timeout", "memory leak"]):
            return "P1"
        if "assertion" in error_lower or "failed" in error_lower:
            return "P2"
        return "P3"

    def _get_environment(self) -> Dict[str, str]:
        import platform
        env = {
            "os": platform.system(),
            "arch": platform.machine(),
            "python": platform.python_version(),
        }
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                env["gpu"] = result.stdout.strip()
        except Exception:
            env["gpu"] = "N/A"
        return env

    def _save_report(self, report: BugReport):
        path = self.output_dir / f"{report.id}.md"
        path.write_text(report.to_markdown())

        json_path = self.output_dir / f"{report.id}.json"
        json_path.write_text(json.dumps(report.to_dict(), indent=2))
