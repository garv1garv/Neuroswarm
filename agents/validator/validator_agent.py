import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

class ValidationSeverity(Enum):
    PASS = "pass"
    WARNING = "warning"
    FAILURE = "failure"
    CRITICAL = "critical"

@dataclass
class ValidationCheck:
    """A single validation check result."""
    name: str
    severity: ValidationSeverity
    message: str
    details: Optional[str] = None
    auto_fix_available: bool = False
    fix_suggestion: str = ""

@dataclass
class ValidationReport:
    """Complete validation report for a set of execution results."""
    plan_id: str
    timestamp: float = field(default_factory=time.time)
    checks: List[ValidationCheck] = field(default_factory=list)
    overall_pass: bool = True
    confidence_score: float = 1.0
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    regression_detected: bool = False
    regressions: List[Dict[str, Any]] = field(default_factory=list)
    bugs_filed: List[Dict[str, str]] = field(default_factory=list)

    @property
    def num_passed(self) -> int:
        return sum(1 for c in self.checks if c.severity == ValidationSeverity.PASS)

    @property
    def num_warnings(self) -> int:
        return sum(1 for c in self.checks if c.severity == ValidationSeverity.WARNING)

    @property
    def num_failures(self) -> int:
        return sum(1 for c in self.checks if c.severity == ValidationSeverity.FAILURE)

    @property
    def num_critical(self) -> int:
        return sum(1 for c in self.checks if c.severity == ValidationSeverity.CRITICAL)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "overall_pass": self.overall_pass,
            "confidence_score": self.confidence_score,
            "summary": self.summary,
            "checks": {
                "passed": self.num_passed,
                "warnings": self.num_warnings,
                "failures": self.num_failures,
                "critical": self.num_critical,
            },
            "regression_detected": self.regression_detected,
            "num_regressions": len(self.regressions),
            "num_bugs_filed": len(self.bugs_filed),
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        """Generate a markdown report."""
        lines = [
            f"# Validation Report: {self.plan_id}",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            "",
            f"## Summary",
            f"**Overall: {'✅ PASS' if self.overall_pass else '❌ FAIL'}** "
            f"(Confidence: {self.confidence_score:.0%})",
            "",
            f"| Status | Count |",
            f"|--------|-------|",
            f"| ✅ Passed | {self.num_passed} |",
            f"| ⚠️ Warnings | {self.num_warnings} |",
            f"| ❌ Failures | {self.num_failures} |",
            f"| 🔥 Critical | {self.num_critical} |",
            "",
        ]

        if self.regressions:
            lines.extend([
                "## Performance Regressions",
                "",
                "| Metric | Baseline | Current | Change |",
                "|--------|----------|---------|--------|",
            ])
            for r in self.regressions:
                lines.append(
                    f"| {r['metric']} | {r['baseline']} | {r['current']} | {r['change']} |"
                )
            lines.append("")

        if self.recommendations:
            lines.extend(["## Recommendations", ""])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        if self.checks:
            lines.extend(["## Detailed Checks", ""])
            for check in self.checks:
                icon = {"pass": "✅", "warning": "⚠️",
                        "failure": "❌", "critical": "🔥"}[check.severity.value]
                lines.append(f"- {icon} **{check.name}**: {check.message}")
                if check.details:
                    lines.append(f"  - Details: {check.details}")

        return "\n".join(lines)

class ValidatorAgent:
    """
    Validation agent for NeuroSwarm.

    Responsibilities:
    - Verify execution results are correct and complete
    - Detect hallucinations via self-consistency
    - Identify performance regressions
    - Generate bug reports for failures
    - Produce validation reports
    """

    SYSTEM_PROMPT = """You are a validation agent in NeuroSwarm. You verify that
execution results are correct, complete, and consistent.

For each result, check:
1. Completeness — all required outputs present
2. Correctness — data is valid and within expected ranges
3. Consistency — results agree across multiple agents
4. Performance — no regressions compared to baselines
5. Safety — no dangerous actions were taken

Output a validation report with pass/fail for each check."""

    def __init__(self, model_name: str = "llama-3.1-8b"):
        self.model_name = model_name
        self.baselines: Dict[str, float] = {}
        self.validation_history: List[ValidationReport] = []

    def set_baselines(self, baselines: Dict[str, float]):
        """Set performance baselines for regression detection."""
        self.baselines = baselines

    def validate(self, plan_id: str,
                 results: List[Dict[str, Any]],
                 plan: Optional[Dict] = None) -> ValidationReport:
        """
        Validate a set of execution results.

        Args:
            plan_id: ID of the execution plan
            results: List of execution result dicts
            plan: Optional original plan for comparison

        Returns:
            ValidationReport with detailed findings
        """
        report = ValidationReport(plan_id=plan_id)

        # Run all validation checks
        self._check_completeness(report, results, plan)
        self._check_correctness(report, results)
        self._check_consistency(report, results)
        self._check_performance(report, results)
        self._check_safety(report, results)

        # Determine overall status
        report.overall_pass = (report.num_critical == 0 and report.num_failures == 0)
        report.confidence_score = self._calculate_confidence(report)

        # Generate summary
        report.summary = self._generate_summary(report)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        # File bugs for critical issues
        if report.num_critical > 0 or report.regression_detected:
            report.bugs_filed = self._generate_bug_reports(report)

        self.validation_history.append(report)
        return report

    def self_consistency_check(self, responses: List[str],
                                threshold: float = 0.7) -> Tuple[bool, float]:
        """
        Check if multiple agent responses are consistent.

        Args:
            responses: List of responses from different agents
            threshold: Minimum agreement ratio to pass

        Returns:
            (is_consistent, agreement_ratio)
        """
        if len(responses) < 2:
            return True, 1.0

        # Simple: check pairwise similarity using hash-based comparison
        # In production, use embedding similarity
        hashes = [hashlib.md5(r.encode()).hexdigest()[:8] for r in responses]

        # Count agreements
        agreements = 0
        comparisons = 0
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                comparisons += 1
                # Check structural similarity (same JSON keys, similar values)
                try:
                    d1 = json.loads(responses[i])
                    d2 = json.loads(responses[j])
                    if set(d1.keys()) == set(d2.keys()):
                        agreements += 1
                except (json.JSONDecodeError, AttributeError):
                    # Fall back to text similarity
                    words1 = set(responses[i].lower().split())
                    words2 = set(responses[j].lower().split())
                    if len(words1 & words2) / max(len(words1 | words2), 1) > 0.5:
                        agreements += 1

        ratio = agreements / max(comparisons, 1)
        return ratio >= threshold, ratio

    def _check_completeness(self, report: ValidationReport,
                             results: List[Dict], plan: Optional[Dict]):
        """Check that all expected outputs are present."""
        if not results:
            report.checks.append(ValidationCheck(
                name="completeness",
                severity=ValidationSeverity.CRITICAL,
                message="No results received"
            ))
            return

        # Check each result has required fields
        required_fields = ["step_id", "success", "output"]
        for result in results:
            missing = [f for f in required_fields if f not in result]
            if missing:
                report.checks.append(ValidationCheck(
                    name=f"completeness_{result.get('step_id', '?')}",
                    severity=ValidationSeverity.FAILURE,
                    message=f"Missing fields: {missing}"
                ))
            else:
                report.checks.append(ValidationCheck(
                    name=f"completeness_{result['step_id']}",
                    severity=ValidationSeverity.PASS,
                    message="All required fields present"
                ))

        # Check plan step coverage
        if plan and "steps" in plan:
            expected_ids = {s["id"] for s in plan["steps"]}
            actual_ids = {r.get("step_id") for r in results}
            missing_steps = expected_ids - actual_ids
            if missing_steps:
                report.checks.append(ValidationCheck(
                    name="plan_coverage",
                    severity=ValidationSeverity.FAILURE,
                    message=f"Missing steps: {missing_steps}"
                ))

    def _check_correctness(self, report: ValidationReport,
                            results: List[Dict]):
        """Check that results contain valid data."""
        for result in results:
            step_id = result.get("step_id", "unknown")

            if not result.get("success", False):
                report.checks.append(ValidationCheck(
                    name=f"correctness_{step_id}",
                    severity=ValidationSeverity.FAILURE,
                    message=f"Step failed: {result.get('error', 'Unknown error')}",
                    details=result.get("error")
                ))
                continue

            # Validate output is parseable
            output = result.get("output", "")
            try:
                data = json.loads(output) if isinstance(output, str) else output
                report.checks.append(ValidationCheck(
                    name=f"correctness_{step_id}",
                    severity=ValidationSeverity.PASS,
                    message="Output is valid structured data"
                ))
            except (json.JSONDecodeError, TypeError):
                report.checks.append(ValidationCheck(
                    name=f"correctness_{step_id}",
                    severity=ValidationSeverity.WARNING,
                    message="Output is not structured JSON",
                    details=output[:200]
                ))

    def _check_consistency(self, report: ValidationReport,
                            results: List[Dict]):
        """Check cross-result consistency."""
        outputs = [r.get("output", "") for r in results if r.get("success")]
        if len(outputs) >= 2:
            is_consistent, ratio = self.self_consistency_check(outputs)
            report.checks.append(ValidationCheck(
                name="self_consistency",
                severity=ValidationSeverity.PASS if is_consistent else ValidationSeverity.WARNING,
                message=f"Agreement ratio: {ratio:.0%}",
                details=f"Checked {len(outputs)} responses"
            ))

    def _check_performance(self, report: ValidationReport,
                            results: List[Dict]):
        """Check for performance regressions."""
        for result in results:
            output = result.get("output", "")
            try:
                data = json.loads(output) if isinstance(output, str) else output
                if not isinstance(data, dict):
                    continue

                for metric, value in data.items():
                    if isinstance(value, (int, float)) and metric in self.baselines:
                        baseline = self.baselines[metric]
                        change = (value - baseline) / baseline * 100

                        if abs(change) > 10:  # >10% change
                            regression = {
                                "metric": metric,
                                "baseline": f"{baseline:.2f}",
                                "current": f"{value:.2f}",
                                "change": f"{change:+.1f}%"
                            }
                            report.regressions.append(regression)
                            report.regression_detected = True
                            report.checks.append(ValidationCheck(
                                name=f"regression_{metric}",
                                severity=ValidationSeverity.WARNING if abs(change) < 20
                                         else ValidationSeverity.FAILURE,
                                message=f"{metric}: {change:+.1f}% vs baseline"
                            ))
                        else:
                            report.checks.append(ValidationCheck(
                                name=f"performance_{metric}",
                                severity=ValidationSeverity.PASS,
                                message=f"{metric}: within tolerance ({change:+.1f}%)"
                            ))

            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

    def _check_safety(self, report: ValidationReport,
                       results: List[Dict]):
        """Check for safety violations."""
        dangerous_patterns = [
            "rm -rf", "format C:", "drop table", "DELETE FROM",
            "shutdown", "reboot", "password", "secret_key",
        ]

        for result in results:
            output = str(result.get("output", ""))
            for pattern in dangerous_patterns:
                if pattern.lower() in output.lower():
                    report.checks.append(ValidationCheck(
                        name=f"safety_{result.get('step_id', '?')}",
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Dangerous pattern detected: '{pattern}'",
                        details=f"Found in output of step {result.get('step_id')}"
                    ))

    def _calculate_confidence(self, report: ValidationReport) -> float:
        """Calculate overall confidence score."""
        if not report.checks:
            return 0.0

        total = len(report.checks)
        passed = report.num_passed
        warnings = report.num_warnings

        score = (passed + warnings * 0.5) / total
        if report.num_critical > 0:
            score *= 0.3
        if report.regression_detected:
            score *= 0.7

        return min(max(score, 0.0), 1.0)

    def _generate_summary(self, report: ValidationReport) -> str:
        """Generate a human-readable summary."""
        status = "PASSED" if report.overall_pass else "FAILED"
        parts = [
            f"Validation {status} with {report.confidence_score:.0%} confidence.",
            f"{report.num_passed} passed, {report.num_warnings} warnings, "
            f"{report.num_failures} failures, {report.num_critical} critical.",
        ]
        if report.regression_detected:
            parts.append(f"⚠️ {len(report.regressions)} performance regression(s) detected.")
        return " ".join(parts)

    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        if report.num_critical > 0:
            recs.append("CRITICAL: Address critical safety/correctness issues before proceeding")
        if report.regression_detected:
            recs.append("Investigate performance regressions — consider reverting recent changes")
        if report.num_failures > 0:
            recs.append("Review and retry failed steps with additional context")
        if report.confidence_score < 0.7:
            recs.append("Low confidence — consider running additional validation passes")
        if not recs:
            recs.append("All checks passed — safe to proceed to deployment")
        return recs

    def _generate_bug_reports(self, report: ValidationReport) -> List[Dict[str, str]]:
        """Generate bug reports for critical issues."""
        bugs = []
        for check in report.checks:
            if check.severity in (ValidationSeverity.CRITICAL, ValidationSeverity.FAILURE):
                bug = {
                    "title": f"[NeuroSwarm] {check.name}: {check.message}",
                    "severity": check.severity.value,
                    "description": check.details or check.message,
                    "component": "neuroswarm-runtime",
                    "plan_id": report.plan_id,
                }
                bugs.append(bug)
        return bugs
