import json
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

@dataclass
class RegressionAlert:
    metric: str
    baseline_value: float
    current_value: float
    change_pct: float
    severity: str  # "minor", "major", "critical"
    context: str = ""

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "baseline": self.baseline_value,
            "current": self.current_value,
            "change_pct": round(self.change_pct, 2),
            "severity": self.severity,
            "context": self.context,
        }

class RegressionDetector:
    """
    Detects performance regressions by comparing results against baselines.

    Features:
    - Statistical comparison with confidence intervals
    - Multi-metric tracking
    - Trend analysis over time
    - Automatic severity classification
    - Baseline management
    """

    THRESHOLDS = {
        "minor": 5.0,    # >5% change
        "major": 10.0,   # >10% change
        "critical": 20.0, # >20% change
    }

    HIGHER_IS_BETTER = {
        "throughput_tps", "tokens_per_second", "gpu_utilization",
        "cache_hit_rate", "occupancy", "bandwidth_gbps",
    }

    LOWER_IS_BETTER = {
        "latency_ms", "latency_p50_ms", "latency_p99_ms",
        "memory_peak_gb", "error_rate", "time_to_first_token_ms",
        "prefill_time_ms", "decode_time_ms",
    }

    def __init__(self, baseline_path: str = "baselines.json"):
        self.baseline_path = Path(baseline_path)
        self.baselines: Dict[str, List[float]] = {}
        self.history: List[Dict[str, float]] = []

        if self.baseline_path.exists():
            self.load_baselines()

    def load_baselines(self):
        """Load baselines from JSON file."""
        with open(self.baseline_path) as f:
            data = json.load(f)
            self.baselines = data.get("baselines", {})
            self.history = data.get("history", [])

    def save_baselines(self):
        """Save baselines to JSON file."""
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_path, "w") as f:
            json.dump({
                "baselines": self.baselines,
                "history": self.history[-100:],  # Keep last 100 entries
                "updated_at": time.time(),
            }, f, indent=2)

    def update_baseline(self, metric: str, value: float):
        """Add a measurement to the baseline history."""
        if metric not in self.baselines:
            self.baselines[metric] = []
        self.baselines[metric].append(value)
        # Keep rolling window
        self.baselines[metric] = self.baselines[metric][-50:]

    def set_baseline(self, metrics: Dict[str, float]):
        """Set baselines from a dictionary of metrics."""
        for metric, value in metrics.items():
            self.update_baseline(metric, value)
        self.save_baselines()

    def check(self, current: Dict[str, float]) -> List[RegressionAlert]:
        """
        Check current metrics against baselines.

        Args:
            current: Dictionary of metric_name → current_value

        Returns:
            List of RegressionAlert for detected regressions
        """
        alerts = []

        for metric, current_value in current.items():
            if metric not in self.baselines or not self.baselines[metric]:
                continue

            baseline_values = self.baselines[metric]
            baseline_mean = statistics.mean(baseline_values)
            baseline_std = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0

            if baseline_mean == 0:
                continue

            change_pct = ((current_value - baseline_mean) / abs(baseline_mean)) * 100

            # Determine if this is a regression
            is_regression = False
            if metric in self.HIGHER_IS_BETTER:
                is_regression = change_pct < 0  # Lower = worse
                effective_change = abs(change_pct) if is_regression else 0
            elif metric in self.LOWER_IS_BETTER:
                is_regression = change_pct > 0  # Higher = worse
                effective_change = abs(change_pct) if is_regression else 0
            else:
                effective_change = abs(change_pct)
                is_regression = effective_change > self.THRESHOLDS["minor"]

            if is_regression and effective_change > self.THRESHOLDS["minor"]:
                # Classify severity
                if effective_change >= self.THRESHOLDS["critical"]:
                    severity = "critical"
                elif effective_change >= self.THRESHOLDS["major"]:
                    severity = "major"
                else:
                    severity = "minor"

                # Statistical significance check
                if baseline_std > 0:
                    z_score = abs(current_value - baseline_mean) / baseline_std
                    if z_score < 2.0:  # Not statistically significant
                        severity = "minor"

                alerts.append(RegressionAlert(
                    metric=metric,
                    baseline_value=round(baseline_mean, 4),
                    current_value=round(current_value, 4),
                    change_pct=round(change_pct, 2),
                    severity=severity,
                    context=f"Baseline: μ={baseline_mean:.4f}, σ={baseline_std:.4f}, "
                            f"n={len(baseline_values)}"
                ))

        # Record in history
        entry = {**current, "_timestamp": time.time(), "_num_alerts": len(alerts)}
        self.history.append(entry)

        return sorted(alerts, key=lambda a: abs(a.change_pct), reverse=True)

    def trend_analysis(self, metric: str, window: int = 10) -> Dict[str, float]:
        """Analyze trend for a metric over recent history."""
        values = []
        for entry in self.history[-window:]:
            if metric in entry:
                values.append(entry[metric])

        if len(values) < 3:
            return {"trend": "insufficient_data", "slope": 0.0}

        # Simple linear regression
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0
        trend = "improving" if (
            (metric in self.HIGHER_IS_BETTER and slope > 0) or
            (metric in self.LOWER_IS_BETTER and slope < 0)
        ) else "degrading" if slope != 0 else "stable"

        return {
            "trend": trend,
            "slope": round(slope, 6),
            "recent_mean": round(statistics.mean(values[-3:]), 4),
            "overall_mean": round(y_mean, 4),
            "volatility": round(statistics.stdev(values) / y_mean * 100, 2) if y_mean != 0 else 0,
        }

    def report(self, alerts: List[RegressionAlert]) -> str:
        """Generate human-readable regression report."""
        if not alerts:
            return "✅ No regressions detected — all metrics within tolerance."

        lines = [
            "⚠️ Performance Regression Report",
            "=" * 50,
            "",
            f"Detected {len(alerts)} regression(s):",
            "",
        ]

        for alert in alerts:
            icon = {"minor": "🟡", "major": "🟠", "critical": "🔴"}[alert.severity]
            direction = "↓" if alert.change_pct < 0 else "↑"
            lines.append(
                f"  {icon} [{alert.severity.upper()}] {alert.metric}: "
                f"{alert.baseline_value} → {alert.current_value} "
                f"({direction}{abs(alert.change_pct):.1f}%)"
            )

        return "\n".join(lines)
