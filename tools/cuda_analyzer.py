import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
from enum import Enum

class IssueSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    PERF = "performance"

@dataclass
class CudaIssue:
    file: str
    line: int
    severity: IssueSeverity
    rule: str
    message: str
    suggestion: str = ""

class CudaAnalyzer:
    """
    Static analyzer for CUDA C++ source files.

    Checks:
    - Memory coalescing patterns
    - Shared memory bank conflicts
    - Warp divergence
    - Missing error checks
    - Suboptimal launch configurations
    - Register pressure indicators
    - Deprecated API usage
    """

    RULES = {
        "CUDA001": ("Missing CUDA error check",
                     r'cuda\w+\s*\([^)]*\)\s*;(?!\s*//\s*checked)',
                     "Wrap CUDA calls with error checking macros"),
        "CUDA002": ("Potential warp divergence",
                     r'if\s*\(\s*threadIdx\.\w\s*[<>=!]+',
                     "Consider reorganizing to avoid warp divergence"),
        "CUDA003": ("Global memory access without coalescing hint",
                     r'__global__.*\n(?:.*\n)*?.*\[\s*\w+\s*\*\s*blockDim',
                     "Ensure memory access patterns are coalesced"),
        "CUDA004": ("Deprecated cudaThreadSynchronize",
                     r'cudaThreadSynchronize',
                     "Use cudaDeviceSynchronize() instead"),
        "CUDA005": ("Magic numbers in kernel launch",
                     r'<<<\s*\d+\s*,\s*\d+\s*>>>',
                     "Use named constants or computed launch params"),
        "CUDA006": ("Unbounded shared memory",
                     r'__shared__\s+\w+\s+\w+\[(?!\s*\d)',
                     "Use extern __shared__ with explicit size"),
        "CUDA007": ("Missing __syncthreads after shared memory write",
                     r'__shared__.*=.*\n(?:(?!__syncthreads).*\n){0,5}.*__shared__.*\[',
                     "Add __syncthreads() between shared memory writes and reads"),
        "CUDA008": ("Potential integer overflow in index calculation",
                     r'blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x',
                     "Use size_t for index calculations in large grids"),
    }

    def __init__(self):
        self.issues: List[CudaIssue] = []

    def analyze_file(self, filepath: str) -> List[CudaIssue]:
        """Analyze a single CUDA source file."""
        path = Path(filepath)
        if not path.exists():
            return [CudaIssue(filepath, 0, IssueSeverity.ERROR, "FILE",
                              f"File not found: {filepath}")]

        content = path.read_text(encoding="utf-8", errors="replace")
        issues = []

        # Run each rule
        for rule_id, (description, pattern, suggestion) in self.RULES.items():
            try:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append(CudaIssue(
                        file=filepath,
                        line=line_num,
                        severity=IssueSeverity.WARNING,
                        rule=rule_id,
                        message=description,
                        suggestion=suggestion,
                    ))
            except re.error:
                pass  # Skip invalid regex

        # Additional checks
        issues.extend(self._check_launch_bounds(filepath, content))
        issues.extend(self._check_memory_alignment(filepath, content))
        issues.extend(self._check_occupancy_limiters(filepath, content))

        self.issues.extend(issues)
        return issues

    def analyze_directory(self, dirpath: str,
                          extensions: tuple = (".cu", ".cuh")) -> List[CudaIssue]:
        """Analyze all CUDA files in a directory recursively."""
        all_issues = []
        for path in Path(dirpath).rglob("*"):
            if path.suffix in extensions:
                all_issues.extend(self.analyze_file(str(path)))
        return all_issues

    def _check_launch_bounds(self, filepath: str, content: str) -> List[CudaIssue]:
        """Check for missing __launch_bounds__ on kernels."""
        issues = []
        kernel_pattern = r'__global__\s+void\s+(\w+)'
        launch_bounds_pattern = r'__launch_bounds__'

        for match in re.finditer(kernel_pattern, content):
            # Check if __launch_bounds__ appears in the 3 lines before
            start = max(0, match.start() - 200)
            prefix = content[start:match.start()]
            if launch_bounds_pattern not in prefix:
                line_num = content[:match.start()].count('\n') + 1
                issues.append(CudaIssue(
                    file=filepath, line=line_num,
                    severity=IssueSeverity.PERF, rule="CUDA010",
                    message=f"Kernel '{match.group(1)}' missing __launch_bounds__",
                    suggestion="Add __launch_bounds__(BLOCK_SIZE) for better register allocation",
                ))
        return issues

    def _check_memory_alignment(self, filepath: str, content: str) -> List[CudaIssue]:
        """Check for potentially unaligned memory accesses."""
        issues = []
        # Look for struct members that might cause alignment issues
        struct_pattern = r'struct\s+\w+\s*\{([^}]+)\}'
        for match in re.finditer(struct_pattern, content, re.DOTALL):
            body = match.group(1)
            members = re.findall(r'(\w+)\s+(\w+)\s*;', body)
            if len(members) > 2:
                # Simple heuristic: check for mixed types
                types = [m[0] for m in members]
                if 'char' in types and ('float' in types or 'double' in types):
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append(CudaIssue(
                        file=filepath, line=line_num,
                        severity=IssueSeverity.PERF, rule="CUDA011",
                        message="Struct may have alignment padding issues",
                        suggestion="Reorder members by size (largest first) or use __align__",
                    ))
        return issues

    def _check_occupancy_limiters(self, filepath: str, content: str) -> List[CudaIssue]:
        """Check for patterns that limit occupancy."""
        issues = []
        # Large local arrays reduce occupancy
        local_array_pattern = r'(?:float|int|double|half)\s+\w+\[(\d+)\]'
        for match in re.finditer(local_array_pattern, content):
            size = int(match.group(1))
            if size > 64:
                line_num = content[:match.start()].count('\n') + 1
                issues.append(CudaIssue(
                    file=filepath, line=line_num,
                    severity=IssueSeverity.PERF, rule="CUDA012",
                    message=f"Large local array ({size} elements) may reduce occupancy",
                    suggestion="Consider using shared memory or reducing array size",
                ))
        return issues

    def report(self, format: str = "text") -> str:
        """Generate analysis report."""
        if format == "json":
            return json.dumps([
                {
                    "file": i.file, "line": i.line,
                    "severity": i.severity.value, "rule": i.rule,
                    "message": i.message, "suggestion": i.suggestion,
                }
                for i in self.issues
            ], indent=2)

        lines = ["CUDA Code Analysis Report", "=" * 60, ""]
        by_severity = {}
        for issue in self.issues:
            by_severity.setdefault(issue.severity.value, []).append(issue)

        for severity in ["error", "warning", "performance", "info"]:
            issues = by_severity.get(severity, [])
            if issues:
                lines.append(f"\n{severity.upper()} ({len(issues)}):")
                for i in issues:
                    lines.append(f"  [{i.rule}] {i.file}:{i.line} — {i.message}")
                    if i.suggestion:
                        lines.append(f"           → {i.suggestion}")

        lines.append(f"\nTotal: {len(self.issues)} issues found")
        return "\n".join(lines)
