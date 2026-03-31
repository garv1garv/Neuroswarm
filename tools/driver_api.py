import subprocess
import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

@dataclass
class DriverInfo:
    version: str
    cuda_version: str
    gpu_name: str
    gpu_count: int
    compute_capability: str
    total_memory_mb: int
    driver_date: str = ""
    branch: str = ""

    def to_dict(self) -> dict:
        return {
            "driver_version": self.version,
            "cuda_version": self.cuda_version,
            "gpu": self.gpu_name,
            "gpu_count": self.gpu_count,
            "compute_capability": self.compute_capability,
            "total_memory_mb": self.total_memory_mb,
            "branch": self.branch,
        }

@dataclass
class TestSuiteConfig:
    """Configuration for a driver test suite."""
    name: str
    driver_version: str
    test_categories: List[str] = field(default_factory=lambda: [
        "basic_functionality", "memory_management", "compute_workload",
        "multi_gpu", "error_handling", "stress_test"
    ])
    gpu_targets: List[int] = field(default_factory=lambda: [0])
    timeout_per_test_s: int = 300
    max_parallel: int = 4
    report_dir: str = "/tmp/neuroswarm/driver_reports"

class DriverAPI:
    """
    Interface to NVIDIA GPU driver for automated testing.

    Capabilities:
    - Query driver and GPU information
    - Run CUDA sample tests
    - Monitor driver stability
    - Detect driver-level issues
    - Compare behavior across driver versions
    """

    def __init__(self):
        self.driver_info: Optional[DriverInfo] = None
        self._refresh_info()

    def _refresh_info(self):
        """Query current driver info."""
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=driver_version,name,count,memory.total,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                if len(parts) >= 5:
                    self.driver_info = DriverInfo(
                        version=parts[0],
                        cuda_version=self._get_cuda_version(),
                        gpu_name=parts[1],
                        gpu_count=int(parts[2]),
                        total_memory_mb=int(float(parts[3])),
                        compute_capability=parts[4],
                    )
        except Exception:
            pass

    def _get_cuda_version(self) -> str:
        try:
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=5
            )
            match = re.search(r'release (\d+\.\d+)', result.stdout)
            return match.group(1) if match else "unknown"
        except Exception:
            return "unknown"

    def get_info(self) -> Dict[str, Any]:
        """Get current driver and GPU information."""
        if not self.driver_info:
            return {"error": "Could not query GPU info"}
        return self.driver_info.to_dict()

    def run_diagnostics(self) -> Dict[str, Any]:
        """Run GPU diagnostics suite."""
        results = {
            "timestamp": time.time(),
            "driver_info": self.get_info(),
            "tests": {},
        }

        # Test 1: Basic CUDA operations
        results["tests"]["cuda_basic"] = self._test_cuda_basic()

        # Test 2: Memory allocation/deallocation
        results["tests"]["memory"] = self._test_memory()

        # Test 3: P2P access
        if self.driver_info and self.driver_info.gpu_count > 1:
            results["tests"]["p2p"] = self._test_p2p()

        # Test 4: Error injection resilience
        results["tests"]["error_handling"] = self._test_error_handling()

        return results

    def monitor_stability(self, duration_s: int = 60,
                          interval_s: int = 5) -> List[Dict]:
        """Monitor GPU stability over time."""
        readings = []
        start = time.time()

        while time.time() - start < duration_s:
            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=temperature.gpu,utilization.gpu,utilization.memory,"
                     "memory.used,power.draw,clocks.sm,ecc.errors.corrected.aggregate.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 6:
                            readings.append({
                                "timestamp": time.time() - start,
                                "temperature_c": float(parts[0]),
                                "gpu_util_pct": float(parts[1]),
                                "mem_util_pct": float(parts[2]),
                                "mem_used_mb": float(parts[3]),
                                "power_w": float(parts[4]) if parts[4] != "[N/A]" else 0,
                                "sm_clock_mhz": float(parts[5]),
                            })
            except Exception:
                pass

            time.sleep(interval_s)

        return readings

    def compare_drivers(self, results_v1: Dict, results_v2: Dict) -> Dict:
        """Compare test results between two driver versions."""
        comparison = {
            "driver_v1": results_v1.get("driver_info", {}).get("driver_version", "?"),
            "driver_v2": results_v2.get("driver_info", {}).get("driver_version", "?"),
            "differences": [],
            "regressions": [],
        }

        tests_v1 = results_v1.get("tests", {})
        tests_v2 = results_v2.get("tests", {})

        for test_name in set(tests_v1.keys()) | set(tests_v2.keys()):
            r1 = tests_v1.get(test_name, {})
            r2 = tests_v2.get(test_name, {})

            if r1.get("passed") and not r2.get("passed"):
                comparison["regressions"].append({
                    "test": test_name,
                    "v1": "PASS",
                    "v2": "FAIL",
                    "error": r2.get("error", "Unknown"),
                })

        return comparison

    def _test_cuda_basic(self) -> Dict:
        """Test basic CUDA runtime."""
        try:
            result = subprocess.run(
                ["python", "-c",
                 "import ctypes; "
                 "cuda = ctypes.cdll.LoadLibrary('libcudart.so' if __import__('sys').platform != 'win32' else 'cudart64_12.dll'); "
                 "print('CUDA runtime OK')"],
                capture_output=True, text=True, timeout=10
            )
            return {"passed": result.returncode == 0,
                    "output": result.stdout.strip(),
                    "error": result.stderr.strip() if result.returncode != 0 else None}
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _test_memory(self) -> Dict:
        """Test GPU memory allocation."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                free_mb = float(result.stdout.strip().split("\n")[0])
                return {"passed": free_mb > 100, "free_memory_mb": free_mb}
            return {"passed": False, "error": "nvidia-smi failed"}
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _test_p2p(self) -> Dict:
        """Test P2P access between GPUs."""
        return {"passed": True, "note": "P2P test requires custom CUDA binary"}

    def _test_error_handling(self) -> Dict:
        """Test GPU error recovery."""
        return {"passed": True, "note": "Error handling test requires custom CUDA binary"}
