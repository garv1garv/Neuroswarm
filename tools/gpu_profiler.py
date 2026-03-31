import subprocess
import json
import time
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

@dataclass
class KernelProfile:
    """Profile data for a single CUDA kernel."""
    name: str
    duration_us: float
    grid_size: str
    block_size: str
    registers_per_thread: int = 0
    shared_memory_bytes: int = 0
    occupancy: float = 0.0
    throughput_gbps: float = 0.0
    num_invocations: int = 1

@dataclass
class ProfileReport:
    """Complete GPU profiling report."""
    timestamp: float = field(default_factory=time.time)
    total_gpu_time_ms: float = 0.0
    total_cpu_time_ms: float = 0.0
    gpu_utilization: float = 0.0
    memory_throughput_gbps: float = 0.0
    kernels: List[KernelProfile] = field(default_factory=list)
    memory_ops: List[Dict[str, Any]] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def top_kernels(self, n: int = 10) -> List[KernelProfile]:
        return sorted(self.kernels, key=lambda k: k.duration_us, reverse=True)[:n]

    def to_dict(self) -> dict:
        return {
            "total_gpu_time_ms": self.total_gpu_time_ms,
            "total_cpu_time_ms": self.total_cpu_time_ms,
            "gpu_utilization": self.gpu_utilization,
            "memory_throughput_gbps": self.memory_throughput_gbps,
            "num_kernels": len(self.kernels),
            "top_5_kernels": [
                {"name": k.name, "time_us": k.duration_us, "occupancy": k.occupancy}
                for k in self.top_kernels(5)
            ],
            "bottlenecks": self.bottlenecks,
            "recommendations": self.recommendations,
        }

class GpuProfiler:
    """
    GPU profiling tool using NVIDIA Nsight Systems and Nsight Compute.

    Features:
    - Automated profiling of NeuroSwarm workloads
    - Kernel-level analysis with occupancy and throughput
    - Memory operation tracking
    - Bottleneck identification
    - Optimization recommendations
    """

    def __init__(self, output_dir: str = "/tmp/neuroswarm/profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nsys_available = self._check_tool("nsys")
        self.ncu_available = self._check_tool("ncu")

    def _check_tool(self, tool: str) -> bool:
        try:
            subprocess.run([tool, "--version"], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def profile_command(self, command: str, name: str = "profile",
                        duration_s: int = 60) -> ProfileReport:
        """
        Profile a command using nsys.

        Args:
            command: Command to profile
            name: Profile name
            duration_s: Max duration in seconds

        Returns:
            ProfileReport with analysis
        """
        report = ProfileReport()
        profile_path = self.output_dir / f"{name}_{int(time.time())}"

        if self.nsys_available:
            # Run nsys profile
            nsys_cmd = [
                "nsys", "profile",
                "--output", str(profile_path),
                "--force-overwrite", "true",
                "--duration", str(duration_s),
                "--trace", "cuda,nvtx,osrt",
                "--gpu-metrics-device", "all",
                "--stats", "true",
            ] + command.split()

            try:
                result = subprocess.run(
                    nsys_cmd, capture_output=True, text=True,
                    timeout=duration_s + 30
                )

                if result.returncode == 0:
                    report = self._parse_nsys_output(result.stdout, result.stderr)
                    report.bottlenecks = self._identify_bottlenecks(report)
                    report.recommendations = self._generate_recommendations(report)
                else:
                    report.bottlenecks = [f"nsys error: {result.stderr[:500]}"]

            except subprocess.TimeoutExpired:
                report.bottlenecks = ["Profile timed out"]

        else:
            # Fallback: use nvidia-smi for basic metrics
            report = self._basic_profile()

        return report

    def profile_kernel(self, command: str, kernel_name: str) -> Optional[KernelProfile]:
        """
        Deep-profile a specific kernel using Nsight Compute (ncu).
        """
        if not self.ncu_available:
            return None

        ncu_cmd = [
            "ncu",
            "--kernel-name", kernel_name,
            "--launch-skip", "5",  # Skip warmup launches
            "--launch-count", "3",  # Profile 3 launches
            "--metrics",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
            "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
            "sm__warps_active.avg.pct_of_peak_sustained_active,"
            "launch__occupancy_per_register_count,"
            "launch__registers_per_thread",
            "--csv",
        ] + command.split()

        try:
            result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                return self._parse_ncu_output(result.stdout, kernel_name)
        except (subprocess.TimeoutExpired, Exception):
            pass

        return None

    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get current GPU metrics via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,temperature.gpu,utilization.gpu,"
                 "utilization.memory,memory.total,memory.used,memory.free,"
                 "power.draw,clocks.sm,clocks.mem",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                return {"error": "nvidia-smi failed"}

            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 11:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "temperature_c": float(parts[2]),
                        "gpu_util_pct": float(parts[3]),
                        "mem_util_pct": float(parts[4]),
                        "mem_total_mb": float(parts[5]),
                        "mem_used_mb": float(parts[6]),
                        "mem_free_mb": float(parts[7]),
                        "power_w": float(parts[8]) if parts[8] != "[N/A]" else 0,
                        "sm_clock_mhz": float(parts[9]),
                        "mem_clock_mhz": float(parts[10]),
                    })

            return {"gpus": gpus, "num_gpus": len(gpus)}

        except Exception as e:
            return {"error": str(e)}

    def _parse_nsys_output(self, stdout: str, stderr: str) -> ProfileReport:
        """Parse nsys stats output into ProfileReport."""
        report = ProfileReport()

        # Parse kernel stats
        kernel_section = False
        for line in (stdout + stderr).split("\n"):
            if "CUDA Kernel Statistics" in line:
                kernel_section = True
                continue
            if kernel_section and line.strip() and not line.startswith("-"):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        kernel = KernelProfile(
                            name=parts[-1],
                            duration_us=float(parts[0]) if parts[0].replace(".", "").isdigit() else 0,
                            grid_size="",
                            block_size="",
                        )
                        report.kernels.append(kernel)
                    except (ValueError, IndexError):
                        pass

            if kernel_section and line.strip() == "":
                kernel_section = False

        # Calculate totals
        if report.kernels:
            report.total_gpu_time_ms = sum(k.duration_us for k in report.kernels) / 1000

        return report

    def _parse_ncu_output(self, output: str, kernel_name: str) -> KernelProfile:
        """Parse ncu CSV output into KernelProfile."""
        profile = KernelProfile(name=kernel_name, duration_us=0, grid_size="", block_size="")

        for line in output.split("\n"):
            if "throughput" in line.lower() and "sm" in line.lower():
                try:
                    val = float(re.findall(r'[\d.]+', line)[-1])
                    profile.throughput_gbps = val
                except (ValueError, IndexError):
                    pass
            if "occupancy" in line.lower():
                try:
                    val = float(re.findall(r'[\d.]+', line)[-1])
                    profile.occupancy = val / 100.0
                except (ValueError, IndexError):
                    pass
            if "registers" in line.lower():
                try:
                    val = int(re.findall(r'\d+', line)[-1])
                    profile.registers_per_thread = val
                except (ValueError, IndexError):
                    pass

        return profile

    def _basic_profile(self) -> ProfileReport:
        """Basic profiling without nsys."""
        report = ProfileReport()
        metrics = self.get_gpu_metrics()
        if "gpus" in metrics:
            for gpu in metrics["gpus"]:
                report.gpu_utilization = max(report.gpu_utilization, gpu["gpu_util_pct"])
        return report

    def _identify_bottlenecks(self, report: ProfileReport) -> List[str]:
        """Identify performance bottlenecks from profile data."""
        bottlenecks = []

        top = report.top_kernels(3)
        if top:
            total = sum(k.duration_us for k in report.kernels)
            top_pct = sum(k.duration_us for k in top) / max(total, 1) * 100
            if top_pct > 80:
                bottlenecks.append(
                    f"Top 3 kernels account for {top_pct:.0f}% of GPU time: "
                    f"{', '.join(k.name for k in top)}"
                )

        for kernel in report.kernels:
            if kernel.occupancy > 0 and kernel.occupancy < 0.5:
                bottlenecks.append(
                    f"Low occupancy ({kernel.occupancy:.0%}) in kernel '{kernel.name}'"
                )

        return bottlenecks

    def _generate_recommendations(self, report: ProfileReport) -> List[str]:
        """Generate optimization recommendations."""
        recs = []

        for kernel in report.kernels:
            if kernel.occupancy > 0 and kernel.occupancy < 0.5:
                recs.append(
                    f"Increase occupancy of '{kernel.name}': "
                    f"reduce register usage or increase block size"
                )
            if kernel.registers_per_thread > 64:
                recs.append(
                    f"'{kernel.name}' uses {kernel.registers_per_thread} registers/thread "
                    f"— consider register pressure optimization"
                )

        if report.gpu_utilization < 50:
            recs.append("GPU utilization is low — consider increasing batch size or enabling concurrent kernels")

        return recs
