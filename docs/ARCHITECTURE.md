# Architecture Documentation

## System Overview

NeuroSwarm is a GPU-native multi-agent reasoning system built from first principles. Unlike traditional CPU-bound frameworks (LangChain, AutoGen), NeuroSwarm keeps the entire agentic pipeline — inference, communication, and orchestration — on the GPU.

```
┌──────────────────────────────────────────────────────────────────┐
│                    NeuroSwarm Architecture                       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           Agent Orchestrator (C++ Core)                  │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │    │
│  │  │ Planner  │  │Executors │  │Validator │             │    │
│  │  │ (70B)    │  │ (8B × N) │  │ (8B)     │             │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘             │    │
│  │       │              │              │                   │    │
│  │  ┌────▼──────────────▼──────────────▼─────┐            │    │
│  │  │     Communication Bus (Zero-Copy GPU)   │            │    │
│  │  │     ├─ Lock-free Ring Buffers           │            │    │
│  │  │     ├─ Pub/Sub Message Routing          │            │    │
│  │  │     └─ Shared Memory Regions            │            │    │
│  │  └────────────────────┬───────────────────┘            │    │
│  │                       │                                │    │
│  │  ┌────────────────────▼───────────────────┐            │    │
│  │  │      Batch Inference Engine             │            │    │
│  │  │      ├─ TensorRT-LLM Backend           │            │    │
│  │  │      ├─ KV Cache Manager               │            │    │
│  │  │      ├─ Speculative Decoding            │            │    │
│  │  │      └─ INT8/INT4 Quantization          │            │    │
│  │  └────────────────────┬───────────────────┘            │    │
│  │                       │                                │    │
│  │  ┌────────────────────▼───────────────────┐            │    │
│  │  │      GPU Subsystem (CUDA)               │            │    │
│  │  │      ├─ Flash Attention V3              │            │    │
│  │  │      ├─ Fused RoPE Embeddings           │            │    │
│  │  │      ├─ Memory Pool (Slab Allocator)    │            │    │
│  │  │      └─ Communication Kernels           │            │    │
│  │  └────────────────────────────────────────┘            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │ Prometheus Metrics│  │ Circuit      │  │ Agent          │    │
│  │ (Port 9090)       │  │ Breaker      │  │ Sandbox        │    │
│  └──────────────────┘  └──────────────┘  └────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## Component Deep Dives

### 1. Agent Orchestrator (`src/orchestrator.cpp`)

The central coordinator manages the complete lifecycle of all agents:

- **Agent Registry**: Tracks agents by ID and role, manages concurrent limits
- **Task Scheduler**: Priority queue with round-robin assignment per role
- **Reasoning Pipeline**: Planner → Executor → Validator loop with retry
- **Monitoring**: Real-time metrics, health checks, deadlock detection

### 2. Custom CUDA Kernels (`cuda/`)

#### Flash Attention V3 (`attention_kernels.cu`)
- **Online softmax**: Avoids materializing the full N×N attention matrix
- **Shared memory tiling**: 64×64 tiles with warp-level coordination
- **Fused RoPE**: Position embeddings applied during attention, not separately
- **GQA support**: Grouped-query attention with configurable Q/KV head ratios

#### Quantization (`quantization_kernels.cu`)
- **Symmetric INT8**: Per-channel calibration with absmax scaling
- **INT4 packing**: 8 values per 32-bit word with bit shifting
- **Mixed-precision GEMM**: INT8 weights × FP16 activations using `dp4a`
- **Dynamic quantization**: Runtime calibration without offline profiling

#### Memory Pool (`memory_pool.cu`)
- **Slab allocator**: Pre-allocated 256KB/1MB/4MB/16MB/64MB slab classes
- **Lock-free allocation**: Atomic operations for concurrent agent access
- **IPC support**: `cudaIpcGetMemHandle` for cross-process memory sharing
- **Per-agent accounting**: Track and limit memory usage per agent

### 3. Communication Bus (`src/communication.cpp`)

Zero-copy GPU-native inter-agent messaging:

- **Ring buffers**: Per-GPU lock-free circular buffers
- **P2P transfers**: Direct GPU-to-GPU copies via NVLink/PCIe
- **Pub/Sub routing**: Agents subscribe to message types
- **Shared regions**: Named GPU memory regions for collaborative state

### 4. Inference Engine (`src/inference_engine.cpp`)

- **TensorRT-LLM integration**: Optimized model execution
- **KV cache management**: Pre-allocated per-layer caches with batch indexing
- **Batch inference**: Continuous batching for throughput optimization
- **Speculative decoding**: Draft model verification for faster generation

### 5. Safety Systems

#### Circuit Breaker (`src/circuit_breaker.cpp`)
State machine: CLOSED → OPEN → HALF_OPEN with configurable thresholds

#### Sandbox (`src/sandbox.cpp`)
Policy-based isolation: file, network, command, GPU memory restrictions

#### Hallucination Detection (`src/agent_runtime.cpp`)
Self-consistency checking with factual grounding against observations

## Data Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  Planner Agent   │ ← Llama 3.1 70B (INT8)
│  Creates plan    │
└────────┬────────┘
         │ ExecutionPlan
         ▼
┌─────────────────┐
│ Executor Pool    │ ← Llama 3.1 8B × N (FP16)
│ (parallel)       │    Each runs in sandbox
│ Step 1 ──────►   │
│ Step 2 ──────►   │    Uses tools (GPU profiler,
│ Step 3 ──────►   │    CUDA analyzer, driver API)
└────────┬────────┘
         │ ExecutionResults
         ▼
┌─────────────────┐
│ Validator Agent  │ ← Llama 3.1 8B (FP16)
│ Checks results   │    Regression detection
│ Files bugs       │    Self-consistency check
└────────┬────────┘
         │ ValidationReport
         ▼
    Final Answer
```

## Performance Targets

| Metric | CPU Framework | NeuroSwarm | Improvement |
|--------|-------------|------------|-------------|
| Throughput | ~300 tok/s | ~15,000 tok/s | **50×** |
| Latency (P50) | ~600ms | ~12ms | **50×** |
| Memory | ~64GB RAM | ~14GB VRAM | **4.5× efficient** |
| Agent Comm | ~10ms (Redis) | ~0.1ms (GPU) | **100×** |

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
cmake --build . -j$(nproc)
```

## GPU Requirements

- NVIDIA Ampere (SM 80+) or newer recommended
- 24GB+ VRAM for single-GPU deployment
- 80GB+ (A100/H100) for full 70B planner model
- CUDA Toolkit 11.8+
- TensorRT-LLM (optional, for production inference)
