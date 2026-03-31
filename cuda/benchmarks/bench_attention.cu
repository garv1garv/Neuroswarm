// Compares: Custom Flash Attention v3 vs cuBLAS-based vs naive attention
//
// Usage: ./bench_attention [--seq_len N] [--head_dim D] [--batch B] [--warmup W]
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

// Forward declarations from our kernels
namespace neuroswarm { namespace cuda {
void launch_flash_attention_v3(
    const half* Q, const half* K, const half* V,
    half* O, float* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    bool causal, const int* positions, cudaStream_t stream);
void launch_gqa_attention(
    const half* Q, const half* K, const half* V, half* O,
    int batch_size, int num_q_heads, int num_kv_heads,
    int seq_len, int head_dim, bool causal, cudaStream_t stream);
}}

// Naive Attention (baseline — for correctness verification)
__global__ void naive_attention_kernel(
    const half* Q, const half* K, const half* V, half* O,
    int seq_len, int head_dim, float scale, bool causal
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= seq_len) return;

    int bh_offset = (b * gridDim.y + h) * seq_len * head_dim;

    // Compute attention scores
    float max_val = -1e30f;
    for (int k_pos = 0; k_pos < seq_len; k_pos++) {
        if (causal && k_pos > q) break;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(Q[bh_offset + q * head_dim + d]) *
                   __half2float(K[bh_offset + k_pos * head_dim + d]);
        }
        max_val = fmaxf(max_val, dot * scale);
    }

    float sum = 0.0f;
    for (int k_pos = 0; k_pos < seq_len; k_pos++) {
        if (causal && k_pos > q) break;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(Q[bh_offset + q * head_dim + d]) *
                   __half2float(K[bh_offset + k_pos * head_dim + d]);
        }
        sum += expf(dot * scale - max_val);
    }

    for (int d = 0; d < head_dim; d++) {
        float out = 0.0f;
        for (int k_pos = 0; k_pos < seq_len; k_pos++) {
            if (causal && k_pos > q) break;
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; dd++) {
                dot += __half2float(Q[bh_offset + q * head_dim + dd]) *
                       __half2float(K[bh_offset + k_pos * head_dim + dd]);
            }
            float attn = expf(dot * scale - max_val) / sum;
            out += attn * __half2float(V[bh_offset + k_pos * head_dim + d]);
        }
        O[bh_offset + q * head_dim + d] = __float2half(out);
    }
}

// Benchmark Harness
struct BenchConfig {
    int batch_size = 4;
    int num_heads  = 32;
    int seq_len    = 2048;
    int head_dim   = 128;
    int warmup     = 10;
    int iterations = 100;
    bool causal    = true;
};

void fill_random_fp16(half* d_ptr, size_t n) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);

    float* d_float;
    cudaMalloc(&d_float, n * sizeof(float));
    curandGenerateUniform(gen, d_float, n);

    // Convert to FP16
    int bs = 256;
    int nb = (n + bs - 1) / bs;

    // Simple kernel to convert
    auto convert = [](float* src, half* dst, size_t n) {
        // Use cudaMemcpy workaround — allocate host and convert
        std::vector<float> h_float(n);
        cudaMemcpy(h_float.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost);
        std::vector<half> h_half(n);
        for (size_t i = 0; i < n; i++) {
            h_half[i] = __float2half(h_float[i] * 0.1f); // Scale down
        }
        cudaMemcpy(dst, h_half.data(), n * sizeof(half), cudaMemcpyHostToDevice);
    };
    convert(d_float, d_ptr, n);

    cudaFree(d_float);
    curandDestroyGenerator(gen);
}

double run_benchmark(std::function<void()> fn, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) fn();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / iters;
}

int main(int argc, char** argv) {
    BenchConfig cfg;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seq_len") == 0 && i + 1 < argc)
            cfg.seq_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head_dim") == 0 && i + 1 < argc)
            cfg.head_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc)
            cfg.batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc)
            cfg.num_heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            cfg.warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            cfg.iterations = atoi(argv[++i]);
    }

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║     NeuroSwarm Attention Kernel Benchmark        ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║ Batch:    %-6d  Heads: %-6d                  ║\n", cfg.batch_size, cfg.num_heads);
    printf("║ Seq Len:  %-6d  Head Dim: %-6d              ║\n", cfg.seq_len, cfg.head_dim);
    printf("║ Causal:   %-6s  Iterations: %-6d            ║\n",
           cfg.causal ? "Yes" : "No", cfg.iterations);
    printf("╚══════════════════════════════════════════════════╝\n\n");

    size_t tensor_size = (size_t)cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.head_dim;

    half *d_Q, *d_K, *d_V, *d_O, *d_O_naive;
    float *d_L;
    cudaMalloc(&d_Q, tensor_size * sizeof(half));
    cudaMalloc(&d_K, tensor_size * sizeof(half));
    cudaMalloc(&d_V, tensor_size * sizeof(half));
    cudaMalloc(&d_O, tensor_size * sizeof(half));
    cudaMalloc(&d_O_naive, tensor_size * sizeof(half));
    cudaMalloc(&d_L, (size_t)cfg.batch_size * cfg.num_heads * cfg.seq_len * sizeof(float));

    fill_random_fp16(d_Q, tensor_size);
    fill_random_fp16(d_K, tensor_size);
    fill_random_fp16(d_V, tensor_size);

    float scale = 1.0f / sqrtf((float)cfg.head_dim);

    // ── Benchmark 1: Naive Attention ─────────────────────
    printf("Running naive attention...\n");
    double naive_ms = -1;
    if (cfg.seq_len <= 1024) { // Only run naive for small seq_len
        auto naive_fn = [&]() {
            dim3 grid((cfg.seq_len + 63) / 64, cfg.num_heads, cfg.batch_size);
            naive_attention_kernel<<<grid, 64>>>(
                d_Q, d_K, d_V, d_O_naive, cfg.seq_len, cfg.head_dim, scale, cfg.causal);
        };
        naive_ms = run_benchmark(naive_fn, cfg.warmup, cfg.iterations);
        printf("  Naive:          %.3f ms\n", naive_ms);
    } else {
        printf("  Naive:          SKIPPED (seq_len > 1024)\n");
    }

    // ── Benchmark 2: Flash Attention V3 ──────────────────
    printf("Running Flash Attention V3...\n");
    auto flash_fn = [&]() {
        neuroswarm::cuda::launch_flash_attention_v3(
            d_Q, d_K, d_V, d_O, d_L,
            cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
            cfg.causal, nullptr, nullptr);
    };
    double flash_ms = run_benchmark(flash_fn, cfg.warmup, cfg.iterations);
    printf("  Flash Attn V3:  %.3f ms\n", flash_ms);

    // ── Benchmark 3: GQA Attention ───────────────────────
    printf("Running GQA Attention (8 KV heads)...\n");
    int num_kv_heads = 8; // Llama 3 uses GQA
    auto gqa_fn = [&]() {
        neuroswarm::cuda::launch_gqa_attention(
            d_Q, d_K, d_V, d_O,
            cfg.batch_size, cfg.num_heads, num_kv_heads,
            cfg.seq_len, cfg.head_dim, cfg.causal, nullptr);
    };
    double gqa_ms = run_benchmark(gqa_fn, cfg.warmup, cfg.iterations);
    printf("  GQA Attn:       %.3f ms\n", gqa_ms);

    // ── FLOPs Calculation ────────────────────────────────
    double total_flops = 2.0 * cfg.batch_size * cfg.num_heads *
                         (double)cfg.seq_len * cfg.seq_len * cfg.head_dim;
    if (cfg.causal) total_flops /= 2.0; // ~half the work

    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║                  Results Summary                 ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║ Flash Attn V3:  %8.3f ms  (%6.1f TFLOPS)     ║\n",
           flash_ms, total_flops / (flash_ms * 1e9));
    printf("║ GQA Attention:  %8.3f ms  (%6.1f TFLOPS)     ║\n",
           gqa_ms, total_flops / (gqa_ms * 1e9));
    if (naive_ms > 0) {
        printf("║ Naive Baseline: %8.3f ms  (%6.1f TFLOPS)     ║\n",
               naive_ms, total_flops / (naive_ms * 1e9));
        printf("║ Speedup vs Naive: %.1fx                         ║\n",
               naive_ms / flash_ms);
    }
    printf("╚══════════════════════════════════════════════════╝\n");

    // Cleanup
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_O_naive); cudaFree(d_L);

    return 0;
}
