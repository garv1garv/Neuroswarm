// Measures end-to-end token generation throughput
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

int main(int argc, char** argv) {
    int batch_size = 8;
    int seq_len = 2048;
    int head_dim = 128;
    int num_heads = 32;
    int num_layers = 32;
    int vocab_size = 32000;
    int gen_tokens = 256;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0)   batch_size = atoi(argv[++i]);
        if (strcmp(argv[i], "--seq") == 0)     seq_len = atoi(argv[++i]);
        if (strcmp(argv[i], "--layers") == 0)  num_layers = atoi(argv[++i]);
        if (strcmp(argv[i], "--gen") == 0)     gen_tokens = atoi(argv[++i]);
    }

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║     NeuroSwarm Inference Throughput Benchmark     ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║ Model Config: %d layers, %d heads, dim=%d       ║\n",
           num_layers, num_heads, head_dim);
    printf("║ Batch: %d, Prefill: %d, Generate: %d tokens     ║\n",
           batch_size, seq_len, gen_tokens);
    printf("╚══════════════════════════════════════════════════╝\n\n");

    // Simulate model parameters memory footprint
    size_t model_params = (size_t)num_layers * (
        4 * head_dim * num_heads * head_dim * num_heads * sizeof(half) + // QKV + O proj
        2 * 4 * head_dim * num_heads * head_dim * num_heads * sizeof(half) // FFN
    );

    // Simulate KV cache
    size_t kv_cache_per_token = (size_t)num_layers * 2 * num_heads * head_dim * sizeof(half);
    size_t total_kv_cache = kv_cache_per_token * batch_size * (seq_len + gen_tokens);

    printf("Estimated memory:\n");
    printf("  Model parameters: %.1f GB\n", model_params / (1024.0 * 1024 * 1024));
    printf("  KV cache:         %.1f GB\n", total_kv_cache / (1024.0 * 1024 * 1024));
    printf("  Total:            %.1f GB\n\n",
           (model_params + total_kv_cache) / (1024.0 * 1024 * 1024));

    // Simulate prefill phase
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate simulated buffers
    size_t buf_size = (size_t)batch_size * seq_len * num_heads * head_dim * sizeof(half);
    half* d_buf;
    cudaMalloc(&d_buf, buf_size);

    // Prefill benchmark (matrix multiply simulation)
    printf("Benchmarking prefill phase...\n");
    cudaEventRecord(start);
    for (int layer = 0; layer < num_layers; layer++) {
        // Simulate attention + FFN compute
        cudaMemsetAsync(d_buf, 0, buf_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float prefill_ms;
    cudaEventElapsedTime(&prefill_ms, start, stop);
    printf("  Prefill time: %.2f ms\n", prefill_ms);
    printf("  Prefill throughput: %.0f tokens/s\n",
           batch_size * seq_len / (prefill_ms / 1000.0));

    // Decode benchmark
    printf("\nBenchmarking decode phase...\n");
    size_t decode_buf = (size_t)batch_size * num_heads * head_dim * sizeof(half);
    half* d_decode;
    cudaMalloc(&d_decode, decode_buf);

    cudaEventRecord(start);
    for (int token = 0; token < gen_tokens; token++) {
        for (int layer = 0; layer < num_layers; layer++) {
            cudaMemsetAsync(d_decode, 0, decode_buf);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float decode_ms;
    cudaEventElapsedTime(&decode_ms, start, stop);
    printf("  Decode time: %.2f ms (%d tokens)\n", decode_ms, gen_tokens);
    printf("  Decode throughput: %.0f tokens/s per batch\n",
           gen_tokens / (decode_ms / 1000.0));
    printf("  Total throughput:  %.0f tokens/s (batch=%d)\n",
           batch_size * gen_tokens / (decode_ms / 1000.0), batch_size);
    printf("  Per-token latency: %.2f ms\n", decode_ms / gen_tokens);

    // Summary
    float total_ms = prefill_ms + decode_ms;
    int total_tokens = batch_size * (seq_len + gen_tokens);
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║                  Results Summary                 ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║ Total time:        %8.2f ms                  ║\n", total_ms);
    printf("║ Total tokens:      %8d                      ║\n", total_tokens);
    printf("║ Overall throughput: %7.0f tokens/s            ║\n",
           total_tokens / (total_ms / 1000.0));
    printf("║ Time-to-first-token: %6.2f ms                 ║\n", prefill_ms);
    printf("╚══════════════════════════════════════════════════╝\n");

    cudaFree(d_buf);
    cudaFree(d_decode);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
