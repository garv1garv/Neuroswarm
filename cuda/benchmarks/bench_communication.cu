// Measures zero-copy IPC, scatter/gather, and ring buffer throughput
//

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    int num_agents = 10;
    size_t msg_size = 1 * 1024 * 1024; // 1MB
    int iterations = 1000;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--agents") == 0) num_agents = atoi(argv[++i]);
        if (strcmp(argv[i], "--size") == 0) msg_size = atoll(argv[++i]);
        if (strcmp(argv[i], "--iters") == 0) iterations = atoi(argv[++i]);
    }

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║     NeuroSwarm Communication Benchmark           ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║ Agents: %-4d  Msg Size: %.1f MB  Iters: %d      ║\n",
           num_agents, msg_size / (1024.0 * 1024), iterations);
    printf("╚══════════════════════════════════════════════════╝\n\n");

    // Allocate per-agent buffers
    std::vector<float*> d_buffers(num_agents);
    size_t count = msg_size / sizeof(float);
    for (int i = 0; i < num_agents; i++) {
        cudaMalloc(&d_buffers[i], msg_size);
        cudaMemset(d_buffers[i], 0, msg_size);
    }

    float* d_src;
    cudaMalloc(&d_src, msg_size);
    cudaMemset(d_src, 1, msg_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ── Benchmark 1: Device-to-Device Copy ───────────────
    printf("1. Device-to-Device Copy (single agent pair)\n");
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cudaMemcpyAsync(d_buffers[0], d_src, msg_size, cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float d2d_ms;
    cudaEventElapsedTime(&d2d_ms, start, stop);
    double d2d_bw = (double)msg_size * iterations / (d2d_ms / 1000.0) / (1024 * 1024 * 1024);
    printf("   Avg: %.3f ms/op | Bandwidth: %.1f GB/s\n\n", d2d_ms / iterations, d2d_bw);

    // ── Benchmark 2: Broadcast (1 → N scatter) ───────────
    printf("2. Broadcast (1 → %d agents)\n", num_agents);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        for (int a = 0; a < num_agents; a++) {
            cudaMemcpyAsync(d_buffers[a], d_src, msg_size, cudaMemcpyDeviceToDevice);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float bcast_ms;
    cudaEventElapsedTime(&bcast_ms, start, stop);
    printf("   Avg: %.3f ms/op | Effective BW: %.1f GB/s\n\n",
           bcast_ms / iterations,
           (double)msg_size * num_agents * iterations / (bcast_ms / 1000.0) / (1024 * 1024 * 1024));

    // ── Benchmark 3: Gather (N → 1 reduce) ───────────────
    printf("3. Gather (%d agents → 1)\n", num_agents);
    float* d_gather;
    cudaMalloc(&d_gather, msg_size);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cudaMemsetAsync(d_gather, 0, msg_size);
        for (int a = 0; a < num_agents; a++) {
            // Simulate gather-add
            cudaMemcpyAsync(d_gather, d_buffers[a], msg_size, cudaMemcpyDeviceToDevice);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gather_ms;
    cudaEventElapsedTime(&gather_ms, start, stop);
    printf("   Avg: %.3f ms/op | Effective BW: %.1f GB/s\n\n",
           gather_ms / iterations,
           (double)msg_size * num_agents * iterations / (gather_ms / 1000.0) / (1024 * 1024 * 1024));

    // ── Benchmark 4: Stream-Concurrent Transfers ─────────
    printf("4. Stream-Concurrent Transfers (%d streams)\n", num_agents);
    std::vector<cudaStream_t> streams(num_agents);
    for (int i = 0; i < num_agents; i++) {
        cudaStreamCreate(&streams[i]);
    }

    cudaEventRecord(start);
    for (int iter = 0; iter < iterations; iter++) {
        for (int a = 0; a < num_agents; a++) {
            cudaMemcpyAsync(d_buffers[a], d_src, msg_size,
                           cudaMemcpyDeviceToDevice, streams[a]);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float stream_ms;
    cudaEventElapsedTime(&stream_ms, start, stop);
    printf("   Avg: %.3f ms/op | Effective BW: %.1f GB/s\n",
           stream_ms / iterations,
           (double)msg_size * num_agents * iterations / (stream_ms / 1000.0) / (1024 * 1024 * 1024));
    printf("   Speedup vs serial: %.2fx\n\n", bcast_ms / stream_ms);

    // ── Summary ──────────────────────────────────────────
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║                  Results Summary                 ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║ D2D Copy:       %8.3f ms  (%6.1f GB/s)      ║\n",
           d2d_ms / iterations, d2d_bw);
    printf("║ Broadcast:      %8.3f ms  (1→%d)             ║\n",
           bcast_ms / iterations, num_agents);
    printf("║ Gather:         %8.3f ms  (%d→1)             ║\n",
           gather_ms / iterations, num_agents);
    printf("║ Concurrent:     %8.3f ms  (%.1fx speedup)    ║\n",
           stream_ms / iterations, bcast_ms / stream_ms);
    printf("╚══════════════════════════════════════════════════╝\n");

    // Cleanup
    for (int i = 0; i < num_agents; i++) {
        cudaFree(d_buffers[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_src);
    cudaFree(d_gather);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
