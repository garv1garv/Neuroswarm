#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <random>

// Forward declarations
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

class AttentionKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetDeviceCount(&num_gpus_);
        if (num_gpus_ == 0) GTEST_SKIP() << "No CUDA devices available";
        cudaSetDevice(0);
    }

    void fill_random(half* d_ptr, size_t n, float scale = 0.1f) {
        std::vector<half> h_data(n);
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, scale);
        for (size_t i = 0; i < n; i++) {
            h_data[i] = __float2half(dist(gen));
        }
        cudaMemcpy(d_ptr, h_data.data(), n * sizeof(half), cudaMemcpyHostToDevice);
    }

    void read_back(const half* d_ptr, std::vector<float>& out, size_t n) {
        std::vector<half> h_data(n);
        cudaMemcpy(h_data.data(), d_ptr, n * sizeof(half), cudaMemcpyDeviceToHost);
        out.resize(n);
        for (size_t i = 0; i < n; i++) {
            out[i] = __half2float(h_data[i]);
        }
    }

    int num_gpus_ = 0;
};

TEST_F(AttentionKernelTest, FlashAttentionV3_OutputNotZero) {
    int B = 1, H = 4, N = 64, D = 64;
    size_t tensor_size = B * H * N * D;

    half *d_Q, *d_K, *d_V, *d_O;
    float* d_L;
    cudaMalloc(&d_Q, tensor_size * sizeof(half));
    cudaMalloc(&d_K, tensor_size * sizeof(half));
    cudaMalloc(&d_V, tensor_size * sizeof(half));
    cudaMalloc(&d_O, tensor_size * sizeof(half));
    cudaMalloc(&d_L, B * H * N * sizeof(float));
    cudaMemset(d_O, 0, tensor_size * sizeof(half));

    fill_random(d_Q, tensor_size);
    fill_random(d_K, tensor_size);
    fill_random(d_V, tensor_size);

    neuroswarm::cuda::launch_flash_attention_v3(
        d_Q, d_K, d_V, d_O, d_L,
        B, H, N, D, true, nullptr, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> output;
    read_back(d_O, output, tensor_size);

    // At least some outputs should be non-zero
    float sum = 0;
    for (float v : output) sum += std::abs(v);
    EXPECT_GT(sum, 0.0f) << "Output is all zeros — kernel produced no output";

    // Check no NaN/Inf
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_FALSE(std::isnan(output[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(output[i])) << "Inf at index " << i;
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);
}

TEST_F(AttentionKernelTest, FlashAttentionV3_CausalMask) {
    int B = 1, H = 1, N = 16, D = 32;
    size_t tensor_size = B * H * N * D;

    half *d_Q, *d_K, *d_V, *d_O_causal, *d_O_full;
    float* d_L;
    cudaMalloc(&d_Q, tensor_size * sizeof(half));
    cudaMalloc(&d_K, tensor_size * sizeof(half));
    cudaMalloc(&d_V, tensor_size * sizeof(half));
    cudaMalloc(&d_O_causal, tensor_size * sizeof(half));
    cudaMalloc(&d_O_full, tensor_size * sizeof(half));
    cudaMalloc(&d_L, B * H * N * sizeof(float));

    fill_random(d_Q, tensor_size);
    fill_random(d_K, tensor_size);
    fill_random(d_V, tensor_size);

    // Run with causal mask
    neuroswarm::cuda::launch_flash_attention_v3(
        d_Q, d_K, d_V, d_O_causal, d_L, B, H, N, D, true, nullptr, nullptr);

    // Run without causal mask
    neuroswarm::cuda::launch_flash_attention_v3(
        d_Q, d_K, d_V, d_O_full, d_L, B, H, N, D, false, nullptr, nullptr);

    cudaDeviceSynchronize();

    std::vector<float> causal_out, full_out;
    read_back(d_O_causal, causal_out, tensor_size);
    read_back(d_O_full, full_out, tensor_size);

    // First row should be identical (no future tokens to mask)
    for (int d = 0; d < D; d++) {
        EXPECT_NEAR(causal_out[d], full_out[d], 1e-2f)
            << "First position differs between causal and full at dim " << d;
    }

    // Later rows should differ (causal masks future tokens)
    float diff_sum = 0;
    for (int i = D; i < N * D; i++) {
        diff_sum += std::abs(causal_out[i] - full_out[i]);
    }
    EXPECT_GT(diff_sum, 0.01f) << "Causal and full outputs are identical — mask not applied";

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O_causal); cudaFree(d_O_full); cudaFree(d_L);
}

TEST_F(AttentionKernelTest, GQA_OutputShape) {
    int B = 2, Q_H = 32, KV_H = 8, N = 128, D = 128;
    size_t q_size = B * Q_H * N * D;
    size_t kv_size = B * KV_H * N * D;

    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, q_size * sizeof(half));
    cudaMalloc(&d_K, kv_size * sizeof(half));
    cudaMalloc(&d_V, kv_size * sizeof(half));
    cudaMalloc(&d_O, q_size * sizeof(half));

    fill_random(d_Q, q_size);
    fill_random(d_K, kv_size);
    fill_random(d_V, kv_size);

    neuroswarm::cuda::launch_gqa_attention(
        d_Q, d_K, d_V, d_O, B, Q_H, KV_H, N, D, true, nullptr);
    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << "GQA kernel launch failed: " << cudaGetErrorString(err);

    std::vector<float> output;
    read_back(d_O, output, q_size);
    float sum = 0;
    for (float v : output) sum += std::abs(v);
    EXPECT_GT(sum, 0.0f);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
}
