// Standalone fused kernels for applying RoPE to Q and K tensors
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace neuroswarm {
namespace cuda {

// RoPE Application Kernel — FP16
// Applies rotary embeddings in-place to Q or K tensor
// Layout: [batch, num_heads, seq_len, head_dim]
__global__ void apply_rope_fp16_kernel(
    half*        __restrict__ tensor,       // [B, H, N, D]
    const int*   __restrict__ positions,    // [B, N]
    int          num_heads,
    int          seq_len,
    int          head_dim,
    float        theta_base                 // Default: 10000.0
) {
    // Global thread index maps to (batch, head, seq_pos, dim_pair)
    int total_pairs = head_dim / 2;
    int total_work = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = gridDim.x * blockDim.x;

    // Iterate over all (batch, head, position, dim_pair) combinations
    int B = gridDim.z;
    int H = num_heads;
    int N = seq_len;
    int D2 = total_pairs;

    for (int idx = total_work; idx < B * H * N * D2;
         idx += total_elements)
    {
        int d2    = idx % D2;
        int rem   = idx / D2;
        int n     = rem % N;
        int rem2  = rem / N;
        int h     = rem2 % H;
        int b     = rem2 / H;

        int pos = positions[b * N + n];

        // Compute rotation angle
        float freq = 1.0f / powf(theta_base, (2.0f * d2) / (float)head_dim);
        float angle = pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        // Load pair
        int offset = ((b * H + h) * N + n) * head_dim;
        float x = __half2float(tensor[offset + 2 * d2]);
        float y = __half2float(tensor[offset + 2 * d2 + 1]);

        // Rotate
        float rx = x * cos_a - y * sin_a;
        float ry = x * sin_a + y * cos_a;

        // Store
        tensor[offset + 2 * d2]     = __float2half(rx);
        tensor[offset + 2 * d2 + 1] = __float2half(ry);
    }
}

// RoPE Application Kernel — FP32
__global__ void apply_rope_fp32_kernel(
    float*       __restrict__ tensor,
    const int*   __restrict__ positions,
    int          num_heads,
    int          seq_len,
    int          head_dim,
    float        theta_base
) {
    int D2 = head_dim / 2;
    int B  = gridDim.z;
    int H  = num_heads;
    int N  = seq_len;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < B * H * N * D2;
         idx += gridDim.x * blockDim.x)
    {
        int d2   = idx % D2;
        int rem  = idx / D2;
        int n    = rem % N;
        int rem2 = rem / N;
        int h    = rem2 % H;
        int b    = rem2 / H;

        int pos = positions[b * N + n];
        float freq  = 1.0f / powf(theta_base, (2.0f * d2) / (float)head_dim);
        float angle = pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        int offset = ((b * H + h) * N + n) * head_dim;
        float x = tensor[offset + 2 * d2];
        float y = tensor[offset + 2 * d2 + 1];

        tensor[offset + 2 * d2]     = x * cos_a - y * sin_a;
        tensor[offset + 2 * d2 + 1] = x * sin_a + y * cos_a;
    }
}

// Precompute RoPE Frequencies (cached on GPU for reuse)
__global__ void precompute_rope_freqs_kernel(
    float*  __restrict__ cos_cache,   // [max_seq_len, head_dim/2]
    float*  __restrict__ sin_cache,   // [max_seq_len, head_dim/2]
    int     max_seq_len,
    int     head_dim,
    float   theta_base
) {
    int pos = blockIdx.x;
    int d2  = threadIdx.x;

    if (pos < max_seq_len && d2 < head_dim / 2) {
        float freq  = 1.0f / powf(theta_base, (2.0f * d2) / (float)head_dim);
        float angle = pos * freq;
        cos_cache[pos * (head_dim / 2) + d2] = cosf(angle);
        sin_cache[pos * (head_dim / 2) + d2] = sinf(angle);
    }
}

// Apply RoPE using precomputed frequencies (faster for inference)
__global__ void apply_rope_cached_kernel(
    half*        __restrict__ tensor,
    const int*   __restrict__ positions,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int          num_heads,
    int          seq_len,
    int          head_dim
) {
    int D2 = head_dim / 2;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < gridDim.z * num_heads * seq_len * D2;
         idx += gridDim.x * blockDim.x)
    {
        int d2   = idx % D2;
        int rem  = idx / D2;
        int n    = rem % seq_len;
        int rem2 = rem / seq_len;
        int h    = rem2 % num_heads;
        int b    = rem2 / num_heads;

        int pos = positions[b * seq_len + n];
        float cos_a = cos_cache[pos * D2 + d2];
        float sin_a = sin_cache[pos * D2 + d2];

        int offset = ((b * num_heads + h) * seq_len + n) * head_dim;
        float x = __half2float(tensor[offset + 2 * d2]);
        float y = __half2float(tensor[offset + 2 * d2 + 1]);

        tensor[offset + 2 * d2]     = __float2half(x * cos_a - y * sin_a);
        tensor[offset + 2 * d2 + 1] = __float2half(x * sin_a + y * cos_a);
    }
}

// Host Launch Functions

void launch_apply_rope_fp16(
    half* tensor, const int* positions,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float theta_base, cudaStream_t stream
) {
    int total = batch_size * num_heads * seq_len * (head_dim / 2);
    int bs = 256;
    int nb = std::min((total + bs - 1) / bs, 65535);
    dim3 grid(nb, 1, batch_size);

    apply_rope_fp16_kernel<<<grid, bs, 0, stream>>>(
        tensor, positions, num_heads, seq_len, head_dim, theta_base
    );
}

void launch_apply_rope_fp32(
    float* tensor, const int* positions,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float theta_base, cudaStream_t stream
) {
    int total = batch_size * num_heads * seq_len * (head_dim / 2);
    int bs = 256;
    int nb = std::min((total + bs - 1) / bs, 65535);
    dim3 grid(nb, 1, batch_size);

    apply_rope_fp32_kernel<<<grid, bs, 0, stream>>>(
        tensor, positions, num_heads, seq_len, head_dim, theta_base
    );
}

void launch_precompute_rope_freqs(
    float* cos_cache, float* sin_cache,
    int max_seq_len, int head_dim, float theta_base,
    cudaStream_t stream
) {
    dim3 grid(max_seq_len);
    dim3 block(head_dim / 2);
    precompute_rope_freqs_kernel<<<grid, block, 0, stream>>>(
        cos_cache, sin_cache, max_seq_len, head_dim, theta_base
    );
}

void launch_apply_rope_cached(
    half* tensor, const int* positions,
    const float* cos_cache, const float* sin_cache,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream
) {
    int total = batch_size * num_heads * seq_len * (head_dim / 2);
    int bs = 256;
    int nb = std::min((total + bs - 1) / bs, 65535);
    dim3 grid(nb, 1, batch_size);

    apply_rope_cached_kernel<<<grid, bs, 0, stream>>>(
        tensor, positions, cos_cache, sin_cache, num_heads, seq_len, head_dim
    );
}

} // namespace cuda
} // namespace neuroswarm
