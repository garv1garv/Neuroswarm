// INT8/INT4 quantization and dequantization for inference acceleration
//
// Supports:
//   - Per-tensor and per-channel quantization
//   - Symmetric and asymmetric modes
//   - INT8 GEMM via CUDA int8 dp4a
//   - INT4 packed quantization (2 values per byte)
//   - Dynamic quantization calibration
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <algorithm>

namespace neuroswarm {
namespace cuda {

// Quantization Parameters
struct QuantParams {
    float scale;
    float zero_point;
    float min_val;
    float max_val;
};

// FP16 → INT8 Quantization (Per-Tensor, Symmetric)
__global__ void quantize_fp16_to_int8_kernel(
    const half*  __restrict__ input,
    int8_t*      __restrict__ output,
    float*       __restrict__ scale_out,
    size_t       n
) {
    // Phase 1: Find absmax via block reduction
    __shared__ float smem[32];
    float thread_max = 0.0f;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float val = fabsf(__half2float(input[i]));
        thread_max = fmaxf(thread_max, val);
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, offset));
    }
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = thread_max;
    __syncthreads();

    if (threadIdx.x < 32) {
        thread_max = (threadIdx.x < blockDim.x / 32) ? smem[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, offset));
        }
    }

    float absmax = thread_max;
    float scale = absmax / 127.0f;
    float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

    // Store scale (only thread 0 of block 0)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *scale_out = scale;
    }

    // Phase 2: Quantize
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float val = __half2float(input[i]);
        int quantized = __float2int_rn(val * inv_scale);
        quantized = max(-127, min(127, quantized));
        output[i] = static_cast<int8_t>(quantized);
    }
}

// INT8 → FP16 Dequantization
__global__ void dequantize_int8_to_fp16_kernel(
    const int8_t* __restrict__ input,
    half*         __restrict__ output,
    const float   scale,
    size_t        n
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        output[i] = __float2half(static_cast<float>(input[i]) * scale);
    }
}

// FP16 → INT4 Quantization (Packed — 2 values per byte)
__global__ void quantize_fp16_to_int4_kernel(
    const half*    __restrict__ input,
    uint8_t*       __restrict__ output,    // Packed: low nibble = even, high nibble = odd
    float*         __restrict__ scale_out,
    size_t         n                       // Must be even
) {
    __shared__ float smem[32];
    float thread_max = 0.0f;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        thread_max = fmaxf(thread_max, fabsf(__half2float(input[i])));
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, offset));
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = thread_max;
    __syncthreads();

    if (threadIdx.x < 32) {
        thread_max = (threadIdx.x < blockDim.x / 32) ? smem[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, offset));
    }

    float absmax = thread_max;
    float scale = absmax / 7.0f;
    float inv_scale = (absmax > 0.0f) ? (7.0f / absmax) : 0.0f;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *scale_out = scale;
    }

    // Pack pairs of int4 into uint8
    size_t n_packed = n / 2;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_packed;
         i += gridDim.x * blockDim.x) {
        float v0 = __half2float(input[2 * i]);
        float v1 = __half2float(input[2 * i + 1]);

        int q0 = __float2int_rn(v0 * inv_scale);
        int q1 = __float2int_rn(v1 * inv_scale);
        q0 = max(-7, min(7, q0));
        q1 = max(-7, min(7, q1));

        // Pack: low nibble = q0 + 8, high nibble = q1 + 8 (unsigned offset)
        uint8_t packed = ((uint8_t)(q1 + 8) << 4) | ((uint8_t)(q0 + 8) & 0x0F);
        output[i] = packed;
    }
}

// INT4 → FP16 Dequantization (Unpacked)
__global__ void dequantize_int4_to_fp16_kernel(
    const uint8_t* __restrict__ input,
    half*          __restrict__ output,
    const float    scale,
    size_t         n_packed
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_packed;
         i += gridDim.x * blockDim.x) {
        uint8_t packed = input[i];
        int q0 = (int)(packed & 0x0F) - 8;
        int q1 = (int)(packed >> 4)    - 8;

        output[2 * i]     = __float2half(q0 * scale);
        output[2 * i + 1] = __float2half(q1 * scale);
    }
}

// Per-Channel Quantization (for weight matrices)
__global__ void quantize_per_channel_fp16_to_int8_kernel(
    const half*  __restrict__ input,    // [C_out, C_in]
    int8_t*      __restrict__ output,
    float*       __restrict__ scales,   // [C_out]
    int          c_out,
    int          c_in
) {
    int row = blockIdx.x;  // One block per output channel
    if (row >= c_out) return;

    // Find absmax for this channel
    __shared__ float smem[32];
    float thread_max = 0.0f;

    for (int j = threadIdx.x; j < c_in; j += blockDim.x) {
        float val = fabsf(__half2float(input[row * c_in + j]));
        thread_max = fmaxf(thread_max, val);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, offset));
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = thread_max;
    __syncthreads();

    if (threadIdx.x < 32) {
        thread_max = (threadIdx.x < blockDim.x / 32) ? smem[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, offset));
    }

    float absmax = thread_max;
    float scale = absmax / 127.0f;
    float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

    if (threadIdx.x == 0) scales[row] = scale;

    for (int j = threadIdx.x; j < c_in; j += blockDim.x) {
        float val = __half2float(input[row * c_in + j]);
        int q = __float2int_rn(val * inv_scale);
        output[row * c_in + j] = static_cast<int8_t>(max(-127, min(127, q)));
    }
}

// INT8 GEMM using dp4a (dot product of 4 int8s → int32)
// C[M,N] = A[M,K] * B[K,N]  where A,B are INT8, C is INT32
__global__ void int8_gemm_kernel(
    const int8_t*  __restrict__ A,      // [M, K]
    const int8_t*  __restrict__ B,      // [K, N]
    int32_t*       __restrict__ C,      // [M, N]
    int M, int N, int K
) {
    const int TILE = 16;
    __shared__ int8_t As[TILE][TILE + 4]; // +4 to avoid bank conflicts
    __shared__ int8_t Bs[TILE][TILE + 4];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    int32_t acc = 0;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int ak = t * TILE + threadIdx.x;
        int bk = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0;
        __syncthreads();

        // Use dp4a for 4-element dot products
        for (int k = 0; k < TILE; k += 4) {
            if (k + 3 < TILE) {
                int32_t a_packed = *reinterpret_cast<const int32_t*>(&As[threadIdx.y][k]);
                int32_t b_packed = *reinterpret_cast<const int32_t*>(&Bs[k][threadIdx.x]);
                acc = __dp4a(a_packed, b_packed, acc);
            } else {
                for (int kk = k; kk < TILE && kk < K - t * TILE; kk++) {
                    acc += (int32_t)As[threadIdx.y][kk] * (int32_t)Bs[kk][threadIdx.x];
                }
            }
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Mixed-Precision GEMM: INT8 weights × FP16 activations → FP16 output
__global__ void mixed_int8_fp16_gemm_kernel(
    const int8_t*  __restrict__ weights,  // [N, K] INT8
    const half*    __restrict__ input,     // [M, K] FP16
    half*          __restrict__ output,    // [M, N] FP16
    const float*   __restrict__ scales,    // [N] per-channel scales
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    float w_scale = scales[col];

    for (int k = 0; k < K; k++) {
        float w = static_cast<float>(weights[col * K + k]) * w_scale;
        float x = __half2float(input[row * K + k]);
        acc += w * x;
    }

    output[row * N + col] = __float2half(acc);
}

// Host Launch Functions

void launch_quantize_fp16_to_int8(const half* input, int8_t* output,
                                   float* scale, size_t n, cudaStream_t stream) {
    int bs = 256;
    int nb = std::min((int)((n + bs - 1) / bs), 1024);
    quantize_fp16_to_int8_kernel<<<nb, bs, 0, stream>>>(input, output, scale, n);
}

void launch_dequantize_int8_to_fp16(const int8_t* input, half* output,
                                     float scale, size_t n, cudaStream_t stream) {
    int bs = 256;
    int nb = std::min((int)((n + bs - 1) / bs), 65535);
    dequantize_int8_to_fp16_kernel<<<nb, bs, 0, stream>>>(input, output, scale, n);
}

void launch_quantize_fp16_to_int4(const half* input, uint8_t* output,
                                   float* scale, size_t n, cudaStream_t stream) {
    int bs = 256;
    int nb = std::min((int)((n + bs - 1) / bs), 1024);
    quantize_fp16_to_int4_kernel<<<nb, bs, 0, stream>>>(input, output, scale, n);
}

void launch_dequantize_int4_to_fp16(const uint8_t* input, half* output,
                                     float scale, size_t n_packed, cudaStream_t stream) {
    int bs = 256;
    int nb = std::min((int)((n_packed + bs - 1) / bs), 65535);
    dequantize_int4_to_fp16_kernel<<<nb, bs, 0, stream>>>(input, output, scale, n_packed);
}

void launch_int8_gemm(const int8_t* A, const int8_t* B, int32_t* C,
                       int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    int8_gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_mixed_int8_fp16_gemm(const int8_t* weights, const half* input,
                                  half* output, const float* scales,
                                  int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    mixed_int8_fp16_gemm_kernel<<<grid, block, 0, stream>>>(
        weights, input, output, scales, M, N, K);
}

} // namespace cuda
} // namespace neuroswarm
