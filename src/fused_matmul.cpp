#include "../include/fused_matmul.hpp"
#include "../include/arena.hpp"
#include <cstring>
#include <immintrin.h>

namespace aaedn
{

constexpr size_t TILE_K = 256;
constexpr size_t TILE_N = 64;
constexpr size_t TILE_K_PREFETCH = 8;
constexpr size_t BLOCK_SIZE_K = 2048;
constexpr size_t BLOCK_SIZE_N = 512;
constexpr size_t BUFFER_SIZE = 128 * 1024;

struct DoubleBuffer
{
    uint8_t weight_buf[2][16384];
    uint16_t scale_buf[2][512];
    size_t buf_idx;
    bool valid[2];
};

static float* input_buffer = nullptr;
static DoubleBuffer db;
static bool initialized = false;
static Arena* g_arena = nullptr;

void fused_matmul_init(Arena* arena)
{
    g_arena = arena;
    input_buffer = (float*)arena_alloc(arena, BUFFER_SIZE, 32);

    db.buf_idx = 0;
    db.valid[0] = false;
    db.valid[1] = false;

    initialized = true;
}

void fused_matmul_destroy()
{
    if (g_arena && input_buffer)
    {
        input_buffer = nullptr;
    }
    initialized = false;
    g_arena = nullptr;
}

static inline void prefetch_weight_tile(const uint8_t* weights, const uint16_t* scales, size_t block_k, size_t block_n,
                                        size_t N, size_t buf_idx)
{
    size_t weight_bytes = block_k * N / 2;
    size_t scale_elems = block_k / 32;

    for (size_t i = 0; i < weight_bytes && i < 16384; ++i)
    {
        db.weight_buf[buf_idx][i] = weights[block_k * N / 2 + i];
    }

    for (size_t i = 0; i < scale_elems && i < 512; ++i)
    {
        db.scale_buf[buf_idx][i] = scales[block_k / 32 + i];
    }

    db.valid[buf_idx] = true;
}

void fused_matmul(const float* input, const uint8_t* weights, const uint16_t* scales, float* output, size_t M, size_t N,
                  size_t K, size_t lda, size_t ldb, size_t ldc)
{
    if (!input || !weights || !scales || !output)
        return;

    if (M == 0 || N == 0 || K == 0)
        return;

    std::memset(output, 0, M * ldc * sizeof(float));

    for (size_t m = 0; m < M; ++m)
    {
        const float* input_row = input + m * lda;

        for (size_t k = 0; k < 64 && k < K; ++k)
        {
            _mm_prefetch(input_row + k + 64, _MM_HINT_T0);
        }

        for (size_t block_k = 0; block_k < K; block_k += BLOCK_SIZE_K)
        {
            size_t k_end = (block_k + BLOCK_SIZE_K < K) ? block_k + BLOCK_SIZE_K : K;
            size_t k_blocks = (k_end - block_k + TILE_K - 1) / TILE_K;

            for (size_t i = 1; i < TILE_K_PREFETCH && i < k_blocks; ++i)
            {
                size_t prefetch_k = block_k + i * TILE_K;
                if (prefetch_k < K)
                {
                    size_t w_offset = (prefetch_k / 32) * N;
                    _mm_prefetch((const char*)(weights + w_offset), _MM_HINT_NTA);
                    _mm_prefetch((const char*)(scales + prefetch_k / 32), _MM_HINT_NTA);
                }
            }

            for (size_t k = block_k; k < k_end; k += TILE_K)
            {
                size_t k_tile_end = (k + TILE_K < k_end) ? k + TILE_K : k_end;

                for (size_t n_base = 0; n_base < N; n_base += BLOCK_SIZE_N)
                {
                    size_t n_end = (n_base + BLOCK_SIZE_N < N) ? n_base + BLOCK_SIZE_N : N;

                    float acc[512];
                    std::memset(acc, 0, (n_end - n_base) * sizeof(float));

                    __m256 sum0 = _mm256_setzero_ps();
                    __m256 sum1 = _mm256_setzero_ps();
                    __m256 sum2 = _mm256_setzero_ps();
                    __m256 sum3 = _mm256_setzero_ps();

                    for (size_t kk = k; kk < k_tile_end; ++kk)
                    {
                        float in_val = input_row[kk];
                        __m256 in_vec = _mm256_set1_ps(in_val);

                        size_t block_idx = kk / 32;
                        float scale = (float)scales[block_idx];
                        __m256 scale_vec = _mm256_set1_ps(scale);

                        for (size_t nn = n_base; nn + 31 < n_end; nn += 32)
                        {
                            uint8_t wbyte0 = weights[block_idx * N + nn];
                            uint8_t wbyte1 = weights[block_idx * N + nn + 1];
                            uint8_t wbyte2 = weights[block_idx * N + nn + 2];
                            uint8_t wbyte3 = weights[block_idx * N + nn + 3];

                            float w0 = (float)(wbyte0 & 0x0F) * scale;
                            float w1 = (float)((wbyte0 >> 4) & 0x0F) * scale;
                            float w2 = (float)(wbyte1 & 0x0F) * scale;
                            float w3 = (float)((wbyte1 >> 4) & 0x0F) * scale;
                            float w4 = (float)(wbyte2 & 0x0F) * scale;
                            float w5 = (float)((wbyte2 >> 4) & 0x0F) * scale;
                            float w6 = (float)(wbyte3 & 0x0F) * scale;
                            float w7 = (float)((wbyte3 >> 4) & 0x0F) * scale;

                            __m256 w_vec = _mm256_set_ps(w7, w6, w5, w4, w3, w2, w1, w0);
                            __m256 acc_vec = _mm256_loadu_ps(&acc[nn - n_base]);
                            acc_vec = _mm256_fmadd_ps(in_vec, w_vec, acc_vec);
                            _mm256_storeu_ps(&acc[nn - n_base], acc_vec);
                        }

                        for (size_t nn = ((n_end - n_base) / 32) * 32; nn < n_end - n_base; ++nn)
                        {
                            uint8_t wbyte = weights[block_idx * N + n_base + nn];
                            float w = (float)(wbyte & 0x0F) * scale;
                            acc[nn] += in_val * w;
                        }
                    }

                    for (size_t nn = 0; nn < n_end - n_base; ++nn)
                    {
                        output[m * ldc + n_base + nn] += acc[nn];
                    }
                }
            }
        }
    }
}

} // namespace aaedn