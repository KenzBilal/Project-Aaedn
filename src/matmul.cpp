#include "../include/matmul.hpp"
#include <cstring>
#include <immintrin.h>

namespace aaedn
{

void matmul(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc)
{
    std::memset(C, 0, M * ldc * sizeof(float));

    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k)
            {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

} // namespace aaedn