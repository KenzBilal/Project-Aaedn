#include "../include/matmul.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main()
{
    constexpr size_t M = 64;
    constexpr size_t N = 64;
    constexpr size_t K = 128;

    float* A = (float*)std::aligned_alloc(32, M * K * sizeof(float));
    float* B = (float*)std::aligned_alloc(32, K * N * sizeof(float));
    float* C = (float*)std::aligned_alloc(32, M * N * sizeof(float));
    float* ref = (float*)std::aligned_alloc(32, M * N * sizeof(float));

    for (size_t i = 0; i < M * K; ++i)
        A[i] = (float)(rand() % 100) / 100.0f;
    for (size_t i = 0; i < K * N; ++i)
        B[i] = (float)(rand() % 100) / 100.0f;
    for (size_t i = 0; i < M * N; ++i)
        C[i] = 0.0f;

    aaedn::matmul(A, B, C, M, N, K, K, N, N);

    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            ref[i * N + j] = sum;
        }
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < M * N; ++i)
    {
        float err = std::abs(C[i] - ref[i]);
        if (err > max_err)
            max_err = err;
    }

    std::free(A);
    std::free(B);
    std::free(C);
    std::free(ref);

    if (max_err > 1e-5f)
    {
        std::printf("FAIL: max error %e > 1e-5\n", max_err);
        return 1;
    }

    std::printf("PASS: matmul max error %e < 1e-5\n", max_err);
    return 0;
}