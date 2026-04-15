#include "../include/rmsnorm.hpp"
#include <cmath>
#include <immintrin.h>

namespace aaedn
{

void rmsnorm(const float* input, float* output, const float* weight, size_t n, float eps)
{
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 7 < n; i += 8)
    {
        __m256 x = _mm256_loadu_ps(&input[i]);
        x = _mm256_mul_ps(x, x);
        sum = _mm256_add_ps(sum, x);
    }

    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_total = _mm_add_ps(sum_low, sum_high);
    sum_total = _mm_hadd_ps(sum_total, sum_total);
    sum_total = _mm_hadd_ps(sum_total, sum_total);

    float sum_scalar;
    _mm_store_ss(&sum_scalar, sum_total);

    for (; i < n; ++i)
    {
        sum_scalar += input[i] * input[i];
    }

    float rms = 1.0f / std::sqrt(sum_scalar / (float)n + eps);
    __m256 scale = _mm256_set1_ps(rms);

    i = 0;
    for (; i + 7 < n; i += 8)
    {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 w = _mm256_loadu_ps(&weight[i]);
        x = _mm256_mul_ps(x, scale);
        x = _mm256_mul_ps(x, w);
        _mm256_storeu_ps(&output[i], x);
    }

    for (; i < n; ++i)
    {
        output[i] = input[i] * rms * weight[i];
    }
}

} // namespace aaedn