#include "../include/dot.hpp"
#include <immintrin.h>

namespace aaedn
{

float dot(const float* a, const float* b, size_t n)
{
    size_t i = 0;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    for (; i + 15 < n; i += 16)
    {
        __m256 a0 = _mm256_loadu_ps(&a[i]);
        __m256 b0 = _mm256_loadu_ps(&b[i]);
        __m256 a1 = _mm256_loadu_ps(&a[i + 8]);
        __m256 b1 = _mm256_loadu_ps(&b[i + 8]);

        sum0 = _mm256_fmadd_ps(a0, b0, sum0);
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
    }

    for (; i + 7 < n; i += 8)
    {
        __m256 a0 = _mm256_loadu_ps(&a[i]);
        __m256 b0 = _mm256_loadu_ps(&b[i]);
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);
    }

    __m256 sum = _mm256_add_ps(sum0, sum1);

    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_total = _mm_add_ps(sum_low, sum_high);

    sum_total = _mm_hadd_ps(sum_total, sum_total);
    sum_total = _mm_hadd_ps(sum_total, sum_total);

    float result;
    _mm_store_ss(&result, sum_total);

    for (; i < n; ++i)
    {
        result += a[i] * b[i];
    }

    return result;
}

} // namespace aaedn