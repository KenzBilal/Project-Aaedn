#include "../include/softmax.hpp"
#include <cmath>
#include <immintrin.h>

namespace aaedn
{

static inline float exp_polynomial(float x)
{
    if (x > 8.0f)
        x = 8.0f;
    if (x < -8.0f)
        x = -8.0f;
    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x2 * x2;
    return 1.0f + x + x2 * 0.5f + x3 * 0.1666667f + x4 * 0.04166667f + x4 * x2 * 0.008333334f;
}

void softmax(const float* input, float* output, size_t n)
{
    float max_val = input[0];
    for (size_t i = 1; i < n; ++i)
    {
        if (input[i] > max_val)
            max_val = input[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        float exp_val = exp_polynomial(input[i] - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }

    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; ++i)
    {
        output[i] *= inv_sum;
    }
}

} // namespace aaedn