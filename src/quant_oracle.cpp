#include "../include/quant_oracle.hpp"
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace aaedn
{

void oracle_init(EntropyOracle* oracle)
{
    std::memset(oracle->histogram, 0, sizeof(oracle->histogram));
    oracle->sample_count = 0;
    oracle->current_entropy = 5.5f;
    oracle->entropy_ema = 5.5f;
    oracle->current_precision = QuantPrecision::BIT_4;
}

void oracle_update(EntropyOracle* oracle, const float* activations, size_t n)
{
    if (!oracle || !activations || n == 0)
        return;

    size_t sample_size = (n >= 1024) ? 1024 : n;
    const float* sample = activations + (n - sample_size);

    std::memset(oracle->histogram, 0, sizeof(oracle->histogram));

    for (size_t i = 0; i < sample_size; ++i)
    {
        float val = sample[i];
        if (val < -8.0f)
            val = -8.0f;
        else if (val > 8.0f)
            val = 8.0f;

        int bin = (int)((val + 8.0f) * 16.0f);
        if (bin < 0)
            bin = 0;
        else if (bin > 255)
            bin = 255;
        oracle->histogram[bin]++;
    }

    oracle->sample_count = sample_size;

    float inv_total = 1.0f / (float)sample_size;
    float entropy = 0.0f;

    for (int i = 0; i < 256; ++i)
    {
        if (oracle->histogram[i] > 0)
        {
            float p = (float)oracle->histogram[i] * inv_total;
            entropy -= p * std::log2(p + 1e-12f);
        }
    }

    float alpha = 0.1f;
    oracle->current_entropy = entropy;
    oracle->entropy_ema = alpha * entropy + (1.0f - alpha) * oracle->entropy_ema;
}

QuantPrecision oracle_get_precision(EntropyOracle* oracle, uint32_t current_layer, uint32_t total_layers)
{
    if (!oracle)
        return QuantPrecision::BIT_4;

    if (current_layer >= total_layers - 4)
        return QuantPrecision::BIT_4;

    float H = oracle->entropy_ema;

    if (H >= 5.5f)
        return QuantPrecision::BIT_4;

    if (H >= 3.5f)
        return QuantPrecision::BIT_4;

    return QuantPrecision::BIT_3;
}

} // namespace aaedn