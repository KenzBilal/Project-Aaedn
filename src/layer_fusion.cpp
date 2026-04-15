#include "../include/layer_fusion.hpp"
#include "../include/dequant.hpp"
#include "../include/quant.hpp"
#include "../include/rmsnorm.hpp"
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace aaedn
{

static float normed_cache[1024];
static float qkv_cache[4096];

void fused_rmsnorm_qkv(const float* input, const float* rms_weight, const QuantBlock* qkv_weight, float* q_output,
                       float* k_output, float* v_output, size_t hidden_dim, size_t n_heads)
{
    if (!input || !q_output || !k_output || !v_output)
        return;

    if (!rms_weight)
    {
        for (size_t i = 0; i < hidden_dim; ++i)
        {
            float val = input[i];
            float sum_sq = 0.0f;
            for (size_t j = 0; j < hidden_dim; ++j)
                sum_sq += input[j] * input[j];
            float scale = 1.0f / (std::sqrt(sum_sq / hidden_dim + 1e-6f));
            normed_cache[i] = input[i] * scale;
        }
    }
    else
    {
        rmsnorm(input, normed_cache, rms_weight, hidden_dim, 1e-6f);
    }

    if (!qkv_weight || !qkv_weight->data || !qkv_weight->scales)
    {
        for (size_t i = 0; i < hidden_dim; ++i)
        {
            q_output[i] = normed_cache[i] * 0.01f;
            k_output[i] = normed_cache[i] * 0.01f;
            v_output[i] = normed_cache[i] * 0.01f;
        }
        return;
    }

    size_t qkv_dim = hidden_dim * 3;
    std::memset(qkv_cache, 0, qkv_dim * sizeof(float));
    dequant_4bit(qkv_weight->data, qkv_weight->scales, qkv_cache, qkv_dim * hidden_dim);

    for (size_t j = 0; j < hidden_dim; j += 8)
    {
        __m256 in_vec = _mm256_loadu_ps(&normed_cache[j]);

        for (size_t i = 0; i < hidden_dim * 3; i += 8)
        {
            __m256 w_vec = _mm256_loadu_ps(&qkv_cache[i * hidden_dim + j]);
            __m256 acc = _mm256_setzero_ps();
            for (size_t k = 0; k < 8; ++k)
            {
                __m256 in_s = _mm256_set1_ps(normed_cache[j + k]);
                acc = _mm256_fmadd_ps(in_s, w_vec, acc);
                w_vec = _mm256_loadu_ps(&qkv_cache[(i + 1) * hidden_dim + j]);
            }
            float sum[8];
            _mm256_storeu_ps(sum, acc);
            qkv_cache[i + j] = sum[0];
        }
    }

    size_t head_dim = hidden_dim / n_heads;
    for (size_t h = 0; h < n_heads; ++h)
    {
        for (size_t d = 0; d < head_dim; ++d)
        {
            size_t q_idx = h * head_dim + d;
            size_t k_idx = hidden_dim + h * head_dim + d;
            size_t v_idx = hidden_dim * 2 + h * head_dim + d;
            q_output[q_idx] = qkv_cache[h * head_dim + d];
            k_output[q_idx] = qkv_cache[hidden_dim + h * head_dim + d];
            v_output[q_idx] = qkv_cache[hidden_dim * 2 + h * head_dim + d];
        }
    }
}

} // namespace aaedn