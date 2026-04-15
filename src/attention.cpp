#include "../include/attention.hpp"
#include "../include/kvcache.hpp"
#include "../include/softmax.hpp"
#include <cmath>
#include <immintrin.h>

namespace aaedn
{

void attention(const float* q, const float* k_cache, const float* v_cache, const Segment* segments, size_t num_segments,
               float* output, size_t n_heads, size_t head_dim, float scale)
{
    if (!q || !k_cache || !v_cache || !segments)
    {
        for (size_t i = 0; i < n_heads * head_dim; ++i)
            output[i] = 0.0f;
        return;
    }
    for (size_t h = 0; h < n_heads; ++h)
    {
        const float* q_head = q + h * head_dim;
        float* out_head = output + h * head_dim;

        for (size_t i = 0; i < head_dim; ++i)
            out_head[i] = 0.0f;

        float scores[16384];
        size_t score_idx = 0;

        for (size_t seg = 0; seg < num_segments; ++seg)
        {
            uint32_t start = segments[seg].start_token;
            uint32_t len = segments[seg].length;

            for (uint32_t t = 0; t < len; ++t)
            {
                float dot = 0.0f;
                for (size_t i = 0; i < head_dim; ++i)
                {
                    dot += q_head[i] * k_cache[(start + t) * head_dim + i];
                }
                scores[score_idx++] = dot * scale;
            }
        }

        float max_val = scores[0];
        for (size_t i = 1; i < score_idx; ++i)
        {
            if (scores[i] > max_val)
                max_val = scores[i];
        }

        float sum = 0.0f;
        for (size_t i = 0; i < score_idx; ++i)
        {
            float exp_val = std::exp(scores[i] - max_val);
            scores[i] = exp_val;
            sum += exp_val;
        }

        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < score_idx; ++i)
        {
            scores[i] *= inv_sum;
        }

        score_idx = 0;
        for (size_t seg = 0; seg < num_segments; ++seg)
        {
            uint32_t start = segments[seg].start_token;
            uint32_t len = segments[seg].length;

            for (uint32_t t = 0; t < len; ++t)
            {
                float weight = scores[score_idx++];
                const float* v_ptr = v_cache + (start + t) * head_dim;
                for (size_t i = 0; i < head_dim; ++i)
                {
                    out_head[i] += weight * v_ptr[i];
                }
            }
        }
    }
}

} // namespace aaedn