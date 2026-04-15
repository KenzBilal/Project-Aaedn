#include "../include/rope.hpp"
#include "../include/arena.hpp"
#include <cmath>

namespace aaedn
{

RoPETables rope_init(Arena* arena, size_t max_seq_len, size_t head_dim)
{
    RoPETables tables{};
    tables.max_seq_len = max_seq_len;
    tables.head_dim = head_dim;

    size_t half_dim = head_dim / 2;
    size_t table_size = max_seq_len * half_dim;

    tables.cos_table = (float*)arena_alloc(arena, table_size * sizeof(float), 32);
    tables.sin_table = (float*)arena_alloc(arena, table_size * sizeof(float), 32);

    float base_freq = 10000.0f;

    for (size_t pos = 0; pos < max_seq_len; ++pos)
    {
        for (size_t i = 0; i < half_dim; ++i)
        {
            float freq = base_freq / std::pow(2.0f, (float)i / (float)half_dim);
            float angle = (float)pos * freq;
            tables.cos_table[pos * half_dim + i] = std::cos(angle);
            tables.sin_table[pos * half_dim + i] = std::sin(angle);
        }
    }

    return tables;
}

void rope_apply(float* q, float* k, size_t seq_len, size_t head_dim, const RoPETables* tables)
{
    if (!tables || !tables->cos_table || !tables->sin_table)
        return;

    size_t half_dim = head_dim / 2;

    for (size_t pos = 0; pos < seq_len; ++pos)
    {
        for (size_t i = 0; i < half_dim; ++i)
        {
            float cos_val = tables->cos_table[pos * half_dim + i];
            float sin_val = tables->sin_table[pos * half_dim + i];

            float q0 = q[pos * head_dim + i];
            float q1 = q[pos * head_dim + i + half_dim];

            q[pos * head_dim + i] = cos_val * q0 - sin_val * q1;
            q[pos * head_dim + i + half_dim] = sin_val * q0 + cos_val * q1;

            float k0 = k[pos * head_dim + i];
            float k1 = k[pos * head_dim + i + half_dim];

            k[pos * head_dim + i] = cos_val * k0 - sin_val * k1;
            k[pos * head_dim + i + half_dim] = sin_val * k0 + cos_val * k1;
        }
    }
}

} // namespace aaedn