#ifndef AAEDN_ROPE_HPP
#define AAEDN_ROPE_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn
{

struct Arena;

struct RoPETables
{
    float* cos_table;
    float* sin_table;
    size_t max_seq_len;
    size_t head_dim;
};

RoPETables rope_init(Arena* arena, size_t max_seq_len, size_t head_dim);
void rope_apply(float* q, float* k, size_t seq_len, size_t head_dim, const RoPETables* tables);

} // namespace aaedn

#endif