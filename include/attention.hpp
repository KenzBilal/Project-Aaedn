#ifndef AAEDN_ATTENTION_HPP
#define AAEDN_ATTENTION_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn
{

struct KVCache;
struct Segment;

void attention(const float* q, const float* k_cache, const float* v_cache, const Segment* segments, size_t num_segments,
               float* output, size_t n_heads, size_t head_dim, float scale);

} // namespace aaedn

#endif