#ifndef AAEDN_KVCACHE_HPP
#define AAEDN_KVCACHE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace aaedn
{

struct Arena;

struct BranchNode
{
    uint16_t parent_branch_id;
    uint16_t divergence_token_idx;
    uint16_t branch_length;
    uint8_t valence_hint;
    std::atomic<uint8_t> prune_flag;
};

struct Segment
{
    uint32_t start_token;
    uint32_t length;
};

struct KVCache
{
    uint8_t* base;
    size_t max_tokens;
    size_t n_layers;
    size_t n_heads;
    size_t head_dim;
    size_t stride;

    BranchNode* branch_table;
    size_t max_branches;

    Segment* segment_tables[6];
    size_t max_agents;

    std::atomic<uint32_t> generation_counter;
};

void kvcache_init(KVCache* kv, Arena* arena, size_t max_tokens, size_t n_layers, size_t n_heads, size_t head_dim,
                  size_t max_agents);
void kvcache_append(KVCache* kv, int agent_id, size_t layer, size_t head, const float* k, const float* v);
void kvcache_compact(KVCache* kv, uint16_t pruned_branch);

} // namespace aaedn

#endif