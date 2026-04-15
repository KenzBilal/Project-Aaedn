#include "../include/kvcache.hpp"
#include "../include/arena.hpp"
#include <cstring>

namespace aaedn
{

static size_t compute_stride(size_t head_dim)
{
    size_t s = head_dim * sizeof(float) * 2;
    s = (s + 31) & ~31;
    return s;
}

void kvcache_init(KVCache* kv, Arena* arena, size_t max_tokens, size_t n_layers, size_t n_heads, size_t head_dim,
                  size_t max_agents)
{
    kv->max_tokens = max_tokens;
    kv->n_layers = n_layers;
    kv->n_heads = n_heads;
    kv->head_dim = head_dim;
    kv->stride = compute_stride(head_dim);

    size_t total_kv = max_tokens * n_layers * n_heads * head_dim * 2 * sizeof(float);
    kv->base = (uint8_t*)arena_alloc(arena, total_kv, 32);

    kv->max_branches = 256;
    kv->branch_table = (BranchNode*)arena_alloc(arena, sizeof(BranchNode) * kv->max_branches, 32);
    std::memset(kv->branch_table, 0, sizeof(BranchNode) * kv->max_branches);

    for (size_t i = 0; i < max_agents; ++i)
    {
        kv->segment_tables[i] = (Segment*)arena_alloc(arena, 32 * sizeof(Segment), 32);
        kv->segment_tables[i][0] = {0, (uint32_t)max_tokens};
    }

    kv->max_agents = max_agents;
    kv->generation_counter.store(0, std::memory_order_release);
}

void kvcache_append(KVCache* kv, int agent_id, size_t layer, size_t head, const float* k, const float* v)
{
    size_t token_idx = 0;
    size_t offset = ((layer * kv->n_heads + head) * kv->max_tokens + token_idx) * kv->stride;

    float* k_ptr = (float*)(kv->base + offset);
    float* v_ptr = (float*)(kv->base + offset + kv->head_dim * sizeof(float));

    for (size_t i = 0; i < kv->head_dim; ++i)
    {
        k_ptr[i] = k[i];
        v_ptr[i] = v[i];
    }
}

void kvcache_compact(KVCache* kv, uint16_t pruned_branch)
{
    kv->branch_table[pruned_branch].prune_flag.store(1, std::memory_order_release);
    kv->generation_counter.fetch_add(1, std::memory_order_release);
}

} // namespace aaedn