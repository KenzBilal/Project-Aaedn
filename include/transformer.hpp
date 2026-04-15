#ifndef AAEDN_TRANSFORMER_HPP
#define AAEDN_TRANSFORMER_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn
{

struct Arena;
struct KVCache;
struct RoPETables;
struct QuantBlock;

struct Transformer
{
    Arena* arena;
    KVCache* kv_cache;
    RoPETables* rope_tables;

    QuantBlock* qkv_weight;
    QuantBlock* out_proj_weight;
    QuantBlock* ffn_gate_weight;
    QuantBlock* ffn_up_weight;
    QuantBlock* ffn_down_weight;

    float* qkv_bias;
    float* out_proj_bias;
    float* ffn_bias;

    float* input_embedding;
    float* lm_head;

    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t hidden_dim;
    uint32_t ffn_dim;
    uint32_t vocab_size;
    uint32_t max_seq_len;
};

void transformer_init(Transformer* tf, Arena* arena, const char* model_path);
void transformer_forward(Transformer* tf, const uint32_t* token_ids, size_t seq_len, float* logits);

} // namespace aaedn

#endif