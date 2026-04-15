#include "../include/transformer.hpp"
#include "../include/arena.hpp"
#include "../include/attention.hpp"
#include "../include/dequant.hpp"
#include "../include/ffn.hpp"
#include "../include/kvcache.hpp"
#include "../include/matmul.hpp"
#include "../include/quant.hpp"
#include "../include/rmsnorm.hpp"
#include "../include/rope.hpp"
#include "../include/tensor.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace aaedn
{

static float rmsnorm_1_scale[1024];
static float rmsnorm_2_scale[1024];
static float dequant_buffer[12288];
static float qkv_buf[3072];
static float q_buf[1024];
static float k_buf[1024];
static float v_buf[1024];
static float attn_output[1024];
static float ffn_out[1024];
static float hidden_states[24][1024];
static float residual[1024];
static float normed[1024];

static uint32_t current_pos = 0;

void transformer_init(Transformer* tf, Arena* arena, const char* model_path)
{
    tf->arena = arena;
    tf->n_layers = 24;
    tf->n_heads = 16;
    tf->head_dim = 64;
    tf->hidden_dim = 1024;
    tf->ffn_dim = 4096;
    tf->vocab_size = 32768;
    tf->max_seq_len = 16384;

    tf->kv_cache = (KVCache*)arena_alloc(arena, sizeof(KVCache), 32);
    kvcache_init(tf->kv_cache, arena, tf->max_seq_len, tf->n_layers, tf->n_heads, tf->head_dim, 6);

    tf->rope_tables = (RoPETables*)arena_alloc(arena, sizeof(RoPETables), 32);
    *tf->rope_tables = rope_init(arena, tf->max_seq_len, tf->head_dim);

    tf->qkv_weight = (QuantBlock*)arena_alloc(arena, sizeof(QuantBlock), 32);
    *tf->qkv_weight = quant_load(arena, model_path);
    if (!tf->qkv_weight->data)
    {
        tf->qkv_weight->data = (uint8_t*)arena_alloc(arena, tf->hidden_dim * tf->hidden_dim * 3 / 2, 32);
        tf->qkv_weight->scales =
            (uint16_t*)arena_alloc(arena, tf->hidden_dim * tf->hidden_dim * 3 / 32 * sizeof(uint16_t), 32);
        tf->qkv_weight->num_blocks = (tf->hidden_dim * tf->hidden_dim * 3 / 2) / 32;
        tf->qkv_weight->block_size = 32;
        tf->qkv_weight->type = QuantType::INT4;
        for (size_t i = 0; i < tf->qkv_weight->num_blocks; ++i)
            tf->qkv_weight->scales[i] = 10000;
    }

    tf->input_embedding = (float*)arena_alloc(arena, tf->vocab_size * tf->hidden_dim * sizeof(float), 32);
    for (size_t i = 0; i < tf->vocab_size * tf->hidden_dim; ++i)
        tf->input_embedding[i] = ((float)(rand() % 256) / 128.0f) - 1.0f;

    tf->lm_head = (float*)arena_alloc(arena, tf->vocab_size * tf->hidden_dim * sizeof(float), 32);
    for (size_t i = 0; i < tf->vocab_size * tf->hidden_dim; ++i)
        tf->lm_head[i] = ((float)(rand() % 256) / 128.0f) - 1.0f;

    tf->out_proj_weight = (QuantBlock*)arena_alloc(arena, sizeof(QuantBlock), 32);
    tf->out_proj_weight->data = (uint8_t*)arena_alloc(arena, tf->hidden_dim * tf->hidden_dim / 2, 32);
    tf->out_proj_weight->scales =
        (uint16_t*)arena_alloc(arena, tf->hidden_dim * tf->hidden_dim / 32 * sizeof(uint16_t), 32);
    tf->out_proj_weight->num_blocks = (tf->hidden_dim * tf->hidden_dim / 2) / 32;
    tf->out_proj_weight->block_size = 32;
    tf->out_proj_weight->type = QuantType::INT4;
    for (size_t i = 0; i < tf->out_proj_weight->num_blocks; ++i)
        tf->out_proj_weight->scales[i] = 10000;

    tf->ffn_gate_weight = (QuantBlock*)arena_alloc(arena, sizeof(QuantBlock), 32);
    tf->ffn_gate_weight->data = (uint8_t*)arena_alloc(arena, tf->ffn_dim * tf->hidden_dim / 2, 32);
    tf->ffn_gate_weight->scales =
        (uint16_t*)arena_alloc(arena, tf->ffn_dim * tf->hidden_dim / 32 * sizeof(uint16_t), 32);
    tf->ffn_gate_weight->num_blocks = (tf->ffn_dim * tf->hidden_dim / 2) / 32;
    tf->ffn_gate_weight->block_size = 32;
    tf->ffn_gate_weight->type = QuantType::INT4;
    for (size_t i = 0; i < tf->ffn_gate_weight->num_blocks; ++i)
        tf->ffn_gate_weight->scales[i] = 10000;

    tf->ffn_up_weight = (QuantBlock*)arena_alloc(arena, sizeof(QuantBlock), 32);
    tf->ffn_up_weight->data = (uint8_t*)arena_alloc(arena, tf->ffn_dim * tf->hidden_dim / 2, 32);
    tf->ffn_up_weight->scales = (uint16_t*)arena_alloc(arena, tf->ffn_dim * tf->hidden_dim / 32 * sizeof(uint16_t), 32);
    tf->ffn_up_weight->num_blocks = (tf->ffn_dim * tf->hidden_dim / 2) / 32;
    tf->ffn_up_weight->block_size = 32;
    tf->ffn_up_weight->type = QuantType::INT4;
    for (size_t i = 0; i < tf->ffn_up_weight->num_blocks; ++i)
        tf->ffn_up_weight->scales[i] = 10000;

    tf->ffn_down_weight = (QuantBlock*)arena_alloc(arena, sizeof(QuantBlock), 32);
    tf->ffn_down_weight->data = (uint8_t*)arena_alloc(arena, tf->hidden_dim * tf->ffn_dim / 2, 32);
    tf->ffn_down_weight->scales =
        (uint16_t*)arena_alloc(arena, tf->hidden_dim * tf->ffn_dim / 32 * sizeof(uint16_t), 32);
    tf->ffn_down_weight->num_blocks = (tf->hidden_dim * tf->ffn_dim / 2) / 32;
    tf->ffn_down_weight->block_size = 32;
    tf->ffn_down_weight->type = QuantType::INT4;
    for (size_t i = 0; i < tf->ffn_down_weight->num_blocks; ++i)
        tf->ffn_down_weight->scales[i] = 10000;

    tf->qkv_bias = (float*)arena_alloc(arena, (tf->hidden_dim * 3) * sizeof(float), 32);
    for (size_t i = 0; i < tf->hidden_dim * 3; ++i)
        tf->qkv_bias[i] = 0.0f;
    tf->out_proj_bias = (float*)arena_alloc(arena, tf->hidden_dim * sizeof(float), 32);
    for (size_t i = 0; i < tf->hidden_dim; ++i)
        tf->out_proj_bias[i] = 0.0f;

    for (size_t i = 0; i < tf->hidden_dim; ++i)
    {
        rmsnorm_1_scale[i] = 1.0f;
        rmsnorm_2_scale[i] = 1.0f;
    }

    current_pos = 0;
}

void transformer_forward(Transformer* tf, const uint32_t* token_ids, size_t seq_len, float* logits)
{
    if (seq_len == 0 || !token_ids || !logits)
    {
        for (size_t i = 0; i < tf->vocab_size; ++i)
            logits[i] = 0.0f;
        return;
    }

    if (token_ids[0] >= tf->vocab_size)
    {
        for (size_t i = 0; i < tf->vocab_size; ++i)
            logits[i] = 0.0f;
        return;
    }

    float* hidden = hidden_states[0];
    for (size_t i = 0; i < tf->hidden_dim; ++i)
        hidden[i] = tf->input_embedding[token_ids[0] * tf->hidden_dim + i];

    for (uint32_t layer = 0; layer < tf->n_layers; ++layer)
    {
        for (size_t i = 0; i < tf->hidden_dim; ++i)
            residual[i] = hidden[i];

        rmsnorm(hidden, normed, rmsnorm_1_scale, tf->hidden_dim, 1e-6f);

        size_t qkv_dim = tf->hidden_dim * 3;
        dequant_4bit(tf->qkv_weight->data, tf->qkv_weight->scales, dequant_buffer, qkv_dim * tf->hidden_dim);

        for (size_t j = 0; j < tf->hidden_dim; ++j)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < tf->hidden_dim; ++k)
                sum += normed[k] * dequant_buffer[k * tf->hidden_dim + j];
            qkv_buf[j] = sum + tf->qkv_bias[j];
        }
        for (size_t j = 0; j < tf->hidden_dim; ++j)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < tf->hidden_dim; ++k)
                sum += normed[k] * dequant_buffer[(tf->hidden_dim + k) * tf->hidden_dim + j];
            qkv_buf[tf->hidden_dim + j] = sum + tf->qkv_bias[tf->hidden_dim + j];
        }
        for (size_t j = 0; j < tf->hidden_dim; ++j)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < tf->hidden_dim; ++k)
                sum += normed[k] * dequant_buffer[(tf->hidden_dim * 2 + k) * tf->hidden_dim + j];
            qkv_buf[tf->hidden_dim * 2 + j] = sum + tf->qkv_bias[tf->hidden_dim * 2 + j];
        }

        for (size_t i = 0; i < tf->hidden_dim; ++i)
        {
            q_buf[i] = qkv_buf[i];
            k_buf[i] = qkv_buf[tf->hidden_dim + i];
            v_buf[i] = qkv_buf[tf->hidden_dim * 2 + i];
        }

        size_t seq_len_kv = current_pos + 1;
        rope_apply(q_buf, k_buf, seq_len_kv, tf->head_dim, tf->rope_tables);

        if (tf->kv_cache && tf->kv_cache->base)
        {
            for (size_t h = 0; h < tf->n_heads; ++h)
            {
                kvcache_append(tf->kv_cache, 0, layer, h, k_buf + h * tf->head_dim, v_buf + h * tf->head_dim);
            }
        }

        float scale = 1.0f / std::sqrt((float)tf->head_dim);

        Segment seg;
        seg.start_token = 0;
        seg.length = (uint32_t)(current_pos + 1);

        attention(q_buf, k_buf, v_buf, &seg, 1, attn_output, tf->n_heads, tf->head_dim, scale);

        dequant_4bit(tf->out_proj_weight->data, tf->out_proj_weight->scales, dequant_buffer + tf->hidden_dim,
                     tf->hidden_dim * tf->hidden_dim);

        for (size_t j = 0; j < tf->hidden_dim; ++j)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < tf->hidden_dim; ++i)
                sum += attn_output[i] * dequant_buffer[tf->hidden_dim + i * tf->hidden_dim + j];
            hidden[j] = sum + tf->out_proj_bias[j] + residual[j];
        }

        for (size_t i = 0; i < tf->hidden_dim; ++i)
            residual[i] = hidden[i];

        rmsnorm(hidden, normed, rmsnorm_2_scale, tf->hidden_dim, 1e-6f);

        ffn_swiglu(normed, ffn_out, 1, tf->hidden_dim, tf->ffn_dim, tf->ffn_gate_weight, tf->ffn_up_weight,
                   tf->ffn_down_weight, tf->ffn_gate_weight->scales, tf->ffn_up_weight->scales,
                   tf->ffn_down_weight->scales);

        for (size_t i = 0; i < tf->hidden_dim; ++i)
            hidden[i] = ffn_out[i] + residual[i];
    }

    rmsnorm(hidden, normed, rmsnorm_1_scale, tf->hidden_dim, 1e-6f);

    for (size_t j = 0; j < tf->vocab_size; ++j)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < tf->hidden_dim; ++i)
            sum += normed[i] * tf->lm_head[j * tf->hidden_dim + i];
        logits[j] = sum;
    }

    current_pos++;
    if (current_pos >= tf->max_seq_len)
        current_pos = 0;
}

} // namespace aaedn