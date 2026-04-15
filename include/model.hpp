#ifndef AAEDN_MODEL_HPP
#define AAEDN_MODEL_HPP

#include <cstdint>

namespace aaedn
{

struct Arena;
struct QuantBlock;

struct ModelMetadata
{
    uint32_t format_version;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t ffn_dim;
    uint32_t payload_checksum;
};

bool model_load_quantized(Arena* arena, const char* path, QuantBlock* out_block, ModelMetadata* out_meta);

} // namespace aaedn

#endif
