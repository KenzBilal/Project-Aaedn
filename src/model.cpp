#include "../include/model.hpp"
#include "../include/quant.hpp"

namespace aaedn
{

bool model_load_quantized(Arena* arena, const char* path, QuantBlock* out_block, ModelMetadata* out_meta)
{
    if (!arena || !path || !out_block || !out_meta)
        return false;

    QuantBlock q = quant_load(arena, path);
    if (!q.data || !q.scales || q.num_blocks == 0)
        return false;

    *out_block = q;
    out_meta->format_version = q.format_version;
    out_meta->n_layers = 24;
    out_meta->n_heads = 16;
    out_meta->head_dim = 64;
    out_meta->vocab_size = 32768;
    out_meta->hidden_dim = 1024;
    out_meta->ffn_dim = 4096;
    out_meta->payload_checksum = q.checksum;
    return true;
}

} // namespace aaedn
