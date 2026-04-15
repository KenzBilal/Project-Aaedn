#include "../include/quant.hpp"
#include "../include/arena.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace aaedn
{

static constexpr uint32_t kModelFormatVersion = 2;

static uint32_t fnv1a32_update(uint32_t h, const uint8_t* data, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        h ^= data[i];
        h *= 0x01000193u;
    }
    return h;
}

static bool verify_magic(const uint8_t* magic)
{
    const char expected[] = "AAEDNMDL";
    return std::memcmp(magic, expected, 8) == 0;
}

static bool read_u32(FILE* f, uint32_t* out) { return std::fread(out, sizeof(uint32_t), 1, f) == 1; }
static bool read_u64(FILE* f, uint64_t* out) { return std::fread(out, sizeof(uint64_t), 1, f) == 1; }

QuantBlock quant_load(Arena* arena, const char* path)
{
    QuantBlock q{};

    FILE* f = std::fopen(path, "rb");
    if (!f)
        return q;

    uint8_t magic[8];
    uint32_t version = 0;
    uint64_t payload_size = 0;
    uint32_t payload_checksum = 0;
    uint32_t n_layers = 0;
    uint32_t n_heads = 0;
    uint32_t head_dim = 0;
    uint32_t vocab_size = 0;
    uint32_t hidden_dim = 0;
    uint32_t ffn_dim = 0;
    uint8_t quant_type = 0;
    uint8_t reserved[15];

    if (std::fread(magic, 1, sizeof(magic), f) != sizeof(magic) || !read_u32(f, &version) ||
        !read_u64(f, &payload_size) || !read_u32(f, &payload_checksum) || !read_u32(f, &n_layers) ||
        !read_u32(f, &n_heads) || !read_u32(f, &head_dim) || !read_u32(f, &vocab_size) || !read_u32(f, &hidden_dim) ||
        !read_u32(f, &ffn_dim) || std::fread(&quant_type, 1, 1, f) != 1 ||
        std::fread(reserved, 1, sizeof(reserved), f) != sizeof(reserved))
    {
        std::fclose(f);
        return q;
    }

    if (!verify_magic(magic) || version != kModelFormatVersion)
    {
        std::fclose(f);
        return q;
    }

    size_t expected_data_size = ((size_t)hidden_dim * (size_t)hidden_dim) / 2;
    size_t expected_scale_size = ((size_t)hidden_dim * (size_t)hidden_dim / 32) * sizeof(uint16_t);
    size_t expected_payload_size = expected_data_size + expected_scale_size;

    if ((uint64_t)expected_payload_size != payload_size)
    {
        std::fclose(f);
        return q;
    }

    q.type = quant_type == static_cast<uint8_t>(QuantType::INT8) ? QuantType::INT8 : QuantType::INT4;
    q.block_size = 32;
    q.format_version = version;
    q.checksum = payload_checksum;

    q.data = (uint8_t*)arena_alloc(arena, expected_data_size, 32);
    q.scales = (uint16_t*)arena_alloc(arena, expected_scale_size, 32);
    q.num_blocks = expected_data_size / 32;

    if (std::fread(q.data, 1, expected_data_size, f) != expected_data_size)
    {
        std::fclose(f);
        q.data = nullptr;
        q.scales = nullptr;
        q.num_blocks = 0;
        return q;
    }

    if (std::fread(q.scales, 1, expected_scale_size, f) != expected_scale_size)
    {
        std::fclose(f);
        q.data = nullptr;
        q.scales = nullptr;
        q.num_blocks = 0;
        return q;
    }

    uint32_t computed = 0x811C9DC5u;
    computed = fnv1a32_update(computed, q.data, expected_data_size);
    computed = fnv1a32_update(computed, reinterpret_cast<const uint8_t*>(q.scales), expected_scale_size);
    if (computed != payload_checksum)
    {
        std::fclose(f);
        q.data = nullptr;
        q.scales = nullptr;
        q.num_blocks = 0;
        return q;
    }

    std::fclose(f);
    return q;
}

void quant_unload(QuantBlock* q)
{
    q->data = nullptr;
    q->scales = nullptr;
    q->num_blocks = 0;
    q->block_size = 0;
    q->format_version = 0;
    q->checksum = 0;
}

} // namespace aaedn