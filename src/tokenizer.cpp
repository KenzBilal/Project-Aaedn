#include "../include/tokenizer.hpp"
#include "../include/arena.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <queue>

namespace aaedn
{

static constexpr uint32_t kTokenizerFormatVersion = 2;

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
    const char expected[] = "AAEDNTOK";
    return std::memcmp(magic, expected, 8) == 0;
}

static bool read_u32(FILE* f, uint32_t* out) { return std::fread(out, sizeof(uint32_t), 1, f) == 1; }
static bool read_u64(FILE* f, uint64_t* out) { return std::fread(out, sizeof(uint64_t), 1, f) == 1; }

Tokenizer tokenizer_load(Arena* arena, const char* path)
{
    Tokenizer tok{};

    FILE* f = std::fopen(path, "rb");
    if (!f)
        return tok;

    uint8_t magic[8];
    uint32_t version = 0;
    uint64_t payload_size = 0;
    uint32_t payload_checksum = 0;
    uint32_t vocab_size = 0;
    uint32_t max_token_bytes = 0;
    uint32_t merge_count = 0;
    uint32_t declared_payload_size = 0;
    uint8_t reserved[24];
    if (std::fread(magic, 1, sizeof(magic), f) != sizeof(magic) || !read_u32(f, &version) || !read_u64(f, &payload_size) ||
        !read_u32(f, &payload_checksum) || !read_u32(f, &vocab_size) || !read_u32(f, &max_token_bytes) ||
        !read_u32(f, &merge_count) || !read_u32(f, &declared_payload_size) ||
        std::fread(reserved, 1, sizeof(reserved), f) != sizeof(reserved))
    {
        std::fclose(f);
        return tok;
    }

    if (!verify_magic(magic) || version != kTokenizerFormatVersion || payload_size != declared_payload_size)
    {
        std::fclose(f);
        return tok;
    }

    long payload_start = std::ftell(f);
    std::fseek(f, 0, SEEK_END);
    long file_end = std::ftell(f);
    std::fseek(f, payload_start, SEEK_SET);
    if (payload_start < 0 || file_end < payload_start || (uint64_t)(file_end - payload_start) != payload_size)
    {
        std::fclose(f);
        return tok;
    }

    std::vector<uint8_t> payload(payload_size);
    if (payload_size == 0 || std::fread(payload.data(), 1, payload_size, f) != payload_size)
    {
        std::fclose(f);
        return tok;
    }

    uint32_t computed = 0x811C9DC5u;
    computed = fnv1a32_update(computed, payload.data(), payload.size());
    if (computed != payload_checksum)
    {
        std::fclose(f);
        return tok;
    }

    const uint8_t* p = payload.data();
    const uint8_t* e = payload.data() + payload.size();
    auto read_payload_u32 = [&](uint32_t* out) {
        if ((size_t)(e - p) < sizeof(uint32_t))
            return false;
        std::memcpy(out, p, sizeof(uint32_t));
        p += sizeof(uint32_t);
        return true;
    };

    uint32_t payload_vocab_size = 0;
    uint32_t payload_max_token_bytes = 0;
    uint32_t payload_merge_count = 0;
    uint32_t string_block_size = 0;
    if (!read_payload_u32(&payload_vocab_size) || !read_payload_u32(&payload_max_token_bytes) ||
        !read_payload_u32(&payload_merge_count) || !read_payload_u32(&string_block_size))
    {
        std::fclose(f);
        return tok;
    }
    if (payload_vocab_size != vocab_size || payload_max_token_bytes != max_token_bytes || payload_merge_count != merge_count ||
        string_block_size > payload_size)
    {
        std::fclose(f);
        return tok;
    }

    tok.vocab_size = vocab_size;
    tok.merge_count = merge_count;
    tok.max_token_bytes = max_token_bytes;
    tok.format_version = version;
    tok.payload_checksum = payload_checksum;

    tok.strings = (char*)arena_alloc(arena, string_block_size, 1);
    if ((size_t)(e - p) < string_block_size)
    {
        std::fclose(f);
        return Tokenizer{};
    }
    std::memcpy(tok.strings, p, string_block_size);
    p += string_block_size;

    tok.id_to_offset = (uint32_t*)arena_alloc(arena, vocab_size * 4, 32);
    size_t offset_bytes = (size_t)vocab_size * sizeof(uint32_t);
    if ((size_t)(e - p) < offset_bytes)
    {
        std::fclose(f);
        return Tokenizer{};
    }
    std::memcpy(tok.id_to_offset, p, offset_bytes);
    p += offset_bytes;

    tok.merges_first = (uint16_t*)arena_alloc(arena, merge_count * 2, 32);
    tok.merges_second = (uint16_t*)arena_alloc(arena, merge_count * 2, 32);
    for (uint32_t i = 0; i < merge_count; ++i)
    {
        if ((size_t)(e - p) < sizeof(uint16_t) * 2)
        {
            std::fclose(f);
            return Tokenizer{};
        }
        std::memcpy(&tok.merges_first[i], p, sizeof(uint16_t));
        p += sizeof(uint16_t);
        std::memcpy(&tok.merges_second[i], p, sizeof(uint16_t));
        p += sizeof(uint16_t);
    }

    std::fclose(f);
    return tok;
}

struct MergeItem
{
    uint32_t rank;
    uint32_t pos;
    uint16_t a;
    uint16_t b;
    bool operator<(const MergeItem& other) const { return rank > other.rank; }
};

std::vector<uint32_t> tokenizer_encode(const Tokenizer* tok, const char* text, bool add_bos)
{
    if (!tok || !text)
        return {};

    std::vector<uint32_t> tokens;

    for (const char* p = text; *p; ++p)
    {
        if ((uint8_t)*p < 256)
            tokens.push_back((uint8_t)*p);
    }

    std::priority_queue<MergeItem> pq;
    for (uint32_t i = 0; i + 1 < tokens.size(); ++i)
    {
        for (uint32_t m = 0; m < tok->merge_count; ++m)
        {
            if (tok->merges_first[m] == tokens[i] && tok->merges_second[m] == tokens[i + 1])
            {
                pq.push({m, i, tok->merges_first[m], tok->merges_second[m]});
                break;
            }
        }
    }

    while (!pq.empty())
    {
        MergeItem top = pq.top();
        pq.pop();

        if (top.pos + 1 >= tokens.size())
            continue;
        if (tokens[top.pos] != top.a || tokens[top.pos + 1] != top.b)
            continue;

        tokens[top.pos] = tok->vocab_size + top.rank;
        tokens.erase(tokens.begin() + top.pos + 1);

        if (top.pos > 0)
        {
            for (uint32_t m = 0; m < tok->merge_count; ++m)
            {
                if (tok->merges_first[m] == tokens[top.pos - 1] && tok->merges_second[m] == tokens[top.pos])
                {
                    pq.push({m, top.pos - 1, tok->merges_first[m], tok->merges_second[m]});
                    break;
                }
            }
        }

        if (top.pos + 1 < tokens.size())
        {
            for (uint32_t m = 0; m < tok->merge_count; ++m)
            {
                if (tok->merges_first[m] == tokens[top.pos] && tok->merges_second[m] == tokens[top.pos + 1])
                {
                    pq.push({m, top.pos, tok->merges_first[m], tok->merges_second[m]});
                    break;
                }
            }
        }
    }

    if (add_bos)
    {
        tokens.insert(tokens.begin(), 32765);
    }

    return tokens;
}

std::string tokenizer_decode(const Tokenizer* tok, const std::vector<uint32_t>& ids)
{
    if (!tok)
        return {};

    std::string out;
    for (uint32_t id : ids)
    {
        if (id == 32766 || id == 32767)
            continue;
        if (id >= tok->vocab_size || !tok->strings || !tok->id_to_offset)
            continue;
        const char* s = tok->strings + tok->id_to_offset[id];
        out += s;
    }
    return out;
}

} // namespace aaedn