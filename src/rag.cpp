#include "../include/rag.hpp"
#include "../include/arena.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace aaedn
{

static constexpr uint32_t kRagFormatVersion = 2;

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
    const char expected[] = "AAEDNRAG";
    return std::memcmp(magic, expected, 8) == 0;
}

static bool read_u32(FILE* f, uint32_t* out) { return std::fread(out, sizeof(uint32_t), 1, f) == 1; }
static bool read_u64(FILE* f, uint64_t* out) { return std::fread(out, sizeof(uint64_t), 1, f) == 1; }

RAGIndex rag_load(Arena* arena, const char* path)
{
    RAGIndex index{};

    FILE* f = std::fopen(path, "rb");
    if (!f)
        return index;

    uint8_t magic[8];
    uint32_t version = 0;
    uint64_t payload_size = 0;
    uint32_t payload_checksum = 0;
    uint32_t zero0 = 0;
    uint32_t zero1 = 0;
    uint32_t zero2 = 0;
    uint32_t zero3 = 0;
    uint8_t reserved[24];
    if (std::fread(magic, 1, sizeof(magic), f) != sizeof(magic) || !read_u32(f, &version) || !read_u64(f, &payload_size) ||
        !read_u32(f, &payload_checksum) || !read_u32(f, &zero0) || !read_u32(f, &zero1) || !read_u32(f, &zero2) ||
        !read_u32(f, &zero3) || std::fread(reserved, 1, sizeof(reserved), f) != sizeof(reserved))
    {
        std::fclose(f);
        return index;
    }

    if (!verify_magic(magic) || version != kRagFormatVersion)
    {
        std::fclose(f);
        return index;
    }

    long payload_start = std::ftell(f);
    std::fseek(f, 0, SEEK_END);
    long file_end = std::ftell(f);
    std::fseek(f, payload_start, SEEK_SET);
    if (payload_start < 0 || file_end < payload_start || (uint64_t)(file_end - payload_start) != payload_size)
    {
        std::fclose(f);
        return index;
    }

    std::vector<uint8_t> payload(payload_size);
    if (payload_size == 0 || std::fread(payload.data(), 1, payload_size, f) != payload_size)
    {
        std::fclose(f);
        return index;
    }
    std::fclose(f);

    uint32_t computed = 0x811C9DC5u;
    computed = fnv1a32_update(computed, payload.data(), payload.size());
    if (computed != payload_checksum)
        return index;

    const uint8_t* p = payload.data();
    const uint8_t* e = payload.data() + payload.size();
    auto read_payload_u32 = [&](uint32_t* out) {
        if ((size_t)(e - p) < sizeof(uint32_t))
            return false;
        std::memcpy(out, p, sizeof(uint32_t));
        p += sizeof(uint32_t);
        return true;
    };

    uint32_t n_tokens = 0;
    uint32_t n_dims = 0;
    uint32_t n_centroids = 0;
    uint32_t token_stride = 0;
    if (!read_payload_u32(&n_tokens) || !read_payload_u32(&n_dims) || !read_payload_u32(&n_centroids) ||
        !read_payload_u32(&token_stride))
        return RAGIndex{};

    size_t centroid_size = (size_t)n_centroids * n_dims * sizeof(float);
    size_t bucket_size = (size_t)n_centroids * sizeof(uint32_t);
    size_t token_data_size = (size_t)n_tokens * token_stride;
    size_t expected_payload_size = sizeof(uint32_t) * 4 + centroid_size + bucket_size + token_data_size;
    if (expected_payload_size != payload.size())
        return RAGIndex{};

    index.n_tokens = n_tokens;
    index.n_dims = n_dims;
    index.n_centroids = n_centroids;
    index.token_stride = token_stride;
    index.format_version = version;
    index.payload_checksum = payload_checksum;

    index.centroids = (float*)arena_alloc(arena, centroid_size, 32);
    std::memcpy(index.centroids, p, centroid_size);
    p += centroid_size;

    index.bucket_offsets = (uint32_t*)arena_alloc(arena, bucket_size, 32);
    std::memcpy(index.bucket_offsets, p, bucket_size);
    p += bucket_size;

    index.token_data = (uint8_t*)arena_alloc(arena, token_data_size, 32);
    std::memcpy(index.token_data, p, token_data_size);
    return index;
}

std::vector<uint32_t> rag_retrieve(const RAGIndex* index, const float* query, size_t top_k)
{
    if (!index || !query || index->n_tokens == 0)
        return {};

    static float centroid_scores[256];

    for (uint32_t c = 0; c < index->n_centroids; ++c)
    {
        const float* centroid = index->centroids + c * index->n_dims;
        __m256 sum = _mm256_setzero_ps();

        for (size_t d = 0; d < index->n_dims; d += 8)
        {
            __m256 q = _mm256_loadu_ps(&query[d]);
            __m256 c_vec = _mm256_loadu_ps(&centroid[d]);
            sum = _mm256_fmadd_ps(q, c_vec, sum);
        }

        float* s = (float*)&sum;
        centroid_scores[c] = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];
    }

    uint32_t top_centroids[8] = {0};
    float top_scores[8] = {-1e9f};

    for (uint32_t c = 0; c < index->n_centroids && c < 256; ++c)
    {
        if (centroid_scores[c] > top_scores[0])
        {
            for (int j = 7; j > 0; --j)
            {
                top_scores[j] = top_scores[j - 1];
                top_centroids[j] = top_centroids[j - 1];
            }
            top_scores[0] = centroid_scores[c];
            top_centroids[0] = c;
        }
    }

    std::vector<uint32_t> results;
    results.reserve(top_k);

    for (int ci = 0; ci < 8 && results.size() < top_k; ++ci)
    {
        uint32_t centroid_id = top_centroids[ci];
        uint32_t bucket_start = index->bucket_offsets[centroid_id * 4];
        uint32_t bucket_end =
            (centroid_id + 1 < index->n_centroids) ? index->bucket_offsets[(centroid_id + 1) * 4] : index->n_tokens;

        float best_score = -1e9f;
        uint32_t best_idx = bucket_start;

        for (uint32_t t = bucket_start; t < bucket_end; ++t)
        {
            float score = 0.0f;
            const uint8_t* token_data = index->token_data + t * index->token_stride;

            for (size_t d = 0; d < index->n_dims; d += 32)
            {
                for (int b = 0; b < 32; b += 8)
                {
                    uint8_t byte = token_data[(d + b) / 2];
                    float v0 = (float)(byte & 0x0F) * 0.1f;
                    float v1 = (float)((byte >> 4) & 0x0F) * 0.1f;
                    score += query[d + b] * v0;
                    if (b + 1 < 32)
                        score += query[d + b + 1] * v1;
                }
            }

            if (score > best_score)
            {
                best_score = score;
                best_idx = t;
            }

            if (results.size() >= top_k && best_score > top_scores[7])
                break;
        }

        results.push_back(best_idx);
    }

    return results;
}

} // namespace aaedn