#ifndef AAEDN_RAG_HPP
#define AAEDN_RAG_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aaedn
{

struct Arena;

struct RAGIndex
{
    float* centroids;
    uint8_t* token_data;
    uint32_t* bucket_offsets;
    uint32_t n_tokens;
    uint32_t n_dims;
    uint32_t n_centroids;
    uint32_t token_stride;
    uint32_t format_version;
    uint32_t payload_checksum;
};

RAGIndex rag_load(Arena* arena, const char* path);
std::vector<uint32_t> rag_retrieve(const RAGIndex* index, const float* query, size_t top_k);

} // namespace aaedn

#endif