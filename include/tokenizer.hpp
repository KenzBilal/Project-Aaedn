#ifndef AAEDN_TOKENIZER_HPP
#define AAEDN_TOKENIZER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aaedn
{

struct Arena;

constexpr uint32_t TOKEN_BOS = 32765;
constexpr uint32_t TOKEN_EOS = 32766;
constexpr uint32_t TOKEN_PAD = 32767;

struct Tokenizer
{
    char* strings;
    uint32_t* id_to_offset;
    uint16_t* merges_first;
    uint16_t* merges_second;
    uint32_t vocab_size;
    uint32_t merge_count;
    uint32_t max_token_bytes;
    uint32_t format_version;
    uint32_t payload_checksum;
};

Tokenizer tokenizer_load(Arena* arena, const char* path);
std::vector<uint32_t> tokenizer_encode(const Tokenizer* tok, const char* text, bool add_bos);
std::string tokenizer_decode(const Tokenizer* tok, const std::vector<uint32_t>& ids);

} // namespace aaedn

#endif