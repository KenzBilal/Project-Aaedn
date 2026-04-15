#include "../include/arena.hpp"
#include "../include/quant.hpp"
#include "../include/rag.hpp"
#include "../include/tokenizer.hpp"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace
{

void write_file(const std::string& path, const std::vector<uint8_t>& data)
{
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)data.data(), (std::streamsize)data.size());
}

std::vector<uint8_t> invalid_magic_file()
{
    return std::vector<uint8_t>(64, 0xAA);
}

std::vector<uint8_t> bad_checksum_model()
{
    std::vector<uint8_t> out;
    const char magic[] = "AAEDNMDL";
    out.insert(out.end(), magic, magic + 8);
    auto push_u32 = [&](uint32_t v) {
        out.push_back((uint8_t)(v & 0xFF));
        out.push_back((uint8_t)((v >> 8) & 0xFF));
        out.push_back((uint8_t)((v >> 16) & 0xFF));
        out.push_back((uint8_t)((v >> 24) & 0xFF));
    };
    auto push_u64 = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i)
            out.push_back((uint8_t)((v >> (i * 8)) & 0xFF));
    };
    std::vector<uint8_t> payload(10, 0x11);
    push_u32(2);
    push_u64(payload.size());
    push_u32(0xDEADBEEF);
    push_u32(24);
    push_u32(16);
    push_u32(64);
    push_u32(32768);
    push_u32(4);
    push_u32(16);
    out.push_back(0);
    for (int i = 0; i < 15; ++i)
        out.push_back(0);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

} // namespace

int main()
{
    const std::string bad_model_magic = "/tmp/aaedn_fail_model_magic.abn";
    const std::string bad_model_checksum = "/tmp/aaedn_fail_model_checksum.abn";
    const std::string bad_tok = "/tmp/aaedn_fail_tokenizer.abt";
    const std::string bad_rag = "/tmp/aaedn_fail_rag.afr";

    write_file(bad_model_magic, invalid_magic_file());
    write_file(bad_model_checksum, bad_checksum_model());
    write_file(bad_tok, invalid_magic_file());
    write_file(bad_rag, invalid_magic_file());

    aaedn::Arena arena = aaedn::arena_create(8 * 1024 * 1024, 64);

    aaedn::QuantBlock q1 = aaedn::quant_load(&arena, bad_model_magic.c_str());
    if (q1.data || q1.scales || q1.num_blocks != 0)
        return 1;

    aaedn::QuantBlock q2 = aaedn::quant_load(&arena, bad_model_checksum.c_str());
    if (q2.data || q2.scales || q2.num_blocks != 0)
        return 1;

    aaedn::Tokenizer tok = aaedn::tokenizer_load(&arena, bad_tok.c_str());
    if (tok.vocab_size != 0 || tok.merge_count != 0)
        return 1;

    aaedn::RAGIndex rag = aaedn::rag_load(&arena, bad_rag.c_str());
    if (rag.n_tokens != 0 || rag.n_dims != 0 || rag.token_stride != 0)
        return 1;

    std::remove(bad_model_magic.c_str());
    std::remove(bad_model_checksum.c_str());
    std::remove(bad_tok.c_str());
    std::remove(bad_rag.c_str());
    aaedn::arena_destroy(&arena);

    std::printf("PASS: startup/load failure-mode validation\n");
    return 0;
}
