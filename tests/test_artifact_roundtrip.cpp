#include "../include/arena.hpp"
#include "../include/quant.hpp"
#include "../include/rag.hpp"
#include "../include/tokenizer.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace
{

uint32_t fnv1a32(const std::vector<uint8_t>& data)
{
    uint32_t h = 0x811C9DC5u;
    for (uint8_t b : data)
    {
        h ^= b;
        h *= 0x01000193u;
    }
    return h;
}

void write_u32(std::vector<uint8_t>& out, uint32_t v)
{
    out.push_back((uint8_t)(v & 0xFF));
    out.push_back((uint8_t)((v >> 8) & 0xFF));
    out.push_back((uint8_t)((v >> 16) & 0xFF));
    out.push_back((uint8_t)((v >> 24) & 0xFF));
}

void write_u64(std::vector<uint8_t>& out, uint64_t v)
{
    for (int i = 0; i < 8; ++i)
        out.push_back((uint8_t)((v >> (i * 8)) & 0xFF));
}

void write_file(const std::string& path, const std::vector<uint8_t>& bytes)
{
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)bytes.data(), (std::streamsize)bytes.size());
}

std::vector<uint8_t> read_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::vector<uint8_t> build_model_artifact()
{
    std::vector<uint8_t> payload;
    payload.push_back(0x12);
    payload.push_back(0x34);
    payload.push_back(0x56);
    payload.push_back(0x78);
    payload.push_back(0x00);
    payload.push_back(0x00);
    uint16_t s0 = 1000;
    uint16_t s1 = 2000;
    payload.push_back((uint8_t)(s0 & 0xFF));
    payload.push_back((uint8_t)((s0 >> 8) & 0xFF));
    payload.push_back((uint8_t)(s1 & 0xFF));
    payload.push_back((uint8_t)((s1 >> 8) & 0xFF));

    std::vector<uint8_t> out;
    const char magic[] = "AAEDNMDL";
    out.insert(out.end(), magic, magic + 8);
    write_u32(out, 2);
    write_u64(out, payload.size());
    write_u32(out, fnv1a32(payload));
    write_u32(out, 24);
    write_u32(out, 16);
    write_u32(out, 64);
    write_u32(out, 32768);
    write_u32(out, 4);
    write_u32(out, 16);
    out.push_back(0);
    for (int i = 0; i < 15; ++i)
        out.push_back(0);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

std::vector<uint8_t> build_tokenizer_artifact()
{
    std::vector<uint8_t> payload;
    write_u32(payload, 2);
    write_u32(payload, 4);
    write_u32(payload, 1);
    write_u32(payload, 4);
    payload.push_back('a');
    payload.push_back(0);
    payload.push_back('b');
    payload.push_back(0);
    write_u32(payload, 0);
    write_u32(payload, 2);
    payload.push_back(0);
    payload.push_back(0);
    payload.push_back(1);
    payload.push_back(0);

    std::vector<uint8_t> out;
    const char magic[] = "AAEDNTOK";
    out.insert(out.end(), magic, magic + 8);
    write_u32(out, 2);
    write_u64(out, payload.size());
    write_u32(out, fnv1a32(payload));
    write_u32(out, 2);
    write_u32(out, 4);
    write_u32(out, 1);
    write_u32(out, payload.size());
    for (int i = 0; i < 24; ++i)
        out.push_back(0);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

std::vector<uint8_t> build_rag_artifact()
{
    std::vector<uint8_t> payload;
    write_u32(payload, 1);
    write_u32(payload, 8);
    write_u32(payload, 1);
    write_u32(payload, 8);
    for (int i = 0; i < 8; ++i)
        payload.push_back(0);
    write_u32(payload, 0);
    for (int i = 0; i < 8; ++i)
        payload.push_back((uint8_t)i);

    std::vector<uint8_t> out;
    const char magic[] = "AAEDNRAG";
    out.insert(out.end(), magic, magic + 8);
    write_u32(out, 2);
    write_u64(out, payload.size());
    write_u32(out, fnv1a32(payload));
    write_u32(out, 0);
    write_u32(out, 0);
    write_u32(out, 0);
    write_u32(out, 0);
    for (int i = 0; i < 24; ++i)
        out.push_back(0);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

} // namespace

int main()
{
    const std::string model_path = "/tmp/aaedn_roundtrip_model.abn";
    const std::string tok_path = "/tmp/aaedn_roundtrip_tokenizer.abt";
    const std::string rag_path = "/tmp/aaedn_roundtrip_index.afr";

    const std::vector<uint8_t> model_bytes = build_model_artifact();
    const std::vector<uint8_t> tok_bytes = build_tokenizer_artifact();
    const std::vector<uint8_t> rag_bytes = build_rag_artifact();
    write_file(model_path, model_bytes);
    write_file(tok_path, tok_bytes);
    write_file(rag_path, rag_bytes);

    aaedn::Arena arena = aaedn::arena_create(8 * 1024 * 1024, 64);

    aaedn::QuantBlock q = aaedn::quant_load(&arena, model_path.c_str());
    if (!q.data || q.format_version != 2 || q.checksum == 0)
        return 1;

    aaedn::Tokenizer tok = aaedn::tokenizer_load(&arena, tok_path.c_str());
    if (tok.vocab_size != 2 || tok.format_version != 2 || tok.payload_checksum == 0)
        return 1;

    aaedn::RAGIndex rag = aaedn::rag_load(&arena, rag_path.c_str());
    if (rag.n_tokens != 1 || rag.format_version != 2 || rag.payload_checksum == 0)
        return 1;

    if (read_file(model_path) != model_bytes || read_file(tok_path) != tok_bytes || read_file(rag_path) != rag_bytes)
        return 1;

    std::remove(model_path.c_str());
    std::remove(tok_path.c_str());
    std::remove(rag_path.c_str());
    aaedn::arena_destroy(&arena);

    std::printf("PASS: deterministic artifact round-trip\n");
    return 0;
}
