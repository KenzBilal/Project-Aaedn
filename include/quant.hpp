#ifndef AAEDN_QUANT_HPP
#define AAEDN_QUANT_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn
{

struct Arena;

enum class QuantType : uint8_t
{
    NONE = 0,
    INT4 = 0,
    INT8 = 1
};

struct QuantBlock
{
    uint8_t* data;
    uint16_t* scales;
    size_t num_blocks;
    size_t block_size;
    QuantType type;
    uint32_t format_version;
    uint32_t checksum;
};

QuantBlock quant_load(Arena* arena, const char* path);
void quant_unload(QuantBlock* q);

} // namespace aaedn

#endif