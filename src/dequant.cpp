#include "../include/dequant.hpp"
#include <immintrin.h>

namespace aaedn
{

void dequant_4bit(const uint8_t* data, const uint16_t* scales, float* output, size_t n)
{
    if (!data || !scales || scales[0] == 0)
    {
        for (size_t i = 0; i < n; ++i)
            output[i] = 0.0f;
        return;
    }

    for (size_t blk = 0; blk < n; blk += 32)
    {
        size_t remaining = (blk + 32 <= n) ? 32 : (n - blk);
        float scale = (float)scales[blk / 32];

        for (size_t i = 0; i < remaining; ++i)
        {
            uint8_t byte = data[(blk + i) / 2];
            uint8_t val = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            output[blk + i] = (float)val * scale;
        }
    }
}

} // namespace aaedn