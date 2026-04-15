#ifndef AAEDN_DEQUANT_HPP
#define AAEDN_DEQUANT_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn
{

void dequant_4bit(const uint8_t* data, const uint16_t* scales, float* output, size_t n);

}

#endif