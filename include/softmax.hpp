#ifndef AAEDN_SOFTMAX_HPP
#define AAEDN_SOFTMAX_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn
{

void softmax(const float* input, float* output, size_t n);

}

#endif