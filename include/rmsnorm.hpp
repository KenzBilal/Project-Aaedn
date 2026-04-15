#ifndef AAEDN_RMSNORM_HPP
#define AAEDN_RMSNORM_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn
{

void rmsnorm(const float* input, float* output, const float* weight, size_t n, float eps = 1e-6f);

}

#endif