#ifndef AAEDN_MATMUL_HPP
#define AAEDN_MATMUL_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn
{

void matmul(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc);

}

#endif