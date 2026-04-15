#ifndef AAEDN_FUSED_MATMUL_HPP
#define AAEDN_FUSED_MATMUL_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn
{

struct Arena;

void fused_matmul_init(Arena* arena);
void fused_matmul(const float* input, const uint8_t* weights, const uint16_t* scales, float* output, size_t M, size_t N,
                  size_t K, size_t lda, size_t ldb, size_t ldc);
void fused_matmul_destroy();

} // namespace aaedn

#endif