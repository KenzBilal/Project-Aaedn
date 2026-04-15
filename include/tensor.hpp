#ifndef AAEDN_TENSOR_HPP
#define AAEDN_TENSOR_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn
{

struct Arena;

struct Tensor
{
    float* data;
    uint32_t dims[4];
    size_t strides[4];
    int ndims;
};

Tensor tensor_create(Arena* arena, int ndims, const uint32_t* dims, size_t align = 32);
size_t tensor_offset(const Tensor* t, const uint32_t* idx);

} // namespace aaedn

#endif