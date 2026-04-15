#include "../include/tensor.hpp"
#include "../include/arena.hpp"

namespace aaedn
{

Tensor tensor_create(Arena* arena, int ndims, const uint32_t* dims, size_t align)
{
    Tensor t{};
    t.ndims = ndims;

    size_t total = 1;
    for (int i = 0; i < ndims; ++i)
    {
        t.dims[i] = dims[i];
        total *= dims[i];
    }
    for (int i = ndims; i < 4; ++i)
    {
        t.dims[i] = 1;
        t.strides[i] = 0;
    }

    t.strides[ndims - 1] = 1;
    for (int i = ndims - 2; i >= 0; --i)
    {
        t.strides[i] = t.strides[i + 1] * t.dims[i + 1];
    }

    t.data = static_cast<float*>(arena_alloc(arena, total * sizeof(float), align));
    return t;
}

size_t tensor_offset(const Tensor* t, const uint32_t* idx)
{
    size_t off = 0;
    for (int i = 0; i < t->ndims; ++i)
    {
        off += idx[i] * t->strides[i];
    }
    return off;
}

} // namespace aaedn