#include "../include/arena.hpp"
#include "../include/tensor.hpp"
#include <cstdio>

int main()
{
    aaedn::Arena arena = aaedn::arena_create(1024 * 1024);

    uint32_t d4[] = {2, 3, 4, 8};
    aaedn::Tensor t = aaedn::tensor_create(&arena, 4, d4);

    uintptr_t addr = reinterpret_cast<uintptr_t>(t.data);
    if ((addr & 31) != 0)
    {
        std::printf("FAIL: not 32-byte aligned: %zx\n", addr);
        return 1;
    }

    if (t.dims[0] != 2 || t.dims[1] != 3 || t.dims[2] != 4 || t.dims[3] != 8)
    {
        std::printf("FAIL: dims mismatch\n");
        return 1;
    }

    if (t.strides[0] != 96 || t.strides[1] != 32 || t.strides[2] != 8 || t.strides[3] != 1)
    {
        std::printf("FAIL: strides mismatch: %zu %zu %zu %zu\n", t.strides[0], t.strides[1], t.strides[2],
                    t.strides[3]);
        return 1;
    }

    uint32_t idx[] = {1, 2, 3, 7};
    size_t off = aaedn::tensor_offset(&t, idx);
    size_t expected = 1 * 96 + 2 * 32 + 3 * 8 + 7 * 1;
    if (off != expected)
    {
        std::printf("FAIL: offset mismatch: %zu != %zu\n", off, expected);
        return 1;
    }

    t.data[off] = 42.0f;
    if (t.data[off] != 42.0f)
    {
        std::printf("FAIL: write/read mismatch\n");
        return 1;
    }

    aaedn::arena_destroy(&arena);

    std::printf("PASS: tensor alignment, strides, offset correct\n");
    return 0;
}