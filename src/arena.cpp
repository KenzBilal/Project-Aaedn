#include "../include/arena.hpp"
#include <cassert>
#include <cstdlib>

namespace aaedn {

Arena arena_create(size_t bytes, size_t align)
{
    void* ptr = std::aligned_alloc(align, bytes);
    assert(ptr != nullptr && "arena_create failed");
    return Arena{static_cast<uint8_t*>(ptr), bytes, 0};
}

void* arena_alloc(Arena* a, size_t bytes, size_t align)
{
    size_t aligned_used = (a->used + align - 1) & ~(align - 1);
    assert(aligned_used + bytes <= a->capacity && "arena overflow");
    a->used = aligned_used + bytes;
    return a->base + aligned_used;
}

void arena_destroy(Arena* a)
{
    std::free(a->base);
    a->base = nullptr;
    a->capacity = 0;
    a->used = 0;
}

}