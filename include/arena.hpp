#ifndef AAEDN_ARENA_HPP
#define AAEDN_ARENA_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn {

struct Arena {
    uint8_t* base;
    size_t   capacity;
    size_t   used;
};

Arena arena_create(size_t bytes, size_t align = 32);
void*  arena_alloc(Arena* a, size_t bytes, size_t align = 32);
void   arena_destroy(Arena* a);

} // namespace aaedn

#endif