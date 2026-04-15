#include "../include/arena.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main()
{
    constexpr size_t ARENA_SIZE = 1024 * 1024;
    constexpr int NUM_ALLOCS = 1000;

    aaedn::Arena arena = aaedn::arena_create(ARENA_SIZE);

    size_t expected_used = 0;
    void* allocs[NUM_ALLOCS];

    for (int i = 0; i < NUM_ALLOCS; ++i)
    {
        size_t size = 1 + (rand() % 256);
        size_t align = 32;

        allocs[i] = aaedn::arena_alloc(&arena, size, align);

        uintptr_t addr = reinterpret_cast<uintptr_t>(allocs[i]);
        if ((addr & (align - 1)) != 0)
        {
            std::printf("FAIL: misaligned at %d: %zx\n", i, addr);
            return 1;
        }

        expected_used += ((arena.used + 31) & ~31) - ((expected_used + 31) & ~31);
        expected_used = arena.used;
    }

    if (arena.used != expected_used)
    {
        std::printf("FAIL: used mismatch: %zu != %zu\n", arena.used, expected_used);
        return 1;
    }

    aaedn::arena_destroy(&arena);

    std::printf("PASS: all %d allocations aligned\n", NUM_ALLOCS);
    return 0;
}