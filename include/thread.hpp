#ifndef AAEDN_THREAD_HPP
#define AAEDN_THREAD_HPP

#include <cstddef>
#include <cstdint>
#include <pthread.h>

namespace aaedn
{

constexpr int NUM_GUARDIANS = 4;
constexpr int NUM_ELASTIC = 8;

struct ThreadCtx
{
    pthread_t pthread;
    int cpu_id;
    void* user_data;
};

using ThreadFn = void* (*)(void*);

ThreadCtx* create_pinned_threads(ThreadFn fn, void* user_data, int num_threads, const int* cpu_ids);
void join_threads(ThreadCtx* ctx, int num_threads);

} // namespace aaedn

#endif