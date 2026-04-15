#include "../include/thread.hpp"
#include <cstdio>
#include <cstdlib>

namespace aaedn
{

static void* thread_wrapper(void* arg)
{
    ThreadCtx* ctx = static_cast<ThreadCtx*>(arg);
    return ctx->user_data;
}

ThreadCtx* create_pinned_threads(ThreadFn fn, void* user_data, int num_threads, const int* cpu_ids)
{
    ThreadCtx* ctx = new ThreadCtx[num_threads];

    for (int i = 0; i < num_threads; ++i)
    {
        ctx[i].cpu_id = cpu_ids[i];
        ctx[i].user_data = user_data;

        pthread_create(&ctx[i].pthread, nullptr, fn, &ctx[i]);

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_ids[i], &cpuset);

        pthread_setaffinity_np(ctx[i].pthread, sizeof(cpu_set_t), &cpuset);
    }

    return ctx;
}

void join_threads(ThreadCtx* ctx, int num_threads)
{
    for (int i = 0; i < num_threads; ++i)
    {
        pthread_join(ctx[i].pthread, nullptr);
    }
    delete[] ctx;
}

} // namespace aaedn