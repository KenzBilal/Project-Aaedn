#include "../include/smt.hpp"
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace aaedn
{

alignas(64) std::atomic<uint32_t> decode_active[NUM_CORES];

static long futex_wait(std::atomic<uint32_t>* addr, uint32_t val)
{
    return syscall(SYS_futex, addr, FUTEX_WAIT, val, nullptr, nullptr, 0);
}

static long futex_wake(std::atomic<uint32_t>* addr, int count)
{
    return syscall(SYS_futex, addr, FUTEX_WAKE, count, nullptr, nullptr, 0);
}

void smt_init()
{
    for (int i = 0; i < NUM_CORES; ++i)
    {
        decode_active[i].store(STATE_IDLE, std::memory_order_release);
    }
}

void smt_elastic_park(int core_id)
{
    while (true)
    {
        uint32_t state = decode_active[core_id].load(std::memory_order_relaxed);
        if (state == STATE_IDLE)
            break;

        if (state == STATE_DECODE_ACTIVE)
        {
            uint32_t expected = STATE_DECODE_ACTIVE;
            if (!decode_active[core_id].compare_exchange_strong(expected, STATE_ELASTIC_PARKED,
                                                                std::memory_order_acq_rel, std::memory_order_relaxed))
            {
                continue;
            }
        }

        futex_wait(&decode_active[core_id], STATE_ELASTIC_PARKED);
    }
}

void smt_guardian_start_decode(int core_id)
{
    decode_active[core_id].store(STATE_DECODE_ACTIVE, std::memory_order_release);
}

void smt_guardian_end_decode(int core_id)
{
    uint32_t state = decode_active[core_id].load(std::memory_order_relaxed);
    if (state == STATE_ELASTIC_PARKED)
    {
        decode_active[core_id].store(STATE_IDLE, std::memory_order_release);
        futex_wake(&decode_active[core_id], 1);
    }
    else
    {
        decode_active[core_id].store(STATE_IDLE, std::memory_order_release);
    }
}

} // namespace aaedn