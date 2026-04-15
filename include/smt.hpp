#ifndef AAEDN_SMT_HPP
#define AAEDN_SMT_HPP

#include <atomic>
#include <cstdint>

namespace aaedn
{

constexpr uint32_t STATE_IDLE = 0;
constexpr uint32_t STATE_DECODE_ACTIVE = 1;
constexpr uint32_t STATE_ELASTIC_PARKED = 2;
constexpr int NUM_CORES = 4;

alignas(64) extern std::atomic<uint32_t> decode_active[NUM_CORES];

void smt_init();
void smt_elastic_park(int core_id);
void smt_elastic_unpark(int core_id);
void smt_guardian_start_decode(int core_id);
void smt_guardian_end_decode(int core_id);

} // namespace aaedn

#endif