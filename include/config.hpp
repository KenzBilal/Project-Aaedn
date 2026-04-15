#ifndef AAEDN_CONFIG_HPP
#define AAEDN_CONFIG_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn
{

enum class RuntimeProfile : uint8_t
{
    SAFE = 0,
    BALANCED = 1,
    MAX = 2
};

struct RuntimeConfig
{
    RuntimeProfile profile;
    size_t arena_size_bytes;
    uint32_t guardian_threads;
    uint32_t elastic_threads;
    float quant_aggressiveness;
    float bandwidth_cap_gbps;
};

bool parse_profile(const char* value, RuntimeProfile* out);
RuntimeConfig config_for_profile(RuntimeProfile profile);

} // namespace aaedn

#endif
