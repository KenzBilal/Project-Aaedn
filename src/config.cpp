#include "../include/config.hpp"
#include <cstring>

namespace aaedn
{

bool parse_profile(const char* value, RuntimeProfile* out)
{
    if (!value || !out)
        return false;
    if (std::strcmp(value, "safe") == 0)
    {
        *out = RuntimeProfile::SAFE;
        return true;
    }
    if (std::strcmp(value, "balanced") == 0)
    {
        *out = RuntimeProfile::BALANCED;
        return true;
    }
    if (std::strcmp(value, "max") == 0)
    {
        *out = RuntimeProfile::MAX;
        return true;
    }
    return false;
}

RuntimeConfig config_for_profile(RuntimeProfile profile)
{
    if (profile == RuntimeProfile::SAFE)
        return {profile, 4096ULL * 1024ULL * 1024ULL, 2, 4, 0.0f, 6.0f};
    if (profile == RuntimeProfile::MAX)
        return {profile, 6144ULL * 1024ULL * 1024ULL, 4, 8, 1.0f, 10.0f};
    return {RuntimeProfile::BALANCED, 5632ULL * 1024ULL * 1024ULL, 4, 6, 0.5f, 9.0f};
}

} // namespace aaedn
