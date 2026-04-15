#include "../include/bandwidth.hpp"
#include <cstdint>
#include <immintrin.h>

namespace aaedn
{

static uint32_t precomputed_decode_pause_count = 0;
static uint32_t normal_pause_count = 100;
static bool decode_active = false;
static float ema_alpha = 0.1f;

void bw_init()
{
    precomputed_decode_pause_count = 142;
    normal_pause_count = precomputed_decode_pause_count;
}

void bw_on_decode_start()
{
    decode_active = true;
    normal_pause_count = precomputed_decode_pause_count;
}

void bw_on_decode_end()
{
    decode_active = false;
}

uint32_t bw_get_pause_count()
{
    return decode_active ? precomputed_decode_pause_count : normal_pause_count;
}

void bw_set_pause_count(uint32_t count)
{
    if (!decode_active)
    {
        normal_pause_count = (uint32_t)((float)count * ema_alpha + (float)normal_pause_count * (1.0f - ema_alpha));
        if (normal_pause_count < precomputed_decode_pause_count)
        {
            normal_pause_count = precomputed_decode_pause_count;
        }
    }
}

} // namespace aaedn