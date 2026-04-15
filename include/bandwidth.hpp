#ifndef AAEDN_BANDWIDTH_HPP
#define AAEDN_BANDWIDTH_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn
{

void bw_init();
void bw_on_decode_start();
void bw_on_decode_end();
uint32_t bw_get_pause_count();
void bw_set_pause_count(uint32_t count);

} // namespace aaedn

#endif