#ifndef AAEDN_QUANT_ORACLE_HPP
#define AAEDN_QUANT_ORACLE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace aaedn
{

enum class QuantPrecision : uint8_t
{
    BIT_4 = 4,
    BIT_3 = 3
};

struct EntropyOracle
{
    uint32_t histogram[256];
    uint32_t sample_count;
    float current_entropy;
    QuantPrecision current_precision;
    float entropy_ema;
};

void oracle_init(EntropyOracle* oracle);
void oracle_update(EntropyOracle* oracle, const float* activations, size_t n);
QuantPrecision oracle_get_precision(EntropyOracle* oracle, uint32_t current_layer, uint32_t total_layers);

} // namespace aaedn

#endif