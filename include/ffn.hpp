#ifndef AAEDN_FFN_HPP
#define AAEDN_FFN_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn
{

struct QuantBlock;

void ffn_swiglu(const float* input, float* output, size_t batch_size, size_t hidden_dim, size_t ffn_dim,
                const QuantBlock* gate_w, const QuantBlock* up_w, const QuantBlock* down_w, const uint16_t* gate_scales,
                const uint16_t* up_scales, const uint16_t* down_scales);

} // namespace aaedn

#endif