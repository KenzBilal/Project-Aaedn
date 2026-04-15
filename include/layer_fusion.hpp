#ifndef AAEDN_LAYER_FUSION_HPP
#define AAEDN_LAYER_FUSION_HPP

#include <cstddef>
#include <cstdint>

namespace aaedn
{

struct QuantBlock;

void fused_rmsnorm_qkv(const float* input, const float* rms_weight, const QuantBlock* qkv_weight, float* q_output,
                       float* k_output, float* v_output, size_t hidden_dim, size_t n_heads);

} // namespace aaedn

#endif