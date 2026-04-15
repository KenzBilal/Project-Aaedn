#include "../include/ffn.hpp"
#include "../include/dequant.hpp"
#include "../include/quant.hpp"
#include <cmath>
#include <cstring>

namespace aaedn
{

static float gate_out_buf[4096];
static float up_out_buf[4096];

static inline float silu_polynomial(float x)
{
    return x / (1.0f + std::exp(-x));
}

void ffn_swiglu(const float* input, float* output, size_t batch_size, size_t hidden_dim, size_t ffn_dim,
                const QuantBlock* gate_w, const QuantBlock* up_w, const QuantBlock* down_w, const uint16_t* gate_scales,
                const uint16_t* up_scales, const uint16_t* down_scales)
{
    if (!gate_w || !up_w || !down_w || !gate_scales || !up_scales || !down_scales)
    {
        for (size_t i = 0; i < hidden_dim; ++i)
            output[i] = 0.0f;
        return;
    }

    float* gate_out = gate_out_buf;
    float* up_out = up_out_buf;

    for (size_t b = 0; b < batch_size; ++b)
    {
        const float* input_ptr = input + b * hidden_dim;

        for (size_t j = 0; j < ffn_dim; ++j)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < hidden_dim; ++i)
            {
                uint8_t val = (gate_w->data[j * hidden_dim / 2 + i / 2] >> ((i % 2) * 4)) & 0x0F;
                float w = (float)val * (float)gate_scales[j / 32];
                sum += input_ptr[i] * w;
            }
            gate_out[j] = sum;
        }

        for (size_t j = 0; j < ffn_dim; ++j)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < hidden_dim; ++i)
            {
                uint8_t val = (up_w->data[j * hidden_dim / 2 + i / 2] >> ((i % 2) * 4)) & 0x0F;
                float w = (float)val * (float)up_scales[j / 32];
                sum += input_ptr[i] * w;
            }
            up_out[j] = sum;
        }

        for (size_t j = 0; j < ffn_dim; ++j)
        {
            gate_out[j] = gate_out[j] * silu_polynomial(gate_out[j]);
        }

        for (size_t i = 0; i < hidden_dim; ++i)
        {
            float sum = 0.0f;
            for (size_t j = 0; j < ffn_dim; ++j)
            {
                uint8_t val = (down_w->data[i * ffn_dim / 2 + j / 2] >> ((j % 2) * 4)) & 0x0F;
                float w = (float)val * (float)down_scales[i / 32];
                sum += gate_out[j] * up_out[j] * w;
            }
            output[b * hidden_dim + i] = sum;
        }
    }
}

} // namespace aaedn