#pragma once

// A BertLayer is a library component, not a standalone FUNCTION/FID.  The
// included attention header suppresses bert_ffn.hpp's legacy FID=202 default
// when no enclosing translation unit owns FUNCTION.
#include "bert_attention.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bert_layer
{
constexpr int kHidden = bert_attention::kHidden;
constexpr int kSimdLanes = DATTYPE / BITLENGTH;

static_assert(kHidden == bert_ffn::kHidden, "attention and FFN hidden sizes must match");
static_assert(BITLENGTH == 32, "BertLayer requires a 32-bit ring");
static_assert(FRACTIONAL == 20, "BertLayer parameters use Q12.20");
static_assert(TRUNC_APPROACH == 2, "BertLayer requires exact truncation");
static_assert(TRUNC_DELAYED == 0, "BertLayer truncates every secure product immediately");
static_assert(PUBLIC_WEIGHTS == 0, "BertLayer currently uses private model parameters");

template <typename Share>
using AShare = Additive_Share<DATATYPE, Share>;

// Non-owning view over one HuggingFace BertLayer.  Linear matrices use the
// two-limb backend representation documented by bert_ffn::gemm_fixed.  All
// biases and both LayerNorm affine parameter pairs remain private Q12.20
// arithmetic shares.
template <typename Share>
struct BertLayerWeights
{
    using A = AShare<Share>;

    bert_attention::WeightsView<Share> attention;
    const A* attention_layernorm_gamma = nullptr;  // [768]
    const A* attention_layernorm_beta = nullptr;   // [768]

    const A* intermediate_weight_high = nullptr;   // [3072,768], backend layout
    const A* intermediate_weight_low = nullptr;
    const A* intermediate_bias = nullptr;          // [3072]
    const A* output_weight_high = nullptr;         // [768,3072], backend layout
    const A* output_weight_low = nullptr;
    const A* output_bias = nullptr;                // [768]
    const A* output_layernorm_gamma = nullptr;     // [768]
    const A* output_layernorm_beta = nullptr;      // [768]
};

namespace detail
{
inline int packed_token_rows(int batch, int sequence_length)
{
    return (batch * sequence_length + kSimdLanes - 1) / kSimdLanes;
}

template <typename Share>
void validate_weights(const BertLayerWeights<Share>& weights)
{
    const auto& attention = weights.attention;
    if (attention.qkv_weight_high == nullptr || attention.qkv_weight_low == nullptr ||
        attention.qkv_bias == nullptr || attention.out_weight_high == nullptr ||
        attention.out_weight_low == nullptr || attention.out_bias == nullptr ||
        weights.attention_layernorm_gamma == nullptr ||
        weights.attention_layernorm_beta == nullptr ||
        weights.intermediate_weight_high == nullptr ||
        weights.intermediate_weight_low == nullptr || weights.intermediate_bias == nullptr ||
        weights.output_weight_high == nullptr || weights.output_weight_low == nullptr ||
        weights.output_bias == nullptr || weights.output_layernorm_gamma == nullptr ||
        weights.output_layernorm_beta == nullptr)
        throw std::invalid_argument("BertLayer received a null model-parameter pointer");
}

// Stage-2 LayerNorm exposes public gamma/beta.  BERT parameters are private,
// so first run the exact same online normalization with identity affine, then
// apply secret gamma in one Q20 multiplication round and add secret beta.
template <typename Share>
void private_layernorm(const AShare<Share>* input,
                       const AShare<Share>* gamma,
                       const AShare<Share>* beta,
                       AShare<Share>* output,
                       int packed_rows)
{
    using A = AShare<Share>;
    if (input == nullptr || gamma == nullptr || beta == nullptr || output == nullptr || packed_rows <= 0)
        throw std::invalid_argument("private_layernorm received an invalid tensor or shape");

    const std::vector<UINT_TYPE> identity_gamma(
        kHidden, layernorm_mpc::fixed(FLOATTYPE(1)));
    const std::vector<UINT_TYPE> zero_beta(
        kHidden, layernorm_mpc::fixed(FLOATTYPE(0)));
    const std::size_t elements = static_cast<std::size_t>(packed_rows) * kHidden;
    std::vector<A> normalized(elements);
    layernorm_mpc::layernorm_mpc_forward<Share>(
        input, identity_gamma, zero_beta, normalized.data(), packed_rows, kHidden);

    std::vector<A> broadcast_gamma(elements);
    for (int row = 0; row < packed_rows; ++row)
        for (int feature = 0; feature < kHidden; ++feature)
            broadcast_gamma[static_cast<std::size_t>(row) * kHidden + feature] = gamma[feature];
    layernorm_mpc::multiply_fixed<Share>(normalized.data(), broadcast_gamma.data(), output,
                                         static_cast<int>(elements));
    for (int row = 0; row < packed_rows; ++row)
        for (int feature = 0; feature < kHidden; ++feature)
        {
            const std::size_t index = static_cast<std::size_t>(row) * kHidden + feature;
            output[index] = output[index] + beta[feature];
        }
}
}  // namespace detail

// Original BERT post-LN order, matching HuggingFace BertLayer in eval mode.
// Dropout is absent.  BertAttentionBlock stops at the output projection, and
// BertFFNBlock is called with include_residual_layernorm=false, so each
// residual and each private-affine LayerNorm happens exactly once below.
//   attention projection -> residual(input) -> LN1
//   FFN Linear/GELU/Linear -> residual(LN1) -> LN2.
//
// Physical input/output layout is
//   [ceil(batch*sequence_length/(DATTYPE/BITLENGTH)), 768].
// A lane denotes logical_token = batch_index*sequence_length + position, so
// the complete layer preserves the same token-contiguous packing convention
// as BertAttentionBlock and BertFFNBlock.
template <typename Share>
class BertLayerBlock
{
  public:
    using A = AShare<Share>;

    void forward(const A* input,
                 A* output,
                 const BertLayerWeights<Share>& weights,
                 int batch,
                 int sequence_length,
                 const std::uint8_t* key_padding_mask = nullptr) const
    {
        if (input == nullptr || output == nullptr)
            throw std::invalid_argument("BertLayer received a null input/output pointer");
        if (batch <= 0 || sequence_length <= 0 ||
            sequence_length > bert_attention::kMaxSequenceLength)
            throw std::invalid_argument("BertLayer expects batch>0 and 1<=sequence_length<=128");
        detail::validate_weights(weights);

        const int packed_rows = detail::packed_token_rows(batch, sequence_length);
        const std::size_t elements = static_cast<std::size_t>(packed_rows) * kHidden;

        // Attention owns self-attention and its output dense only.  Keeping
        // residual/LN here prevents either operation from being applied twice.
        std::vector<A> attention_projection(elements);
        bert_attention::BertAttentionBlock<Share>().forward(
            input, attention_projection.data(), weights.attention,
            batch, sequence_length, key_padding_mask);

        std::vector<A> attention_residual(elements);
        for (std::size_t i = 0; i < elements; ++i)
            attention_residual[i] = input[i] + attention_projection[i];
        std::vector<A> attention_normalized(elements);
        detail::private_layernorm<Share>(
            attention_residual.data(), weights.attention_layernorm_gamma,
            weights.attention_layernorm_beta, attention_normalized.data(), packed_rows);

        // Reuse FID=202's verified Linear/GELU/Linear implementation, while
        // explicitly disabling that standalone block's residual and LN.  The
        // complete BertLayer applies its own second residual/LN below.
        std::vector<A> ffn_projection(elements);
        bert_ffn::BertFFNBlock<Share>().forward(
            attention_normalized.data(), packed_rows,
            weights.intermediate_weight_high, weights.intermediate_weight_low,
            weights.intermediate_bias, weights.output_weight_high,
            weights.output_weight_low, weights.output_bias,
            weights.output_layernorm_gamma, weights.output_layernorm_beta,
            ffn_projection.data(), false);

        std::vector<A> ffn_residual(elements);
        for (std::size_t i = 0; i < elements; ++i)
            ffn_residual[i] = attention_normalized[i] + ffn_projection[i];
        detail::private_layernorm<Share>(
            ffn_residual.data(), weights.output_layernorm_gamma,
            weights.output_layernorm_beta, output, packed_rows);
    }
};

template <typename Share>
void bert_layer_forward(const AShare<Share>* input,
                        AShare<Share>* output,
                        const BertLayerWeights<Share>& weights,
                        int batch,
                        int sequence_length,
                        const std::uint8_t* key_padding_mask = nullptr)
{
    BertLayerBlock<Share>().forward(
        input, output, weights, batch, sequence_length, key_padding_mask);
}
}  // namespace bert_layer
