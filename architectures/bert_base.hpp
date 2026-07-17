#pragma once

// Streaming BERT-base inference graph for Trio/PIGEON.
//
// This file deliberately owns activations only.  Model tensors are supplied
// one encoder layer at a time by the FID=203 entry point, so a 28 GiB node
// never retains twelve copies of the two-limb private Linear representation.
// Public token/type ids select embedding rows at MODELOWNER; only the selected
// rows and private affine/model parameters enter the MPC graph.

#include "bert_embedding.hpp"
#include "bert_layer.hpp"
#include "bert_pooler.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bert_base
{
constexpr int kLayers = 12;
constexpr int kHidden = 768;
constexpr int kClasses = 2;
constexpr int kMaxSequenceLength = 128;
constexpr int kSimdLanes = DATTYPE / BITLENGTH;

static_assert(kHidden == bert_embedding::kHidden, "embedding hidden size must be 768");
static_assert(kHidden == bert_layer::kHidden, "encoder hidden size must be 768");
static_assert(kHidden == bert_pooler::kHidden, "pooler hidden size must be 768");
static_assert(BITLENGTH == 32, "BERT-base requires a 32-bit fixed-point ring");
static_assert(FRACTIONAL == 20, "BERT-base model files use Q12.20");
static_assert(TRUNC_APPROACH == 2, "BERT-base requires exact truncation");
static_assert(TRUNC_DELAYED == 0, "BERT-base truncates secure products immediately");
static_assert(PUBLIC_WEIGHTS == 0, "FID=203 keeps model weights private");
static_assert(PROTOCOL == 5, "FID=203 is implemented for Trio (PROTOCOL=5)");
static_assert(PROCESS_NUM == 1, "FID=203 streams one unsharded encoder process");
static_assert(SPLIT_ROLES_OFFSET == 0,
              "FID=203 does not support split-role replica constellations");

template <typename Share>
using AShare = Additive_Share<DATATYPE, Share>;

inline int packed_token_rows(int batch_size, int sequence_length)
{
    if (batch_size <= 0 || sequence_length <= 0 || sequence_length > kMaxSequenceLength)
        throw std::invalid_argument(
            "BERT-base expects batch>0 and 1<=sequence_length<=128");
    return (batch_size * sequence_length + kSimdLanes - 1) / kSimdLanes;
}

inline int packed_batch_rows(int batch_size)
{
    if (batch_size <= 0)
        throw std::invalid_argument("BERT-base batch size must be positive");
    return (batch_size + kSimdLanes - 1) / kSimdLanes;
}

template <typename Share>
class BertBase
{
  public:
    using A = AShare<Share>;

    // Embedding lookup uses public ids but shares only the selected summed
    // rows.  The returned layout is [ceil(B*L/lanes),768].
    std::vector<A> embed(const std::vector<std::uint32_t>& token_ids,
                         const std::vector<std::uint32_t>& token_type_ids,
                         int batch_size,
                         int sequence_length,
                         const bert_embedding::OwnerWeights& owner_weights) const
    {
        const int packed_rows = packed_token_rows(batch_size, sequence_length);
        std::vector<A> activation(static_cast<std::size_t>(packed_rows) * kHidden);
        bert_embedding::BertEmbedding<Share>().forward(
            token_ids, token_type_ids, batch_size, sequence_length,
            owner_weights, activation.data());
        return activation;
    }

    // Execute one post-LN BertLayer and replace the previous activation.
    // The swap makes the previous [B,L,768] buffer die before the caller loads
    // the next layer's private weights.
    void forward_layer_in_place(std::vector<A>& activation,
                                const bert_layer::BertLayerWeights<Share>& weights,
                                int batch_size,
                                int sequence_length,
                                const std::uint8_t* key_padding_mask) const
    {
        const std::size_t expected =
            static_cast<std::size_t>(packed_token_rows(batch_size, sequence_length)) * kHidden;
        if (activation.size() != expected)
            throw std::invalid_argument("BERT-base encoder activation has the wrong shape");
        std::vector<A> next(expected);
        bert_layer::BertLayerBlock<Share>().forward(
            activation.data(), next.data(), weights,
            batch_size, sequence_length, key_padding_mask);
        activation.swap(next);
    }

    // Pool CLS, evaluate private pooler+tanh and the selected private task
    // classifier.  Logits remain arithmetic shares in [ceil(B/lanes),2].
    std::vector<A> pool_and_classify(
        const std::vector<A>& activation,
        int batch_size,
        int sequence_length,
        const bert_pooler::PrivateLinear<Share>& pooler,
        const bert_pooler::PrivateLinear<Share>& classifier) const
    {
        const std::size_t expected =
            static_cast<std::size_t>(packed_token_rows(batch_size, sequence_length)) * kHidden;
        if (activation.size() != expected)
            throw std::invalid_argument("BERT-base final activation has the wrong shape");
        const int packed_batch = packed_batch_rows(batch_size);
        std::vector<A> pooled(static_cast<std::size_t>(packed_batch) * kHidden);
        bert_pooler::BertPooler<Share> block;
        block.forward(activation.data(), batch_size, sequence_length,
                      pooler, pooled.data());
        std::vector<A> logits(static_cast<std::size_t>(packed_batch) * kClasses);
        block.classify(pooled.data(), packed_batch, classifier, logits.data());
        return logits;
    }
};
}  // namespace bert_base
