#pragma once

#include "../../../datatypes/Additive_Share.hpp"
#include "../../../programs/functions/transformer/layernorm_mpc.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// BERT embeddings for the HPMPC arithmetic-share path.
//
// BertBase::embed 调用 BertEmbedding::forward。MODELOWNER 用公开 token id、
// position、token type 在本地把三张表的对应行相加，只分享相加后的行和 γ、β。
// 词表本身不进入协议。随后用 layernorm_mpc_forward(γ=1, β=0)，再秘密乘 γ、加 β。
//
// Public-input contract
// ---------------------
// token_ids and token_type_ids are deliberately plain uint32 vectors.  Every
// party must receive the same public vectors before calling forward(); they
// must never be passed as Additive_Share values.  Under that contract the
// model owner may select rows locally and secret-share only the selected Q20
// values.  No secret-dependent local table lookup (and no ORAM claim) is made.
//
// Tensor layout
// -------------
// Logical tokens use the same flattened layout as BertFFN/BertAttention:
//
//   logical_token = batch_index * sequence_length + position
//   row  = logical_token / (DATTYPE / BITLENGTH)
//   lane = logical_token % (DATTYPE / BITLENGTH)
//
// Features remain contiguous.  Pooler must locally gather the non-contiguous
// CLS lanes; bert_pooler.hpp does that by transposing both Trio share
// components without opening them.
namespace bert_embedding
{
constexpr int kHidden = 768;
constexpr int kDefaultVocabulary = 30522;
constexpr int kDefaultPositions = 512;
constexpr int kDefaultTokenTypes = 2;
constexpr int kSimdLanes = DATTYPE / BITLENGTH;

static_assert(BITLENGTH == 32, "BERT embeddings require a 32-bit fixed-point ring");
static_assert(FRACTIONAL == 20, "BERT embedding weights must use Q12.20");
static_assert(TRUNC_APPROACH == 2, "BERT embedding LayerNorm requires exact truncation");
static_assert(DATTYPE % BITLENGTH == 0, "DATTYPE must contain an integral number of lanes");
static_assert(PUBLIC_WEIGHTS == 0, "BERT embedding tables are private model parameters");
static_assert(MODELOWNER >= 0 && MODELOWNER < 3, "MODELOWNER must be 0, 1, or 2");

template <typename Share>
using AShare = Additive_Share<DATATYPE, Share>;

// These raw Q12.20 values exist only at MODELOWNER during PHASE_LIVE.  Other
// parties may pass the default-empty structure; no full embedding table is
// transmitted.  The exported int32 bit patterns can be copied to UINT_TYPE.
struct OwnerWeights
{
    int vocabulary_size = kDefaultVocabulary;
    int position_count = kDefaultPositions;
    int token_type_count = kDefaultTokenTypes;
    std::vector<UINT_TYPE> word;
    std::vector<UINT_TYPE> position;
    std::vector<UINT_TYPE> token_type;
    std::vector<UINT_TYPE> gamma;
    std::vector<UINT_TYPE> beta;
};

inline int packed_batch_count(int batch_size)
{
    if (batch_size <= 0)
        throw std::invalid_argument("BERT embedding batch_size must be positive");
    return (batch_size + kSimdLanes - 1) / kSimdLanes;
}

inline int packed_token_rows(int batch_size, int sequence_length)
{
    if (sequence_length <= 0)
        throw std::invalid_argument("BERT embedding sequence_length must be positive");
    if (sequence_length > kDefaultPositions)
        throw std::invalid_argument("BERT embedding sequence_length exceeds 512");
    if (batch_size > std::numeric_limits<int>::max() / sequence_length)
        throw std::overflow_error("BERT embedding packed row count overflows int");
    const int logical_tokens = batch_size * sequence_length;
    return (logical_tokens + kSimdLanes - 1) / kSimdLanes;
}

inline void validate_owner_weights(const OwnerWeights& weights)
{
    if (weights.vocabulary_size <= 0 || weights.position_count <= 0 ||
        weights.token_type_count <= 0)
        throw std::invalid_argument("BERT embedding table dimensions must be positive");
    const auto exact_size = [](const std::vector<UINT_TYPE>& values, std::size_t expected,
                               const char* name) {
        if (values.size() != expected)
            throw std::invalid_argument(std::string("unexpected element count for ") + name);
    };
    exact_size(weights.word, static_cast<std::size_t>(weights.vocabulary_size) * kHidden,
               "word embeddings");
    exact_size(weights.position, static_cast<std::size_t>(weights.position_count) * kHidden,
               "position embeddings");
    exact_size(weights.token_type, static_cast<std::size_t>(weights.token_type_count) * kHidden,
               "token-type embeddings");
    exact_size(weights.gamma, kHidden, "embedding LayerNorm gamma");
    exact_size(weights.beta, kHidden, "embedding LayerNorm beta");
}

template <typename Share>
void prepare_private_packed(const std::vector<DATATYPE>& owner_values,
                            std::vector<AShare<Share>>& destination)
{
    if (owner_values.size() != destination.size())
        throw std::invalid_argument("private packed input size mismatch");
    for (std::size_t i = 0; i < destination.size(); ++i)
    {
#if MODELOWNER == 0
        destination[i].template prepare_receive_from<P_0>(owner_values[i]);
#elif MODELOWNER == 1
        destination[i].template prepare_receive_from<P_1>(owner_values[i]);
#elif MODELOWNER == 2
        destination[i].template prepare_receive_from<P_2>(owner_values[i]);
#endif
    }
}

template <typename Share>
void complete_private_packed(std::vector<AShare<Share>>& destination)
{
    for (auto& value : destination)
    {
#if MODELOWNER == 0
        value.template complete_receive_from<P_0>();
#elif MODELOWNER == 1
        value.template complete_receive_from<P_1>();
#elif MODELOWNER == 2
        value.template complete_receive_from<P_2>();
#endif
    }
}

template <typename Share>
class BertEmbedding
{
  public:
    using A = AShare<Share>;

    // Returns the number of physical output rows.  output must have room for
    // packed_token_rows(batch_size, sequence_length) * 768 shares.
    int forward(const std::vector<std::uint32_t>& token_ids,
                const std::vector<std::uint32_t>& token_type_ids,
                int batch_size,
                int sequence_length,
                const OwnerWeights& owner_weights,
                A* output) const
    {
        if (output == nullptr)
            throw std::invalid_argument("BERT embedding output cannot be null");
        const int packed_rows = packed_token_rows(batch_size, sequence_length);
        const std::size_t logical_tokens = static_cast<std::size_t>(batch_size) * sequence_length;
        if (token_ids.size() != logical_tokens)
            throw std::invalid_argument("public token_ids must have shape [batch, sequence]");
        if (!token_type_ids.empty() && token_type_ids.size() != logical_tokens)
            throw std::invalid_argument("public token_type_ids must be empty or [batch, sequence]");

        const std::size_t elements = static_cast<std::size_t>(packed_rows) * kHidden;
        std::vector<DATATYPE> packed_selected(elements);
        std::vector<DATATYPE> packed_affine(2 * kHidden);

#if PARTY == MODELOWNER
        if (current_phase == PHASE_LIVE)
        {
            validate_owner_weights(owner_weights);
            if (sequence_length > owner_weights.position_count)
                throw std::invalid_argument("position embedding table is shorter than sequence_length");

            for (int packed_row = 0; packed_row < packed_rows; ++packed_row)
                for (int feature = 0; feature < kHidden; ++feature)
                {
                    alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, kSimdLanes> lanes{};
                    for (int lane = 0; lane < kSimdLanes; ++lane)
                    {
                        const int logical_token = packed_row * kSimdLanes + lane;
                        if (logical_token >= static_cast<int>(logical_tokens))
                            continue;
                        const int position = logical_token % sequence_length;
                        const std::uint32_t word_id = token_ids[logical_token];
                        const std::uint32_t type_id =
                            token_type_ids.empty() ? 0U : token_type_ids[logical_token];
                        if (word_id >= static_cast<std::uint32_t>(owner_weights.vocabulary_size))
                            throw std::out_of_range("public token id exceeds embedding vocabulary");
                        if (type_id >= static_cast<std::uint32_t>(owner_weights.token_type_count))
                            throw std::out_of_range("public token type id exceeds embedding table");
                        // UINT_TYPE addition intentionally implements Q20 ring addition.
                        lanes[lane] =
                            owner_weights.word[static_cast<std::size_t>(word_id) * kHidden + feature] +
                            owner_weights.position[static_cast<std::size_t>(position) * kHidden + feature] +
                            owner_weights.token_type[static_cast<std::size_t>(type_id) * kHidden + feature];
                    }
                    orthogonalize_arithmetic(lanes.data(),
                                             &packed_selected[static_cast<std::size_t>(packed_row) *
                                                                  kHidden + feature],
                                             1);
                }

            for (int feature = 0; feature < kHidden; ++feature)
            {
                alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, kSimdLanes> gamma_lanes{};
                alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, kSimdLanes> beta_lanes{};
                gamma_lanes.fill(owner_weights.gamma[feature]);
                beta_lanes.fill(owner_weights.beta[feature]);
                orthogonalize_arithmetic(gamma_lanes.data(), &packed_affine[feature], 1);
                orthogonalize_arithmetic(beta_lanes.data(), &packed_affine[kHidden + feature], 1);
            }
        }
#endif

        // Selected rows and private affine parameters share one input round.
        std::vector<A> summed(elements);
        std::vector<A> affine(2 * kHidden);
        prepare_private_packed<Share>(packed_selected, summed);
        prepare_private_packed<Share>(packed_affine, affine);
        Share::communicate();
        complete_private_packed<Share>(summed);
        complete_private_packed<Share>(affine);

        // Stage-2 LayerNorm exposes public affine parameters.  Reuse its
        // normalization with identity affine, then securely apply the private
        // model-owned gamma and beta.
        const std::vector<UINT_TYPE> identity_gamma(
            kHidden, layernorm_mpc::fixed(FLOATTYPE(1)));
        const std::vector<UINT_TYPE> identity_beta(
            kHidden, layernorm_mpc::fixed(FLOATTYPE(0)));
        std::vector<A> normalized(elements);
        layernorm_mpc::layernorm_mpc_forward<Share>(
            summed.data(), identity_gamma, identity_beta, normalized.data(), packed_rows, kHidden);

        std::vector<A> broadcast_gamma(elements);
        for (int row = 0; row < packed_rows; ++row)
            for (int feature = 0; feature < kHidden; ++feature)
                broadcast_gamma[static_cast<std::size_t>(row) * kHidden + feature] = affine[feature];
        layernorm_mpc::multiply_fixed<Share>(normalized.data(), broadcast_gamma.data(), output,
                                             static_cast<int>(elements));
        for (int row = 0; row < packed_rows; ++row)
            for (int feature = 0; feature < kHidden; ++feature)
                output[static_cast<std::size_t>(row) * kHidden + feature] =
                    output[static_cast<std::size_t>(row) * kHidden + feature] + affine[kHidden + feature];
        return packed_rows;
    }
};
}  // namespace bert_embedding
