#pragma once

// BertAttention is a submodule and deliberately has no FUNCTION/FID entry.
// bert_ffn.hpp currently supplies the verified private fixed-point Linear
// primitive, but also installs a standalone FUNCTION macro when included by
// itself.  Do not leak that default into a future bert_layer/bert_encoder
// translation unit.
#ifdef FUNCTION
#define HPMPC_BERT_ATTENTION_CALLER_OWNS_FUNCTION 1
#endif
#include "bert_ffn.hpp"
#ifndef HPMPC_BERT_ATTENTION_CALLER_OWNS_FUNCTION
#ifdef FUNCTION
#undef FUNCTION
#endif
#else
#undef HPMPC_BERT_ATTENTION_CALLER_OWNS_FUNCTION
#endif

#include "../../../programs/functions/transformer/softmax_mpc.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace bert_attention
{
constexpr int kHidden = 768;
constexpr int kHeads = 12;
constexpr int kHeadDim = 64;
constexpr int kQkv = 3 * kHidden;
constexpr int kMaxSequenceLength = 128;
constexpr int kLimbBits = FRACTIONAL / 2;
constexpr UINT_TYPE kLimbBase = UINT_TYPE(1) << kLimbBits;
constexpr int kSimdLanes = DATTYPE / BITLENGTH;

static_assert(kHeads * kHeadDim == kHidden, "BERT head geometry must cover hidden_size");
static_assert(BITLENGTH == 32, "BertAttention requires a 32-bit ring");
static_assert(FRACTIONAL == 20, "BertAttention parameters use Q12.20");
static_assert(TRUNC_APPROACH == 2, "BertAttention requires exact truncation");
static_assert(TRUNC_DELAYED == 0, "BertAttention truncates every secure product immediately");
static_assert(PUBLIC_WEIGHTS == 0, "BertAttention currently uses private model parameters");
static_assert(PROTOCOL == 5, "The local SIMD-lane transpose is audited for Trio (PROTOCOL=5)");

template <typename Share>
using AShare = Additive_Share<DATATYPE, Share>;

// Weight matrices use the same representation as bert_ffn::gemm_fixed:
// each private Q20 weight is split into signed high and non-negative low
// 10-bit limbs.  With CUDA enabled the model loader must transpose [out,in]
// to [in,out], exactly as bert_ffn::load_and_share_model does.
//
// The exporter must pre-divide both the Q rows of qkv_weight and the Q slice
// of qkv_bias by sqrt(head_dim)=8.  Doing so before private sharing removes a
// secure truncation over the complete [B,H,L,L] score tensor.  Consequently
// forward() MUST NOT divide the QK product by 8 again.
template <typename Share>
struct WeightsView
{
    using A = AShare<Share>;

    const A* qkv_weight_high = nullptr;  // [2304,768], backend layout described above
    const A* qkv_weight_low = nullptr;
    const A* qkv_bias = nullptr;         // [2304], Q slice already divided by 8
    const A* out_weight_high = nullptr;  // [768,768], backend layout described above
    const A* out_weight_low = nullptr;
    const A* out_bias = nullptr;         // [768]
};

namespace detail
{
inline int ceil_div(int numerator, int denominator)
{
    return (numerator + denominator - 1) / denominator;
}

// init-only schedule shares are empty classes: they carry communication
// counts but no DATATYPE payload.  Trio PRE/LIVE shares carry exactly the two
// local DATATYPE components conventionally named p1/p2.  Those members are
// private in OECL*, so this adapter copies their trivially-copyable object
// representation and transposes each component independently.  No share is
// opened and no communication is introduced by a layout conversion.
template <typename A>
inline constexpr bool kScheduleOnlyShare = sizeof(A) < sizeof(DATATYPE);

struct ShareLanes
{
    std::array<std::array<UINT_TYPE, kSimdLanes>, 2> component{};
};

template <typename A>
ShareLanes unpack_share_lanes(const A& value)
{
    static_assert(std::is_trivially_copyable_v<A>, "Trio Additive_Share must be trivially copyable");
    static_assert(sizeof(A) == 2 * sizeof(DATATYPE),
                  "Trio PRE/LIVE Additive_Share must contain exactly p1 and p2");

    std::array<DATATYPE, 2> raw{};
    std::memcpy(raw.data(), &value, sizeof(value));
    ShareLanes lanes{};
    for (int component = 0; component < 2; ++component)
        unorthogonalize_arithmetic(&raw[component], lanes.component[component].data(), 1);
    return lanes;
}

template <typename A>
A pack_share_lanes(const ShareLanes& lanes)
{
    static_assert(std::is_trivially_copyable_v<A>, "Trio Additive_Share must be trivially copyable");
    static_assert(sizeof(A) == 2 * sizeof(DATATYPE),
                  "Trio PRE/LIVE Additive_Share must contain exactly p1 and p2");

    std::array<DATATYPE, 2> raw{};
    for (int component = 0; component < 2; ++component)
    {
        // hpmpc's orthogonalizer predates const-correct input pointers.
        auto mutable_lanes = lanes.component[component];
        orthogonalize_arithmetic(mutable_lanes.data(), &raw[component], 1);
    }
    A value;
    std::memcpy(&value, raw.data(), sizeof(value));
    return value;
}

template <typename A>
std::vector<ShareLanes> unpack_vector(const A* values, std::size_t count)
{
    std::vector<ShareLanes> unpacked(count);
    for (std::size_t i = 0; i < count; ++i)
        unpacked[i] = unpack_share_lanes(values[i]);
    return unpacked;
}

template <typename Share>
void token_qkv_to_head_layout(const AShare<Share>* qkv,
                              int batch,
                              int sequence_length,
                              std::vector<AShare<Share>>& query,
                              std::vector<AShare<Share>>& key,
                              std::vector<AShare<Share>>& value)
{
    using A = AShare<Share>;
    const int token_packs = ceil_div(batch * sequence_length, kSimdLanes);
    const int head_packs = ceil_div(batch * kHeads, kSimdLanes);
    const std::size_t head_elements =
        static_cast<std::size_t>(head_packs) * sequence_length * kHeadDim;
    query.assign(head_elements, A(UINT_TYPE(0)));
    key.assign(head_elements, A(UINT_TYPE(0)));
    value.assign(head_elements, A(UINT_TYPE(0)));

    if constexpr (kScheduleOnlyShare<A>)
    {
        return;
    }
    else
    {
        const std::size_t qkv_elements = static_cast<std::size_t>(token_packs) * kQkv;
        const std::vector<ShareLanes> source = unpack_vector(qkv, qkv_elements);

        auto repack_projection = [&](int projection_offset, std::vector<A>& destination) {
            for (int head_pack = 0; head_pack < head_packs; ++head_pack)
                for (int position = 0; position < sequence_length; ++position)
                    for (int dim = 0; dim < kHeadDim; ++dim)
                    {
                        ShareLanes destination_lanes{};
                        for (int lane = 0; lane < kSimdLanes; ++lane)
                        {
                            const int batch_head = head_pack * kSimdLanes + lane;
                            if (batch_head >= batch * kHeads)
                                continue;
                            const int batch_index = batch_head / kHeads;
                            const int head = batch_head % kHeads;
                            const int logical_token = batch_index * sequence_length + position;
                            const int token_pack = logical_token / kSimdLanes;
                            const int token_lane = logical_token % kSimdLanes;
                            const int feature = projection_offset + head * kHeadDim + dim;
                            const ShareLanes& source_lanes =
                                source[static_cast<std::size_t>(token_pack) * kQkv + feature];
                            for (int component = 0; component < 2; ++component)
                                destination_lanes.component[component][lane] =
                                    source_lanes.component[component][token_lane];
                        }
                        const std::size_t destination_index =
                            (static_cast<std::size_t>(head_pack) * sequence_length + position) * kHeadDim + dim;
                        destination[destination_index] = pack_share_lanes<A>(destination_lanes);
                    }
        };

        repack_projection(0, query);
        repack_projection(kHidden, key);
        repack_projection(2 * kHidden, value);
    }
}

template <typename Share>
void head_to_token_layout(const AShare<Share>* context,
                          int batch,
                          int sequence_length,
                          AShare<Share>* token_layout)
{
    using A = AShare<Share>;
    const int token_packs = ceil_div(batch * sequence_length, kSimdLanes);
    const int head_packs = ceil_div(batch * kHeads, kSimdLanes);
    const std::size_t output_elements = static_cast<std::size_t>(token_packs) * kHidden;
    std::fill(token_layout, token_layout + output_elements, A(UINT_TYPE(0)));

    if constexpr (kScheduleOnlyShare<A>)
    {
        return;
    }
    else
    {
        const std::size_t context_elements =
            static_cast<std::size_t>(head_packs) * sequence_length * kHeadDim;
        const std::vector<ShareLanes> source = unpack_vector(context, context_elements);

        for (int token_pack = 0; token_pack < token_packs; ++token_pack)
            for (int feature = 0; feature < kHidden; ++feature)
            {
                const int head = feature / kHeadDim;
                const int dim = feature % kHeadDim;
                ShareLanes destination_lanes{};
                for (int lane = 0; lane < kSimdLanes; ++lane)
                {
                    const int logical_token = token_pack * kSimdLanes + lane;
                    if (logical_token >= batch * sequence_length)
                        continue;
                    const int batch_index = logical_token / sequence_length;
                    const int position = logical_token % sequence_length;
                    const int batch_head = batch_index * kHeads + head;
                    const int head_pack = batch_head / kSimdLanes;
                    const int head_lane = batch_head % kSimdLanes;
                    const ShareLanes& source_lanes =
                        source[(static_cast<std::size_t>(head_pack) * sequence_length + position) * kHeadDim + dim];
                    for (int component = 0; component < 2; ++component)
                        destination_lanes.component[component][lane] =
                            source_lanes.component[component][head_lane];
                }
                token_layout[static_cast<std::size_t>(token_pack) * kHidden + feature] =
                    pack_share_lanes<A>(destination_lanes);
            }
    }
}

// Batched dynamic Q20 GEMM.  Existing GEMM consumes A[m,f] and B[p,f] and
// computes C=A*B^T.  Both operands are secret activations here, so both are
// split into 10-bit limbs.  HH + ((HL+LH)>>10) retains Q20 in the 32-bit ring;
// LL is below one raw output LSB per scalar product and is omitted, matching
// the already-tested bert_ffn::gemm_fixed strategy.
// QK^T 与 AV：左右矩阵都是秘密激活，所以两侧都在这里拆 10-bit limb。
// 固定权重的 Linear 已经由模型所有者拆好，走 bert_ffn::gemm_fixed。
// 公式与 gemm_fixed 相同，低位乘低位被省略。groups 是 head-pack 的个数。
template <typename Share>
void gemm_dynamic_fixed(const AShare<Share>* left,
                        const AShare<Share>* right_transposed,
                        AShare<Share>* output,
                        int groups,
                        int rows,
                        int out_features,
                        int in_features)
{
    using A = AShare<Share>;
    if (groups <= 0 || rows <= 0 || out_features <= 0 || in_features <= 0)
        throw std::invalid_argument("dynamic attention GEMM dimensions must be positive");

    const std::size_t left_per_group = static_cast<std::size_t>(rows) * in_features;
    const std::size_t right_per_group = static_cast<std::size_t>(out_features) * in_features;
    const std::size_t output_per_group = static_cast<std::size_t>(rows) * out_features;
    const std::size_t left_count = static_cast<std::size_t>(groups) * left_per_group;
    const std::size_t right_count = static_cast<std::size_t>(groups) * right_per_group;
    const std::size_t output_count = static_cast<std::size_t>(groups) * output_per_group;

    std::vector<A> high(left_count + right_count);
    std::copy(left, left + left_count, high.begin());
    std::copy(right_transposed, right_transposed + right_count,
              high.begin() + static_cast<std::ptrdiff_t>(left_count));
    trunc_exact_in_place(high.data(), static_cast<int>(high.size()), kLimbBits);
    A* left_high = high.data();
    A* right_high = high.data() + static_cast<std::ptrdiff_t>(left_count);

    std::vector<A> left_low(left_count);
    std::vector<A> right_low(right_count);
    for (std::size_t i = 0; i < left_count; ++i)
        left_low[i] = left[i] - left_high[i].mult_public(kLimbBase);
    for (std::size_t i = 0; i < right_count; ++i)
        right_low[i] = right_transposed[i] - right_high[i].mult_public(kLimbBase);

    std::vector<A> high_high(output_count, A(UINT_TYPE(0)));
    std::vector<A> high_low(output_count, A(UINT_TYPE(0)));
    std::vector<A> low_high(output_count, A(UINT_TYPE(0)));
    for (int group = 0; group < groups; ++group)
    {
        const std::size_t left_offset = static_cast<std::size_t>(group) * left_per_group;
        const std::size_t right_offset = static_cast<std::size_t>(group) * right_per_group;
        const std::size_t output_offset = static_cast<std::size_t>(group) * output_per_group;
        prepare_GEMM(left_high + left_offset, right_high + right_offset,
                     high_high.data() + output_offset, rows, out_features, in_features, false);
        prepare_GEMM(left_high + left_offset, right_low.data() + right_offset,
                     high_low.data() + output_offset, rows, out_features, in_features, false);
        prepare_GEMM(left_low.data() + left_offset, right_high + right_offset,
                     low_high.data() + output_offset, rows, out_features, in_features, false);
    }
    Share::communicate();
    for (int group = 0; group < groups; ++group)
    {
        const std::size_t output_offset = static_cast<std::size_t>(group) * output_per_group;
        complete_GEMM(high_high.data() + output_offset, rows, out_features);
        complete_GEMM(high_low.data() + output_offset, rows, out_features);
        complete_GEMM(low_high.data() + output_offset, rows, out_features);
    }

    for (std::size_t i = 0; i < output_count; ++i)
        high_low[i] = high_low[i] + low_high[i];
    trunc_exact_in_place(high_low.data(), static_cast<int>(output_count), kLimbBits);
    for (std::size_t i = 0; i < output_count; ++i)
        output[i] = high_high[i] + high_low[i];
}

template <typename Share>
void add_key_padding_mask(AShare<Share>* scores,
                          int batch,
                          int sequence_length,
                          const std::uint8_t* key_padding_mask)
{
    if (key_padding_mask == nullptr)
        return;

    using A = AShare<Share>;
    const int head_packs = ceil_div(batch * kHeads, kSimdLanes);
    const UINT_TYPE masked_value = transformer_mpc::fixed(FLOATTYPE(-16));
    for (int head_pack = 0; head_pack < head_packs; ++head_pack)
        for (int query = 0; query < sequence_length; ++query)
            for (int key = 0; key < sequence_length; ++key)
            {
                alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, kSimdLanes> lane_values{};
                for (int lane = 0; lane < kSimdLanes; ++lane)
                {
                    const int batch_head = head_pack * kSimdLanes + lane;
                    if (batch_head >= batch * kHeads)
                        continue;
                    const int batch_index = batch_head / kHeads;
                    const std::uint8_t keep =
                        key_padding_mask[static_cast<std::size_t>(batch_index) * sequence_length + key];
                    if (keep > 1)
                        throw std::invalid_argument("key_padding_mask values must be 0 or 1");
                    if (keep == 0)
                        lane_values[lane] = masked_value;
                }
                DATATYPE packed_mask{};
                orthogonalize_arithmetic(lane_values.data(), &packed_mask, 1);
                const std::size_t index =
                    (static_cast<std::size_t>(head_pack) * sequence_length + query) * sequence_length + key;
                scores[index] = scores[index] + A::get_share_from_public_dat(packed_mask);
            }
}
}  // namespace detail

// Multi-head attention，模块边界止于输出投影。残差和 LayerNorm 在 bert_layer.hpp。
// 步骤与 BertAttentionBlock::forward 里的编号一致：
//   1. bert_ffn::gemm_fixed 做联合 QKV。Q 权重和 bias 已除以 8，这里不再缩放。
//   2. token_qkv_to_head_layout 本地重排两份 Trio 分量，无通信。
//   3. gemm_dynamic_fixed 算 QK^T。CUDA 路径把 K 从 [sequence,head_dim] 转成 [head_dim,sequence]。
//   4. add_key_padding_mask 加公开 -16。Q12.20 表示不了 -10000，且指数在约 -11 以下已为 0。
//   5. softmax_mpc_forward。
//   6. gemm_dynamic_fixed 算 AV。CPU 需要转置 V；CUDA 直接使用现有 [sequence,head_dim]。
//   7. head_to_token_layout 后，gemm_fixed 做输出投影。
//
// Input/output physical layout is FlexNN's token-packed matrix:
//   [ceil(batch*sequence_length/(DATTYPE/BITLENGTH)), 768]
// and lane r represents logical token row r.  Q/K/V are locally converted to
// a head-packed layout whose lanes represent (batch,head).  This is essential:
// merely reshaping the vector indices would mix token lanes during QK/AV GEMM.
//
// key_padding_mask is public [batch,sequence_length], with 1=keep and 0=mask.
// Masked scores receive -16.0 (Q20), which is representable in the 32-bit ring
// and lies below softmax_mpc's clipped exponential tail.  The common -10000
// sentinel is intentionally not used because Q12.20 can only represent
// approximately [-2048,2048).
template <typename Share>
class BertAttentionBlock
{
  public:
    using A = AShare<Share>;

    void forward(const A* input,
                 A* output,
                 const WeightsView<Share>& weights,
                 int batch,
                 int sequence_length,
                 const std::uint8_t* key_padding_mask = nullptr) const
    {
        if (input == nullptr || output == nullptr || weights.qkv_weight_high == nullptr ||
            weights.qkv_weight_low == nullptr || weights.qkv_bias == nullptr ||
            weights.out_weight_high == nullptr || weights.out_weight_low == nullptr ||
            weights.out_bias == nullptr)
            throw std::invalid_argument("BertAttention received a null tensor/weight pointer");
        if (batch <= 0 || sequence_length <= 0 || sequence_length > kMaxSequenceLength)
            throw std::invalid_argument("BertAttention expects batch>0 and 1<=sequence_length<=128");

        const int token_packs = detail::ceil_div(batch * sequence_length, kSimdLanes);
        const int head_packs = detail::ceil_div(batch * kHeads, kSimdLanes);

        // 1. Joint private QKV projection.  The Q slice is already /8 by the
        // exporter, including its bias, so the following QK result has the
        // required 1/sqrt(64) scale without another truncation.
        std::vector<A> qkv(static_cast<std::size_t>(token_packs) * kQkv, A(UINT_TYPE(0)));
        bert_ffn::gemm_fixed<Share>(input, weights.qkv_weight_high, weights.qkv_weight_low,
                                    qkv.data(), token_packs, kQkv, kHidden);
        for (int row = 0; row < token_packs; ++row)
            for (int feature = 0; feature < kQkv; ++feature)
                qkv[static_cast<std::size_t>(row) * kQkv + feature] =
                    qkv[static_cast<std::size_t>(row) * kQkv + feature] + weights.qkv_bias[feature];

        // [token-pack, Q|K|V] -> [head-pack, sequence, head_dim].
        std::vector<A> query;
        std::vector<A> key;
        std::vector<A> value;
        detail::token_qkv_to_head_layout<Share>(qkv.data(), batch, sequence_length, query, key, value);
        qkv.clear();
        qkv.shrink_to_fit();

        // 2. QK^T.  The CPU GEMM consumes its right operand as logical
        // [out,in], whereas the CUDA backend consumes the same matrix in the
        // transposed physical layout [in,out] (the private-weight loader in
        // bert_ffn follows this convention too).  K is naturally
        // [sequence,head_dim], so only the CUDA path needs a local transpose.
        const std::size_t score_count =
            static_cast<std::size_t>(head_packs) * sequence_length * sequence_length;
        std::vector<A> scores(score_count, A(UINT_TYPE(0)));
        std::vector<A> key_backend;
        const A* key_operand = key.data();
#if USE_CUDA_GEMM > 0
        key_backend.resize(key.size());
        for (int head_pack = 0; head_pack < head_packs; ++head_pack)
            for (int position = 0; position < sequence_length; ++position)
                for (int dim = 0; dim < kHeadDim; ++dim)
                    key_backend[(static_cast<std::size_t>(head_pack) * kHeadDim + dim) *
                                    sequence_length + position] =
                        key[(static_cast<std::size_t>(head_pack) * sequence_length + position) *
                                kHeadDim + dim];
        key_operand = key_backend.data();
#endif
        detail::gemm_dynamic_fixed<Share>(query.data(), key_operand, scores.data(),
                                          head_packs, sequence_length, sequence_length, kHeadDim);
        query.clear();
        key.clear();
        key_backend.clear();

        // 3. Public key-padding mask; no MPC round.
        detail::add_key_padding_mask<Share>(scores.data(), batch, sequence_length, key_padding_mask);

        // 4. The verified arithmetic API performs its own A->B comparisons for
        // row max and returns arithmetic shares.  No extra Usuba conversion is
        // needed at this call site.
        std::vector<A> probabilities(score_count);
        transformer_mpc::softmax_mpc_forward<Share>(scores.data(), probabilities.data(),
                                                     head_packs * sequence_length, sequence_length);
        scores.clear();
        scores.shrink_to_fit();

        // 5. A*V.  The logical right operand is V^T=[head_dim,sequence].
        // CPU GEMM therefore needs a materialized V transpose.  CUDA expects
        // that logical matrix in [in,out] physical layout, which is exactly
        // V's existing [sequence,head_dim] storage and must not be transposed.
        const std::size_t value_count =
            static_cast<std::size_t>(head_packs) * sequence_length * kHeadDim;
        std::vector<A> value_backend;
        const A* value_operand = value.data();
#if USE_CUDA_GEMM == 0
        value_backend.resize(value_count);
        for (int head_pack = 0; head_pack < head_packs; ++head_pack)
            for (int position = 0; position < sequence_length; ++position)
                for (int dim = 0; dim < kHeadDim; ++dim)
                    value_backend[(static_cast<std::size_t>(head_pack) * kHeadDim + dim) *
                                      sequence_length + position] =
                        value[(static_cast<std::size_t>(head_pack) * sequence_length + position) * kHeadDim + dim];
        value_operand = value_backend.data();
#endif

        std::vector<A> context(value_count, A(UINT_TYPE(0)));
        detail::gemm_dynamic_fixed<Share>(probabilities.data(), value_operand, context.data(),
                                          head_packs, sequence_length, kHeadDim, sequence_length);
        probabilities.clear();
        probabilities.shrink_to_fit();
        value.clear();
        value.shrink_to_fit();
        value_backend.clear();
        value_backend.shrink_to_fit();

        // [head-pack,sequence,head_dim] -> [token-pack,hidden].
        std::vector<A> token_context(static_cast<std::size_t>(token_packs) * kHidden);
        detail::head_to_token_layout<Share>(context.data(), batch, sequence_length, token_context.data());
        context.clear();
        context.shrink_to_fit();

        // 6. Private output projection.  Residual and LayerNorm belong to
        // BertLayer, not this submodule.
        bert_ffn::gemm_fixed<Share>(token_context.data(), weights.out_weight_high,
                                    weights.out_weight_low, output, token_packs, kHidden, kHidden);
        for (int row = 0; row < token_packs; ++row)
            for (int feature = 0; feature < kHidden; ++feature)
                output[static_cast<std::size_t>(row) * kHidden + feature] =
                    output[static_cast<std::size_t>(row) * kHidden + feature] + weights.out_bias[feature];
    }
};
}  // namespace bert_attention
