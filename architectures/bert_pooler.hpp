#pragma once

#include "../../../datatypes/Additive_Share.hpp"
#include "../../../programs/functions/GEMM.hpp"
#include "../../../programs/functions/comparisons.hpp"
#include "../../../programs/functions/exact_truncation.hpp"
#include "../../../programs/functions/transformer/layernorm_mpc.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

// BERT Pooler and two-class task head for the HPMPC arithmetic-share path.
//
// BertBase::pool_and_classify 先 forward 再 classify。
// gather_cls 从 token-pack 的 lane 里取出每个样本的 CLS，只重排 Trio 的两份本地分量。
// private_gemm_fixed 与 bert_ffn::gemm_fixed 是同一套 limb GEMM。
// tanh_piecewise 在 [-2,2) 用 x*(c1+x^2*(c3+c5*x^2))，区间外饱和到 -1 或 +1。
// classify 只做 768→2。两个 logit 的 argmax 留到揭示之后。
// The input uses the global token-pack layout produced by bert_embedding.hpp:
// logical_token=batch*sequence_length+position, then row/lane=divmod(token,lanes).
namespace bert_pooler
{
constexpr int kHidden = 768;
constexpr int kClasses = 2;
constexpr int kLimbBits = FRACTIONAL / 2;
constexpr UINT_TYPE kLimbBase = UINT_TYPE(1) << kLimbBits;
constexpr int kSimdLanes = DATTYPE / BITLENGTH;

// tanh is evaluated securely as
//   -1,                         x < -2
//   c1*x + c3*x^3 + c5*x^5,   -2 <= x < 2
//   +1,                         x >= 2
// with comparisons over secret shares.  The odd least-squares fit used
// 200001 uniformly spaced points on [-2,2]:
//   c1= 0.9581543010268436
//   c3=-0.2126759995022808
//   c5= 0.024075099934863436
// In real arithmetic the polynomial max error on [-2,2] is 0.021277; the
// saturation error outside is at most 1-tanh(2)=0.035973.  Q20 coefficient
// quantization adds < 1.5e-5 on this interval.  This is an explicit approximation
// bound, not a claim about accumulated upstream fixed-point error.
constexpr FLOATTYPE kTanhBoundary = FLOATTYPE(2);
constexpr FLOATTYPE kTanhC1 = FLOATTYPE(0.9581543010268436);
constexpr FLOATTYPE kTanhC3 = FLOATTYPE(-0.2126759995022808);
constexpr FLOATTYPE kTanhC5 = FLOATTYPE(0.024075099934863436);

static_assert(BITLENGTH == 32, "BERT Pooler requires a 32-bit fixed-point ring");
static_assert(FRACTIONAL == 20, "BERT Pooler weights and tanh coefficients require Q12.20");
static_assert(TRUNC_APPROACH == 2, "BERT Pooler requires exact truncation");
static_assert(TRUNC_DELAYED == 0, "BERT Pooler truncates immediately after private GEMM");
static_assert(PUBLIC_WEIGHTS == 0, "BERT Pooler/classifier weights must remain private");
static_assert(MODELOWNER >= 0 && MODELOWNER < 3, "MODELOWNER must be 0, 1, or 2");
static_assert(PROTOCOL == 5, "Pooler local lane gathering is audited for Trio (PROTOCOL=5)");

template <typename Share>
using AShare = Additive_Share<DATATYPE, Share>;

namespace detail
{
inline int ceil_div(int numerator, int denominator)
{
    return (numerator + denominator - 1) / denominator;
}

// Same audited adapter as BertAttention's local layout conversion.  Trio
// PRE/LIVE arithmetic shares contain two DATATYPE components; schedule-only
// init shares contain no payload.  Repacking each component is a local SIMD
// permutation and never reconstructs the secret.
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
        auto mutable_lanes = lanes.component[component];
        orthogonalize_arithmetic(mutable_lanes.data(), &raw[component], 1);
    }
    A value;
    std::memcpy(&value, raw.data(), sizeof(value));
    return value;
}

template <typename Share>
void gather_cls(const AShare<Share>* token_layout,
                int batch_size,
                int sequence_length,
                AShare<Share>* cls_layout)
{
    using A = AShare<Share>;
    const int packed_batch = ceil_div(batch_size, kSimdLanes);
    std::fill(cls_layout, cls_layout + static_cast<std::size_t>(packed_batch) * kHidden,
              A(UINT_TYPE(0)));
    if constexpr (kScheduleOnlyShare<A>)
    {
        return;
    }
    else
    {
        // Cache only the source packs containing a CLS token.  Each feature is
        // repacked independently because SIMD lanes are part of both Trio
        // share components.
        for (int output_pack = 0; output_pack < packed_batch; ++output_pack)
            for (int feature = 0; feature < kHidden; ++feature)
            {
                ShareLanes destination{};
                for (int output_lane = 0; output_lane < kSimdLanes; ++output_lane)
                {
                    const int batch = output_pack * kSimdLanes + output_lane;
                    if (batch >= batch_size)
                        continue;
                    const int logical_cls = batch * sequence_length;
                    const int source_pack = logical_cls / kSimdLanes;
                    const int source_lane = logical_cls % kSimdLanes;
                    const ShareLanes source = unpack_share_lanes(
                        token_layout[static_cast<std::size_t>(source_pack) * kHidden + feature]);
                    for (int component = 0; component < 2; ++component)
                        destination.component[component][output_lane] =
                            source.component[component][source_lane];
                }
                cls_layout[static_cast<std::size_t>(output_pack) * kHidden + feature] =
                    pack_share_lanes<A>(destination);
            }
    }
}
}  // namespace detail

inline DATATYPE broadcast_scalar(UINT_TYPE value)
{
    constexpr int lanes = DATTYPE / BITLENGTH;
    alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
    lane_values.fill(value);
    DATATYPE packed{};
    orthogonalize_arithmetic(lane_values.data(), &packed, 1);
    return packed;
}

template <typename Share>
void receive_private_vector(const std::vector<UINT_TYPE>& owner_values,
                            std::vector<AShare<Share>>& destination)
{
    if (owner_values.size() != destination.size())
        throw std::invalid_argument("private Pooler vector size mismatch");
    for (std::size_t i = 0; i < destination.size(); ++i)
    {
        DATATYPE packed{};
#if PARTY == MODELOWNER
        packed = broadcast_scalar(owner_values[i]);
#endif
#if MODELOWNER == 0
        destination[i].template prepare_receive_from<P_0>(packed);
#elif MODELOWNER == 1
        destination[i].template prepare_receive_from<P_1>(packed);
#elif MODELOWNER == 2
        destination[i].template prepare_receive_from<P_2>(packed);
#endif
    }
    Share::communicate();
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
struct PrivateLinear
{
    int output_features;
    int input_features;
    std::vector<AShare<Share>> weight_high;
    std::vector<AShare<Share>> weight_low;
    std::vector<AShare<Share>> bias;

    PrivateLinear(int output_count, int input_count)
        : output_features(output_count), input_features(input_count),
          weight_high(static_cast<std::size_t>(output_count) * input_count),
          weight_low(static_cast<std::size_t>(output_count) * input_count),
          bias(output_count)
    {
        if (output_count <= 0 || input_count <= 0)
            throw std::invalid_argument("private Linear dimensions must be positive");
    }
};

// Split raw Q12.20 matrix weights into the same 10-bit limbs used by the
// verified FFN path, then input-share them from MODELOWNER.  owner_weight is
// row-major [output_features,input_features].  Other parties pass empty raw
// vectors; their zero placeholders are never interpreted as model values.
template <typename Share>
PrivateLinear<Share> share_private_linear(const std::vector<UINT_TYPE>& owner_weight,
                                          const std::vector<UINT_TYPE>& owner_bias,
                                          int output_features,
                                          int input_features)
{
    PrivateLinear<Share> result(output_features, input_features);
    std::vector<UINT_TYPE> high(result.weight_high.size(), UINT_TYPE(0));
    std::vector<UINT_TYPE> low(result.weight_low.size(), UINT_TYPE(0));
    std::vector<UINT_TYPE> bias(result.bias.size(), UINT_TYPE(0));
#if PARTY == MODELOWNER
    if (current_phase == PHASE_LIVE)
    {
        if (owner_weight.size() != high.size() || owner_bias.size() != bias.size())
            throw std::invalid_argument("MODELOWNER Pooler/classifier tensor shape mismatch");
        for (int out = 0; out < output_features; ++out)
            for (int in = 0; in < input_features; ++in)
            {
                const std::size_t source = static_cast<std::size_t>(out) * input_features + in;
                const std::int32_t value = static_cast<std::int32_t>(owner_weight[source]);
                const std::int32_t high_signed = value >> kLimbBits;
                const UINT_TYPE high_raw = static_cast<UINT_TYPE>(high_signed);
                const UINT_TYPE low_raw = static_cast<UINT_TYPE>(value) - high_raw * kLimbBase;
#if USE_CUDA_GEMM > 0
                const std::size_t destination = static_cast<std::size_t>(in) * output_features + out;
#else
                const std::size_t destination = source;
#endif
                high[destination] = high_raw;
                low[destination] = low_raw;
            }
        bias = owner_bias;
    }
#endif
    receive_private_vector<Share>(high, result.weight_high);
    receive_private_vector<Share>(low, result.weight_low);
    receive_private_vector<Share>(bias, result.bias);
    return result;
}

// Compute (input*weight)>>20 while retaining the Q40 high bits in the 32-bit
// ring.  This is intentionally identical to the verified BertFFNBlock limb
// decomposition; weight matrices are already transposed by the loader when
// USE_CUDA_GEMM requires it.
template <typename Share>
void private_gemm_fixed(const AShare<Share>* input,
                        const AShare<Share>* weight_high,
                        const AShare<Share>* weight_low,
                        AShare<Share>* output,
                        int rows,
                        int output_features,
                        int input_features)
{
    using A = AShare<Share>;
    if (input == nullptr || weight_high == nullptr || weight_low == nullptr || output == nullptr)
        throw std::invalid_argument("private_gemm_fixed received a null tensor");
    if (rows <= 0 || output_features <= 0 || input_features <= 0)
        throw std::invalid_argument("private_gemm_fixed dimensions must be positive");

    const int input_elements = rows * input_features;
    const int output_elements = rows * output_features;
    std::vector<A> input_high(input, input + input_elements);
    trunc_exact_in_place(input_high.data(), input_elements, kLimbBits);
    std::vector<A> input_low(input_elements);
    for (int i = 0; i < input_elements; ++i)
        input_low[i] = input[i] - input_high[i].mult_public(kLimbBase);

    std::vector<A> high_high(output_elements, A(UINT_TYPE(0)));
    std::vector<A> high_low(output_elements, A(UINT_TYPE(0)));
    std::vector<A> low_high(output_elements, A(UINT_TYPE(0)));
    prepare_GEMM(input_high.data(), const_cast<A*>(weight_high), high_high.data(),
                 rows, output_features, input_features, false);
    prepare_GEMM(input_high.data(), const_cast<A*>(weight_low), high_low.data(),
                 rows, output_features, input_features, false);
    prepare_GEMM(input_low.data(), const_cast<A*>(weight_high), low_high.data(),
                 rows, output_features, input_features, false);
    Share::communicate();
    complete_GEMM(high_high.data(), rows, output_features);
    complete_GEMM(high_low.data(), rows, output_features);
    complete_GEMM(low_high.data(), rows, output_features);

    for (int i = 0; i < output_elements; ++i)
        high_low[i] = high_low[i] + low_high[i];
    trunc_exact_in_place(high_low.data(), output_elements, kLimbBits);
    for (int i = 0; i < output_elements; ++i)
        output[i] = high_high[i] + high_low[i];
}

template <typename Share>
void tanh_piecewise(const AShare<Share>* input, AShare<Share>* output, int len)
{
    using A = AShare<Share>;
    if (len < 0)
        throw std::invalid_argument("tanh_piecewise length cannot be negative");
    if (len == 0)
        return;
    if (input == nullptr || output == nullptr)
        throw std::invalid_argument("tanh_piecewise received a null tensor");

    // One batched comparison invocation returns x<-2 and x<2 as arithmetic
    // 0/1 shares.  No party obtains a secret interval index.
    std::vector<A> comparisons(2 * len);
    std::vector<A> less(2 * len);
    for (int i = 0; i < len; ++i)
    {
        comparisons[i] = input[i] + A(layernorm_mpc::fixed(kTanhBoundary));
        comparisons[len + i] = input[i] - A(layernorm_mpc::fixed(kTanhBoundary));
    }
    pack_additive<0, BITLENGTH>(comparisons.data(), less.data(), 2 * len,
                                LTZ<0, BITLENGTH, Share, DATATYPE>);

    std::vector<A> inside(len);
    std::vector<A> high(len);
    for (int i = 0; i < len; ++i)
    {
        inside[i] = less[len + i] - less[i];
        high[i] = A(UINT_TYPE(1)) - less[len + i];
    }

    // Mask before polynomial evaluation.  Values outside [-2,2] become zero,
    // preventing unused branches from overflowing on large secret inputs.
    std::vector<A> polynomial_x(len);
    layernorm_mpc::multiply_bit<Share>(inside.data(), input, polynomial_x.data(), len);

    // Odd degree-5 Horner form: x*(c1 + x^2*(c3 + c5*x^2)).
    std::vector<A> x2(len);
    std::vector<A> c5_x2(len);
    std::vector<A> inner(len);
    std::vector<A> x2_inner(len);
    std::vector<A> outer(len);
    std::vector<A> polynomial(len);
    layernorm_mpc::multiply_fixed<Share>(polynomial_x.data(), polynomial_x.data(), x2.data(), len);
    layernorm_mpc::multiply_public_fixed<Share>(
        x2.data(), layernorm_mpc::fixed(kTanhC5), c5_x2.data(), len);
    for (int i = 0; i < len; ++i)
        inner[i] = A(layernorm_mpc::fixed(kTanhC3)) + c5_x2[i];
    layernorm_mpc::multiply_fixed<Share>(x2.data(), inner.data(), x2_inner.data(), len);
    for (int i = 0; i < len; ++i)
        outer[i] = A(layernorm_mpc::fixed(kTanhC1)) + x2_inner[i];
    layernorm_mpc::multiply_fixed<Share>(polynomial_x.data(), outer.data(), polynomial.data(), len);

    for (int i = 0; i < len; ++i)
        output[i] = less[i].mult_public(layernorm_mpc::fixed(FLOATTYPE(-1))) +
                    high[i].mult_public(layernorm_mpc::fixed(FLOATTYPE(1))) + polynomial[i];
}

template <typename Share>
class BertPooler
{
  public:
    using A = AShare<Share>;

    // input shape: [ceil(batch_size*sequence_length/lanes), 768].
    // output shape: [packed_batch, 768], with batch examples in SIMD lanes.
    void forward(const A* input,
                 int batch_size,
                 int sequence_length,
                 const PrivateLinear<Share>& pooler,
                 A* output) const
    {
        if (input == nullptr || output == nullptr)
            throw std::invalid_argument("BERT Pooler received a null tensor");
        if (batch_size <= 0 || sequence_length <= 0)
            throw std::invalid_argument("BERT Pooler dimensions must be positive");
        if (pooler.input_features != kHidden || pooler.output_features != kHidden)
            throw std::invalid_argument("BERT Pooler private Linear must be 768x768");

        const int packed_batch = detail::ceil_div(batch_size, kSimdLanes);
        // CLS tokens occupy non-contiguous token lanes when L is not one.
        // Repack both local Trio share components; no value is reconstructed.
        std::vector<A> cls(static_cast<std::size_t>(packed_batch) * kHidden);
        detail::gather_cls<Share>(input, batch_size, sequence_length, cls.data());

        std::vector<A> projected(cls.size());
        private_gemm_fixed<Share>(cls.data(), pooler.weight_high.data(), pooler.weight_low.data(),
                                  projected.data(), packed_batch, kHidden, kHidden);
        for (int pack = 0; pack < packed_batch; ++pack)
            for (int feature = 0; feature < kHidden; ++feature)
                projected[static_cast<std::size_t>(pack) * kHidden + feature] =
                    projected[static_cast<std::size_t>(pack) * kHidden + feature] + pooler.bias[feature];
        tanh_piecewise<Share>(projected.data(), output, static_cast<int>(projected.size()));
    }

    // The task head is a private 768->2 Linear.  Argmax is intentionally left
    // until after the two logits are reconstructed; no MPC Softmax is needed.
    void classify(const A* pooled,
                  int packed_batch,
                  const PrivateLinear<Share>& classifier,
                  A* logits) const
    {
        if (pooled == nullptr || logits == nullptr)
            throw std::invalid_argument("BERT classifier received a null tensor");
        if (packed_batch <= 0)
            throw std::invalid_argument("BERT classifier packed_batch must be positive");
        if (classifier.input_features != kHidden || classifier.output_features != kClasses)
            throw std::invalid_argument("BERT classifier private Linear must be 768x2");

        private_gemm_fixed<Share>(pooled, classifier.weight_high.data(), classifier.weight_low.data(),
                                  logits, packed_batch, kClasses, kHidden);
        for (int pack = 0; pack < packed_batch; ++pack)
            for (int output_feature = 0; output_feature < kClasses; ++output_feature)
                logits[static_cast<std::size_t>(pack) * kClasses + output_feature] =
                    logits[static_cast<std::size_t>(pack) * kClasses + output_feature] +
                    classifier.bias[output_feature];
    }
};
}  // namespace bert_pooler
