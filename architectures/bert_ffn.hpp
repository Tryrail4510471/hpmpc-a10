#pragma once

#define RESULTTYPE DATATYPE
#ifndef FUNCTION
#define FUNCTION bert_ffn_program
#endif

#include "../../../datatypes/Additive_Share.hpp"
#include "../../../programs/functions/GEMM.hpp"
#include "../../../programs/functions/exact_truncation.hpp"
#include "../../../programs/functions/transformer/gelu_mpc.hpp"
#include "../../../programs/functions/transformer/layernorm_mpc.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bert_ffn
{
constexpr int kHidden = 768;
constexpr int kIntermediate = 3072;
constexpr int kLimbBits = FRACTIONAL / 2;
constexpr UINT_TYPE kLimbBase = UINT_TYPE(1) << kLimbBits;
constexpr std::uint32_t kParameterCount =
    kIntermediate * kHidden + kIntermediate + kHidden * kIntermediate + kHidden + kHidden + kHidden;

template <typename Share>
using AShare = Additive_Share<DATATYPE, Share>;

// Compute (A*W)>>FRACTIONAL without losing the Q40 high bits in the 32-bit
// ring. W is pre-split by the model owner into 10-bit limbs. The low*low term
// is below one output LSB per scalar product and is intentionally omitted.
//
// (a*b)>>20 = ah*bh + ((ah*bl + al*bh)>>10).  Attention 的 QKV、输出投影和
// FFN 的两层 Linear 都走这里。权重在 load_and_share_model 里已经拆好；激活在函数内
// 用 trunc_exact_in_place 拆 high。prepare_GEMM 在 USE_CUDA_GEMM>0 时把本地整数乘
// 交给 CUTLASS，mask_and_send 与截断仍留在 CPU。两边都是秘密矩阵时用
// bert_attention::detail::gemm_dynamic_fixed，而不是本函数。
template <typename Share>
void gemm_fixed(const AShare<Share>* input,
                const AShare<Share>* weight_high,
                const AShare<Share>* weight_low,
                AShare<Share>* output,
                int rows,
                int out_features,
                int in_features)
{
    using A = AShare<Share>;
    const int input_elements = rows * in_features;
    const int output_elements = rows * out_features;
    std::vector<A> input_high(input, input + input_elements);
    trunc_exact_in_place(input_high.data(), input_elements, kLimbBits);
    std::vector<A> input_low(input_elements);
    for (int i = 0; i < input_elements; ++i)
        input_low[i] = input[i] - input_high[i].mult_public(kLimbBase);

    std::vector<A> high_high(output_elements, A(UINT_TYPE(0)));
    std::vector<A> high_low(output_elements, A(UINT_TYPE(0)));
    std::vector<A> low_high(output_elements, A(UINT_TYPE(0)));
    prepare_GEMM(input_high.data(), const_cast<A*>(weight_high), high_high.data(),
                 rows, out_features, in_features, false);
    prepare_GEMM(input_high.data(), const_cast<A*>(weight_low), high_low.data(),
                 rows, out_features, in_features, false);
    prepare_GEMM(input_low.data(), const_cast<A*>(weight_high), low_high.data(),
                 rows, out_features, in_features, false);
    Share::communicate();
    complete_GEMM(high_high.data(), rows, out_features);
    complete_GEMM(high_low.data(), rows, out_features);
    complete_GEMM(low_high.data(), rows, out_features);

    for (int i = 0; i < output_elements; ++i)
        high_low[i] = high_low[i] + low_high[i];
    trunc_exact_in_place(high_low.data(), output_elements, kLimbBits);
    for (int i = 0; i < output_elements; ++i)
        output[i] = high_high[i] + high_low[i];
}

static_assert(BITLENGTH == 32, "BERT FFN currently requires a 32-bit fixed-point ring");
static_assert(FRACTIONAL == 20, "BERT FFN weights are exported as Q12.20");
static_assert(TRUNC_APPROACH == 2, "BERT FFN requires exact truncation (TRUNC_APPROACH=2)");
static_assert(TRUNC_DELAYED == 0, "BERT FFN truncates after each GEMM before bias/residual addition");
static_assert(PUBLIC_WEIGHTS == 0, "BERT FFN currently treats model parameters as private shares");
static_assert(MODELOWNER >= 0 && MODELOWNER < 3, "BERT FFN MODELOWNER must be 0, 1, or 2");
static_assert(DATAOWNER >= 0 && DATAOWNER < 3, "BERT FFN DATAOWNER must be 0, 1, or 2");

inline std::string env_or_default(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value == nullptr || *value == '\0' ? std::string(fallback) : std::string(value);
}

inline int positive_env_int(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        return fallback;
    const int parsed = std::stoi(value);
    if (parsed <= 0)
        throw std::invalid_argument(std::string(name) + " must be positive");
    return parsed;
}

inline std::vector<UINT_TYPE> read_counted_q20(const std::string& path, std::uint32_t expected)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open Q20 tensor file: " + path);
    std::uint32_t count = 0;
    stream.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!stream || count != expected)
        throw std::runtime_error("unexpected element count in " + path);
    std::vector<std::int32_t> signed_values(count);
    stream.read(reinterpret_cast<char*>(signed_values.data()),
                static_cast<std::streamsize>(count) * sizeof(std::int32_t));
    if (!stream)
        throw std::runtime_error("truncated Q20 tensor file: " + path);
    std::vector<UINT_TYPE> values(count);
    for (std::uint32_t i = 0; i < count; ++i)
        values[i] = static_cast<UINT_TYPE>(signed_values[i]);
    return values;
}

inline void write_counted_q20(const std::string& path, const std::vector<std::int32_t>& values)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
        throw std::runtime_error("cannot create FFN output file: " + path);
    const std::uint32_t count = static_cast<std::uint32_t>(values.size());
    stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size()) * sizeof(std::int32_t));
    if (!stream)
        throw std::runtime_error("failed to write FFN output file: " + path);
}

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
void receive_private_vector(const std::vector<UINT_TYPE>& owner_values, std::vector<AShare<Share>>& destination)
{
    if (owner_values.size() != destination.size())
        throw std::invalid_argument("private vector size mismatch");
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

// FlexNN represents [batch, sequence, feature] as the row-major matrix
// [batch*sequence, feature]. Each physical row below additionally SIMD-packs
// DATTYPE/BITLENGTH logical token rows, so no transpose or materialized reshape
// is required before either GEMM.
template <typename Share>
class BertFFNBlock
{
  public:
    using A = AShare<Share>;

    void forward(const A* input,
                 int packed_rows,
                 const A* w1_high,
                 const A* w1_low,
                 const A* b1,
                 const A* w2_high,
                 const A* w2_low,
                 const A* b2,
                 const A* gamma,
                 const A* beta,
                 A* output,
                 bool include_residual_layernorm = true) const
    {
        if (packed_rows <= 0)
            throw std::invalid_argument("BERT FFN needs at least one packed row");

        start_layer_stats("Linear768x3072", 2);
        std::vector<A> intermediate(static_cast<std::size_t>(packed_rows) * kIntermediate, A(UINT_TYPE(0)));
        gemm_fixed<Share>(input, w1_high, w1_low, intermediate.data(),
                          packed_rows, kIntermediate, kHidden);
        for (int row = 0; row < packed_rows; ++row)
            for (int col = 0; col < kIntermediate; ++col)
                intermediate[static_cast<std::size_t>(row) * kIntermediate + col] =
                    intermediate[static_cast<std::size_t>(row) * kIntermediate + col] + b1[col];
        stop_layer_stats(2);

        start_layer_stats("GELU", 3);
        std::vector<A> activated(intermediate.size());
        gelu_mpc::GELU<Share>(intermediate.data(), intermediate.data() + intermediate.size(), activated.data());
        stop_layer_stats(3);

        start_layer_stats("Linear3072x768", 4);
        std::vector<A> projected(static_cast<std::size_t>(packed_rows) * kHidden, A(UINT_TYPE(0)));
        gemm_fixed<Share>(activated.data(), w2_high, w2_low, projected.data(),
                          packed_rows, kHidden, kIntermediate);
        for (int row = 0; row < packed_rows; ++row)
            for (int col = 0; col < kHidden; ++col)
            {
                const std::size_t index = static_cast<std::size_t>(row) * kHidden + col;
                projected[index] = projected[index] + b2[col];
            }
        stop_layer_stats(4);

        start_layer_stats("Residual", 5);
        if (include_residual_layernorm)
            for (int row = 0; row < packed_rows; ++row)
                for (int col = 0; col < kHidden; ++col)
                {
                    const std::size_t index = static_cast<std::size_t>(row) * kHidden + col;
                // Both operands are already Q12.20. Truncating this sum would
                // incorrectly divide it by 2^20; truncation belongs immediately
                // after each Q20*Q20 GEMM, as done above.
                    projected[index] = projected[index] + input[index];
                }
        stop_layer_stats(5);

        // Reuse Stage-2 LayerNorm for normalization. Its public affine is set
        // to identity, then private gamma/beta are applied below.
        start_layer_stats("LayerNorm", 6);
        const std::vector<UINT_TYPE> unit_gamma(kHidden, layernorm_mpc::fixed(FLOATTYPE(1)));
        const std::vector<UINT_TYPE> zero_beta(kHidden, layernorm_mpc::fixed(FLOATTYPE(0)));
        std::vector<A> normalized(projected.size());
        if (include_residual_layernorm)
            layernorm_mpc::layernorm_mpc_forward<Share>(projected.data(), unit_gamma, zero_beta,
                                                        normalized.data(), packed_rows, kHidden);
        stop_layer_stats(6);

        start_layer_stats("LayerNormAffine", 7);
        std::vector<A> broadcast_gamma(projected.size());
        if (include_residual_layernorm)
        {
            for (int row = 0; row < packed_rows; ++row)
                for (int col = 0; col < kHidden; ++col)
                    broadcast_gamma[static_cast<std::size_t>(row) * kHidden + col] = gamma[col];
            layernorm_mpc::multiply_fixed<Share>(normalized.data(), broadcast_gamma.data(), output,
                                                 static_cast<int>(normalized.size()));
            for (int row = 0; row < packed_rows; ++row)
                for (int col = 0; col < kHidden; ++col)
                    output[static_cast<std::size_t>(row) * kHidden + col] =
                        output[static_cast<std::size_t>(row) * kHidden + col] + beta[col];
        }
        else
            std::copy(projected.begin(), projected.end(), output);
        stop_layer_stats(7);
    }
};

struct ModelSharesBase
{
    static constexpr std::size_t w1_count = static_cast<std::size_t>(kIntermediate) * kHidden;
    static constexpr std::size_t b1_count = kIntermediate;
    static constexpr std::size_t w2_count = static_cast<std::size_t>(kHidden) * kIntermediate;
    static constexpr std::size_t b2_count = kHidden;
    static constexpr std::size_t gamma_count = kHidden;
    static constexpr std::size_t beta_count = kHidden;
};

template <typename Share>
struct ModelShares : ModelSharesBase
{
    std::vector<AShare<Share>> w1_high;
    std::vector<AShare<Share>> w1_low;
    std::vector<AShare<Share>> b1;
    std::vector<AShare<Share>> w2_high;
    std::vector<AShare<Share>> w2_low;
    std::vector<AShare<Share>> b2;
    std::vector<AShare<Share>> gamma;
    std::vector<AShare<Share>> beta;

    ModelShares()
        : w1_high(w1_count), w1_low(w1_count), b1(b1_count),
          w2_high(w2_count), w2_low(w2_count), b2(b2_count), gamma(gamma_count), beta(beta_count)
    {
    }
};

template <typename Share>
ModelShares<Share> load_and_share_model(const std::string& path)
{
    std::vector<UINT_TYPE> raw(kParameterCount, UINT_TYPE(0));
#if PARTY == MODELOWNER
    if (current_phase == PHASE_LIVE)
        raw = read_counted_q20(path, kParameterCount);
#endif
    ModelShares<Share> model;
    std::size_t offset = 0;
    auto receive_weight = [&](auto& high_destination, auto& low_destination, int out_features, int in_features) {
        std::vector<UINT_TYPE> high(high_destination.size(), UINT_TYPE(0));
        std::vector<UINT_TYPE> low(low_destination.size(), UINT_TYPE(0));
#if PARTY == MODELOWNER
        for (int out = 0; out < out_features; ++out)
            for (int in = 0; in < in_features; ++in)
            {
                const std::size_t source = offset + static_cast<std::size_t>(out) * in_features + in;
                const std::int32_t value = static_cast<std::int32_t>(raw[source]);
                const std::int32_t high_signed = value >> kLimbBits;
                const UINT_TYPE high_raw = static_cast<UINT_TYPE>(high_signed);
                const UINT_TYPE low_raw = static_cast<UINT_TYPE>(value) - high_raw * kLimbBase;
#if USE_CUDA_GEMM > 0
                const std::size_t destination = static_cast<std::size_t>(in) * out_features + out;
#else
                const std::size_t destination = static_cast<std::size_t>(out) * in_features + in;
#endif
                high[destination] = high_raw;
                low[destination] = low_raw;
            }
#endif
        receive_private_vector<Share>(high, high_destination);
        receive_private_vector<Share>(low, low_destination);
        offset += high_destination.size();
    };
    auto receive_slice = [&](auto& destination) {
        std::vector<UINT_TYPE> slice(destination.size());
#if PARTY == MODELOWNER
        std::copy(raw.begin() + static_cast<std::ptrdiff_t>(offset),
                  raw.begin() + static_cast<std::ptrdiff_t>(offset + destination.size()), slice.begin());
#endif
        receive_private_vector<Share>(slice, destination);
        offset += destination.size();
    };
    receive_weight(model.w1_high, model.w1_low, kIntermediate, kHidden);
    receive_slice(model.b1);
    receive_weight(model.w2_high, model.w2_low, kHidden, kIntermediate);
    receive_slice(model.b2);
    receive_slice(model.gamma);
    receive_slice(model.beta);
    return model;
}
}  // namespace bert_ffn

// FID=202 standalone entry point. Input/output files use the same compact
// counted-Q20 format as the exported parameters: uint32 count followed by
// little-endian int32 values. HPMPC_BERT_FFN_ROWS is the logical B*L row count.
template <typename Share>
void bert_ffn_program(DATATYPE* result)
{
    using A = bert_ffn::AShare<Share>;
    constexpr int lanes = DATTYPE / BITLENGTH;
    const int logical_rows = bert_ffn::positive_env_int("HPMPC_BERT_FFN_ROWS", 1);
    const int total_workers = bert_ffn::positive_env_int("HPMPC_BERT_FFN_TOTAL_WORKERS", 1);
    const int worker = process_offset;
    if (worker < 0 || worker >= total_workers)
        throw std::runtime_error("process_offset is outside HPMPC_BERT_FFN_TOTAL_WORKERS");
    const int worker_row_begin = logical_rows * worker / total_workers;
    const int worker_row_end = logical_rows * (worker + 1) / total_workers;
    const int worker_rows = worker_row_end - worker_row_begin;
    if (worker_rows <= 0)
        throw std::runtime_error("too many BERT FFN workers for the number of logical rows");
    const int packed_rows = (worker_rows + lanes - 1) / lanes;
    const bool include_residual_layernorm =
        bert_ffn::env_or_default("HPMPC_BERT_FFN_STAGE", "full") != "core";
    const std::string model_path = bert_ffn::env_or_default(
        "HPMPC_BERT_FFN_WEIGHTS", "nn/Pygeon/models/pretrained/bert_base_uncased_layer0_ffn_q20.bin");
    const std::string input_path =
        bert_ffn::env_or_default("HPMPC_BERT_FFN_INPUT", "/tmp/hpmpc_bert_ffn_input_q20.bin");
    const std::string output_path =
        bert_ffn::env_or_default("HPMPC_BERT_FFN_OUTPUT", "/tmp/hpmpc_bert_ffn_output_q20.bin");

    std::vector<UINT_TYPE> raw_input(static_cast<std::size_t>(logical_rows) * bert_ffn::kHidden, UINT_TYPE(0));
#if PARTY == DATAOWNER
    if (current_phase == PHASE_LIVE)
        raw_input = bert_ffn::read_counted_q20(input_path, static_cast<std::uint32_t>(raw_input.size()));
#endif
    start_layer_stats("InputShare", 0);
    std::vector<DATATYPE> packed_input(static_cast<std::size_t>(packed_rows) * bert_ffn::kHidden);
#if PARTY == DATAOWNER
    for (int packed_row = 0; packed_row < packed_rows; ++packed_row)
        for (int col = 0; col < bert_ffn::kHidden; ++col)
        {
            alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
            for (int lane = 0; lane < lanes; ++lane)
            {
                const int row = worker_row_begin + packed_row * lanes + lane;
                if (row < worker_row_end)
                    lane_values[lane] = raw_input[static_cast<std::size_t>(row) * bert_ffn::kHidden + col];
            }
            orthogonalize_arithmetic(lane_values.data(),
                                     &packed_input[static_cast<std::size_t>(packed_row) * bert_ffn::kHidden + col], 1);
        }
#endif

    std::vector<A> input(packed_input.size());
    for (std::size_t i = 0; i < input.size(); ++i)
#if DATAOWNER == 0
        input[i].template prepare_receive_from<P_0>(packed_input[i]);
#elif DATAOWNER == 1
        input[i].template prepare_receive_from<P_1>(packed_input[i]);
#elif DATAOWNER == 2
        input[i].template prepare_receive_from<P_2>(packed_input[i]);
#else
#error "BERT FFN DATAOWNER must be 0, 1, or 2"
#endif
    Share::communicate();
    for (auto& value : input)
    {
#if DATAOWNER == 0
        value.template complete_receive_from<P_0>();
#elif DATAOWNER == 1
        value.template complete_receive_from<P_1>();
#elif DATAOWNER == 2
        value.template complete_receive_from<P_2>();
#endif
    }
    stop_layer_stats(0);

    start_layer_stats("ModelShare", 1);
    auto model = bert_ffn::load_and_share_model<Share>(model_path);
    stop_layer_stats(1);
    std::vector<A> output(input.size());
    bert_ffn::BertFFNBlock<Share>().forward(input.data(), packed_rows,
                                            model.w1_high.data(), model.w1_low.data(), model.b1.data(),
                                            model.w2_high.data(), model.w2_low.data(), model.b2.data(),
                                            model.gamma.data(), model.beta.data(), output.data(),
                                            include_residual_layernorm);

    start_layer_stats("OutputReveal", 8);
    for (auto& value : output)
        value.prepare_reveal_to_all();
    Share::communicate();
    std::vector<DATATYPE> revealed(output.size());
    for (std::size_t i = 0; i < output.size(); ++i)
        revealed[i] = output[i].complete_reveal_to_all();
    stop_layer_stats(8);
    *result = revealed[0];

    if (current_phase == PHASE_LIVE)
    {
        std::vector<std::int32_t> raw_output(static_cast<std::size_t>(worker_rows) * bert_ffn::kHidden);
        for (int packed_row = 0; packed_row < packed_rows; ++packed_row)
            for (int col = 0; col < bert_ffn::kHidden; ++col)
            {
                alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
                unorthogonalize_arithmetic(
                    &revealed[static_cast<std::size_t>(packed_row) * bert_ffn::kHidden + col], lane_values.data(), 1);
                for (int lane = 0; lane < lanes; ++lane)
                {
                    const int row = worker_row_begin + packed_row * lanes + lane;
                    if (row < worker_row_end)
                        raw_output[static_cast<std::size_t>(row - worker_row_begin) * bert_ffn::kHidden + col] =
                            static_cast<std::int32_t>(lane_values[lane]);
                }
            }
        if (total_workers == 1)
            bert_ffn::write_counted_q20(output_path, raw_output);
        else
            bert_ffn::write_counted_q20(output_path + ".part-" + std::to_string(worker) + "-of-" +
                                            std::to_string(total_workers) + ".bin",
                                        raw_output);
    }
}
