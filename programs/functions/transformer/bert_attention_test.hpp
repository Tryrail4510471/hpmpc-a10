#pragma once

// Test-only FUNCTION_IDENTIFIER=205 harness for the BertAttention submodule.
// The production module intentionally remains unregistered; BertLayer and the
// full BERT entry point call it directly.
#define RESULTTYPE DATATYPE
#ifndef FUNCTION
#define FUNCTION bert_attention_test_program
#endif

#include "../../../nn/PIGEON/architectures/bert_attention.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace bert_attention_test
{
constexpr std::uint32_t kQkvWeightCount =
    bert_attention::kQkv * bert_attention::kHidden;
constexpr std::uint32_t kQkvBiasCount = bert_attention::kQkv;
constexpr std::uint32_t kOutWeightCount =
    bert_attention::kHidden * bert_attention::kHidden;
constexpr std::uint32_t kOutBiasCount = bert_attention::kHidden;
constexpr std::uint32_t kLayerNormCount = 2 * bert_attention::kHidden;
constexpr std::uint32_t kParameterCount =
    kQkvWeightCount + kQkvBiasCount + kOutWeightCount + kOutBiasCount + kLayerNormCount;

template <typename Share>
using A = bert_attention::AShare<Share>;

template <typename Share>
struct ModelShares
{
    std::vector<A<Share>> qkv_high;
    std::vector<A<Share>> qkv_low;
    std::vector<A<Share>> qkv_bias;
    std::vector<A<Share>> out_high;
    std::vector<A<Share>> out_low;
    std::vector<A<Share>> out_bias;

    ModelShares()
        : qkv_high(kQkvWeightCount), qkv_low(kQkvWeightCount), qkv_bias(kQkvBiasCount),
          out_high(kOutWeightCount), out_low(kOutWeightCount), out_bias(kOutBiasCount)
    {
    }

    bert_attention::WeightsView<Share> view() const
    {
        return {qkv_high.data(), qkv_low.data(), qkv_bias.data(),
                out_high.data(), out_low.data(), out_bias.data()};
    }
};

template <typename Share>
ModelShares<Share> load_and_share_model(const std::string& path)
{
    std::vector<UINT_TYPE> raw(kParameterCount, UINT_TYPE(0));
#if PARTY == MODELOWNER
    if (current_phase == PHASE_LIVE)
        raw = bert_ffn::read_counted_q20(path, kParameterCount);
#endif

    ModelShares<Share> model;
    std::size_t offset = 0;
    auto receive_weight = [&](auto& high_destination,
                              auto& low_destination,
                              int output_features,
                              int input_features) {
        std::vector<UINT_TYPE> high(high_destination.size(), UINT_TYPE(0));
        std::vector<UINT_TYPE> low(low_destination.size(), UINT_TYPE(0));
#if PARTY == MODELOWNER
        if (current_phase == PHASE_LIVE)
        {
            for (int output_feature = 0; output_feature < output_features; ++output_feature)
                for (int input_feature = 0; input_feature < input_features; ++input_feature)
                {
                    const std::size_t source =
                        offset + static_cast<std::size_t>(output_feature) * input_features + input_feature;
                    const std::int32_t value = static_cast<std::int32_t>(raw[source]);
                    const std::int32_t high_signed = value >> bert_attention::kLimbBits;
                    const UINT_TYPE high_raw = static_cast<UINT_TYPE>(high_signed);
                    const UINT_TYPE low_raw =
                        static_cast<UINT_TYPE>(value) - high_raw * bert_attention::kLimbBase;
#if USE_CUDA_GEMM > 0
                    const std::size_t destination =
                        static_cast<std::size_t>(input_feature) * output_features + output_feature;
#else
                    const std::size_t destination =
                        static_cast<std::size_t>(output_feature) * input_features + input_feature;
#endif
                    high[destination] = high_raw;
                    low[destination] = low_raw;
                }
        }
#endif
        bert_ffn::receive_private_vector<Share>(high, high_destination);
        bert_ffn::receive_private_vector<Share>(low, low_destination);
        offset += high_destination.size();
    };
    auto receive_slice = [&](auto& destination) {
        std::vector<UINT_TYPE> slice(destination.size(), UINT_TYPE(0));
#if PARTY == MODELOWNER
        if (current_phase == PHASE_LIVE)
            std::copy(raw.begin() + static_cast<std::ptrdiff_t>(offset),
                      raw.begin() + static_cast<std::ptrdiff_t>(offset + destination.size()),
                      slice.begin());
#endif
        bert_ffn::receive_private_vector<Share>(slice, destination);
        offset += destination.size();
    };

    receive_weight(model.qkv_high, model.qkv_low,
                   bert_attention::kQkv, bert_attention::kHidden);
    receive_slice(model.qkv_bias);
    receive_weight(model.out_high, model.out_low,
                   bert_attention::kHidden, bert_attention::kHidden);
    receive_slice(model.out_bias);
    // The test exporter appends attention-output LayerNorm gamma/beta for the
    // following BertLayer test.  This pre-residual module deliberately skips
    // those two tensors.
    offset += kLayerNormCount;
    if (offset != raw.size())
        throw std::runtime_error("BertAttention model stream layout mismatch");
    return model;
}

inline std::vector<std::uint8_t> read_public_keep_mask(const std::string& path,
                                                       int batch,
                                                       int sequence_length)
{
    const std::uint32_t count = static_cast<std::uint32_t>(batch * sequence_length);
    const std::vector<UINT_TYPE> additive = bert_ffn::read_counted_q20(path, count);
    std::vector<std::uint8_t> keep(count, 1);
    for (std::uint32_t index = 0; index < count; ++index)
        keep[index] = static_cast<std::int32_t>(additive[index]) == 0 ? 1U : 0U;
    return keep;
}
}  // namespace bert_attention_test

template <typename Share>
void bert_attention_test_program(DATATYPE* result)
{
    using A = bert_attention_test::A<Share>;
    constexpr int lanes = DATTYPE / BITLENGTH;
    const int batch = bert_ffn::positive_env_int("HPMPC_BERT_ATTENTION_BATCH", 2);
    const int sequence_length =
        bert_ffn::positive_env_int("HPMPC_BERT_ATTENTION_SEQUENCE", 128);
    if (sequence_length > bert_attention::kMaxSequenceLength)
        throw std::invalid_argument("HPMPC_BERT_ATTENTION_SEQUENCE must be <=128");

    const int logical_rows = batch * sequence_length;
    const int packed_rows = (logical_rows + lanes - 1) / lanes;
    const std::size_t logical_elements =
        static_cast<std::size_t>(logical_rows) * bert_attention::kHidden;
    const std::size_t packed_elements =
        static_cast<std::size_t>(packed_rows) * bert_attention::kHidden;
    const std::string model_path = bert_ffn::env_or_default(
        "HPMPC_BERT_ATTENTION_WEIGHTS",
        "nn/Pygeon/models/pretrained/bert_base_uncased_layer0_attention_q20.bin");
    const std::string input_path = bert_ffn::env_or_default(
        "HPMPC_BERT_ATTENTION_INPUT", "/tmp/hpmpc_bert_attention_input_q20.bin");
    const std::string mask_path = bert_ffn::env_or_default(
        "HPMPC_BERT_ATTENTION_MASK", "/tmp/hpmpc_bert_attention_mask_q20.bin");
    const std::string output_path = bert_ffn::env_or_default(
        "HPMPC_BERT_ATTENTION_OUTPUT", "/tmp/hpmpc_bert_attention_output_q20.bin");

    std::vector<UINT_TYPE> raw_input(logical_elements, UINT_TYPE(0));
#if PARTY == DATAOWNER
    if (current_phase == PHASE_LIVE)
        raw_input = bert_ffn::read_counted_q20(
            input_path, static_cast<std::uint32_t>(logical_elements));
#endif

    std::vector<DATATYPE> packed_input(packed_elements);
#if PARTY == DATAOWNER
    if (current_phase == PHASE_LIVE)
    {
        for (int packed_row = 0; packed_row < packed_rows; ++packed_row)
            for (int feature = 0; feature < bert_attention::kHidden; ++feature)
            {
                alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
                for (int lane = 0; lane < lanes; ++lane)
                {
                    const int logical_row = packed_row * lanes + lane;
                    if (logical_row < logical_rows)
                        lane_values[lane] = raw_input[
                            static_cast<std::size_t>(logical_row) * bert_attention::kHidden + feature];
                }
                orthogonalize_arithmetic(
                    lane_values.data(),
                    &packed_input[static_cast<std::size_t>(packed_row) * bert_attention::kHidden + feature],
                    1);
            }
    }
#endif

    std::vector<A> input(packed_elements);
    for (std::size_t index = 0; index < packed_elements; ++index)
    {
#if DATAOWNER == 0
        input[index].template prepare_receive_from<P_0>(packed_input[index]);
#elif DATAOWNER == 1
        input[index].template prepare_receive_from<P_1>(packed_input[index]);
#elif DATAOWNER == 2
        input[index].template prepare_receive_from<P_2>(packed_input[index]);
#endif
    }
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

    auto model = bert_attention_test::load_and_share_model<Share>(model_path);
    std::vector<std::uint8_t> keep_mask(
        static_cast<std::size_t>(batch) * sequence_length, std::uint8_t(1));
    if (current_phase == PHASE_LIVE)
        keep_mask = bert_attention_test::read_public_keep_mask(mask_path, batch, sequence_length);

    const std::string stage = bert_ffn::env_or_default("HPMPC_BERT_ATTENTION_STAGE", "full");
    int output_features = bert_attention::kHidden;
    bool head_layout = false;
    std::vector<A> output;
    if (stage == "qkv" || stage == "query")
    {
        std::vector<A> qkv(
            static_cast<std::size_t>(packed_rows) * bert_attention::kQkv);
        bert_ffn::gemm_fixed<Share>(
            input.data(), model.qkv_high.data(), model.qkv_low.data(), qkv.data(),
            packed_rows, bert_attention::kQkv, bert_attention::kHidden);
        for (int row = 0; row < packed_rows; ++row)
            for (int feature = 0; feature < bert_attention::kQkv; ++feature)
                qkv[static_cast<std::size_t>(row) * bert_attention::kQkv + feature] =
                    qkv[static_cast<std::size_t>(row) * bert_attention::kQkv + feature] +
                    model.qkv_bias[feature];
        if (stage == "qkv")
        {
            output_features = bert_attention::kQkv;
            output = std::move(qkv);
        }
        else
        {
            std::vector<A> key;
            std::vector<A> value;
            bert_attention::detail::token_qkv_to_head_layout<Share>(
                qkv.data(), batch, sequence_length, output, key, value);
            output_features = bert_attention::kHeadDim;
            head_layout = true;
        }
    }
    else if (stage == "full")
    {
        output.resize(packed_elements);
        bert_attention::BertAttentionBlock<Share>().forward(
            input.data(), output.data(), model.view(), batch, sequence_length, keep_mask.data());
    }
    else
        throw std::invalid_argument("HPMPC_BERT_ATTENTION_STAGE must be full, qkv, or query");

    for (auto& value : output)
        value.prepare_reveal_to_all();
    Share::communicate();
    std::vector<DATATYPE> revealed(output.size());
    for (std::size_t index = 0; index < output.size(); ++index)
        revealed[index] = output[index].complete_reveal_to_all();
    *result = revealed.front();

    if (current_phase == PHASE_LIVE)
    {
        std::vector<std::int32_t> raw_output;
        if (head_layout)
        {
            const int logical_heads = batch * bert_attention::kHeads;
            const int head_packs = (logical_heads + lanes - 1) / lanes;
            raw_output.resize(
                static_cast<std::size_t>(logical_heads) * sequence_length * output_features);
            for (int head_pack = 0; head_pack < head_packs; ++head_pack)
                for (int position = 0; position < sequence_length; ++position)
                    for (int feature = 0; feature < output_features; ++feature)
                    {
                        alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
                        unorthogonalize_arithmetic(
                            &revealed[(static_cast<std::size_t>(head_pack) * sequence_length + position) *
                                          output_features +
                                      feature],
                            lane_values.data(), 1);
                        for (int lane = 0; lane < lanes; ++lane)
                        {
                            const int logical_head = head_pack * lanes + lane;
                            if (logical_head < logical_heads)
                                raw_output[(static_cast<std::size_t>(logical_head) * sequence_length + position) *
                                               output_features +
                                           feature] = static_cast<std::int32_t>(lane_values[lane]);
                        }
                    }
        }
        else
        {
            raw_output.resize(static_cast<std::size_t>(logical_rows) * output_features);
            for (int packed_row = 0; packed_row < packed_rows; ++packed_row)
                for (int feature = 0; feature < output_features; ++feature)
                {
                    alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
                    unorthogonalize_arithmetic(
                        &revealed[static_cast<std::size_t>(packed_row) * output_features + feature],
                        lane_values.data(), 1);
                    for (int lane = 0; lane < lanes; ++lane)
                    {
                        const int logical_row = packed_row * lanes + lane;
                        if (logical_row < logical_rows)
                            raw_output[static_cast<std::size_t>(logical_row) * output_features + feature] =
                                static_cast<std::int32_t>(lane_values[lane]);
                    }
                }
        }
        bert_ffn::write_counted_q20(output_path, raw_output);
    }
}
