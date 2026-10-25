#pragma once

// Test-only FUNCTION_IDENTIFIER=206 harness for one complete post-LN
// BertLayer.  Production encoder code calls BertLayerBlock directly; this
// entry point exists only to validate layer 0 against a plaintext reference.
#define RESULTTYPE DATATYPE
#ifndef FUNCTION
#define FUNCTION bert_layer_test_program
#endif

#include "../../../nn/PIGEON/architectures/bert_layer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bert_layer_test
{
constexpr std::size_t kQkvWeightCount =
    static_cast<std::size_t>(bert_attention::kQkv) * bert_attention::kHidden;
constexpr std::size_t kQkvBiasCount = bert_attention::kQkv;
constexpr std::size_t kAttentionWeightCount =
    static_cast<std::size_t>(bert_attention::kHidden) * bert_attention::kHidden;
constexpr std::size_t kHiddenCount = bert_layer::kHidden;
constexpr std::size_t kIntermediateWeightCount =
    static_cast<std::size_t>(bert_ffn::kIntermediate) * bert_ffn::kHidden;
constexpr std::size_t kOutputWeightCount =
    static_cast<std::size_t>(bert_ffn::kHidden) * bert_ffn::kIntermediate;

template <typename Share>
using A = bert_layer::AShare<Share>;

inline std::string join_path(const std::string& directory, const char* filename)
{
    if (directory.empty())
        throw std::invalid_argument("HPMPC_BERT_LAYER_WEIGHTS_DIR must not be empty");
    return directory.back() == '/' ? directory + filename : directory + "/" + filename;
}

inline std::string join_path(const std::string& directory, const std::string& filename)
{
    return join_path(directory, filename.c_str());
}

inline std::string layer_filename(int layer_index, const char* suffix)
{
    if (layer_index < 0 || layer_index >= 12)
        throw std::out_of_range("BERT layer index must be in [0,11]");
    return "layer" + std::to_string(layer_index) + "_" + suffix + ".bin";
}

// The full exporter stores Q itself in ordinary Q12.20.  Attention's secure
// graph instead consumes Q/8, so apply the public scaling to the model owner's
// raw integers before secret sharing.  Ties round away from zero; the result
// differs from direct floating-point quantization by at most one Q20 unit.
inline UINT_TYPE rounded_divide_q20_by_8(UINT_TYPE raw)
{
    const std::int64_t signed_value = static_cast<std::int32_t>(raw);
    const std::int64_t magnitude = signed_value < 0 ? -signed_value : signed_value;
    const std::int64_t rounded = (magnitude + 4) / 8;
    const std::int64_t result = signed_value < 0 ? -rounded : rounded;
    return static_cast<UINT_TYPE>(static_cast<std::int32_t>(result));
}

inline std::vector<UINT_TYPE> read_owner_tensor(const std::string& path,
                                                std::uint32_t expected)
{
#if PARTY == MODELOWNER
    if (current_phase == PHASE_LIVE)
        return bert_ffn::read_counted_q20(path, expected);
#else
    (void)path;
    (void)expected;
#endif
    return {};
}

inline std::vector<std::uint8_t> read_public_counted_u8(const std::string& path,
                                                        std::uint32_t expected)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open counted-u8 tensor file: " + path);
    std::uint32_t count = 0;
    stream.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!stream || count != expected)
        throw std::runtime_error("unexpected element count in " + path);
    std::vector<std::uint8_t> values(count);
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size()));
    if (!stream)
        throw std::runtime_error("truncated counted-u8 tensor file: " + path);
    char trailing = 0;
    if (stream.read(&trailing, 1))
        throw std::runtime_error("trailing payload in counted-u8 tensor file: " + path);
    for (const std::uint8_t value : values)
        if (value > 1)
            throw std::runtime_error("BERT key-padding mask must contain only 0 or 1");
    return values;
}

template <typename Share>
void share_private_slice(const std::vector<UINT_TYPE>& owner_values,
                         std::vector<A<Share>>& destination)
{
#if PARTY == MODELOWNER
    if (current_phase == PHASE_LIVE && owner_values.size() != destination.size())
        throw std::runtime_error("private BERT parameter size mismatch");
#endif
    std::vector<UINT_TYPE> values(destination.size(), UINT_TYPE(0));
#if PARTY == MODELOWNER
    if (current_phase == PHASE_LIVE)
        values = owner_values;
#endif
    bert_ffn::receive_private_vector<Share>(values, destination);
}

// 把 HuggingFace 的 [out,in] Q20 权重拆成 10-bit high/low。
// USE_CUDA_GEMM>0 时目的下标改成 [in,out]，与 CUTLASS 的物理布局一致；
// CPU GEMM 保持 [out,in]。Q 的 /8 不在这里做，见 rounded_divide_q20_by_8。
template <typename Share>
void share_private_matrix(const std::vector<UINT_TYPE>& owner_values,
                          int output_features,
                          int input_features,
                          std::vector<A<Share>>& high_destination,
                          std::vector<A<Share>>& low_destination)
{
    const std::size_t expected =
        static_cast<std::size_t>(output_features) * input_features;
    if (high_destination.size() != expected || low_destination.size() != expected)
        throw std::invalid_argument("private BERT matrix destination has the wrong size");
#if PARTY == MODELOWNER
    if (current_phase == PHASE_LIVE && owner_values.size() != expected)
        throw std::runtime_error("private BERT matrix source has the wrong size");
#endif

    std::vector<UINT_TYPE> high(expected, UINT_TYPE(0));
    std::vector<UINT_TYPE> low(expected, UINT_TYPE(0));
#if PARTY == MODELOWNER
    if (current_phase == PHASE_LIVE)
        for (int output_feature = 0; output_feature < output_features; ++output_feature)
            for (int input_feature = 0; input_feature < input_features; ++input_feature)
            {
                const std::size_t source =
                    static_cast<std::size_t>(output_feature) * input_features + input_feature;
                const std::int32_t value = static_cast<std::int32_t>(owner_values[source]);
                const std::int32_t high_signed = value >> bert_ffn::kLimbBits;
                const UINT_TYPE high_raw = static_cast<UINT_TYPE>(high_signed);
                const UINT_TYPE low_raw =
                    static_cast<UINT_TYPE>(value) - high_raw * bert_ffn::kLimbBase;
#if USE_CUDA_GEMM > 0
                const std::size_t destination =
                    static_cast<std::size_t>(input_feature) * output_features + output_feature;
#else
                const std::size_t destination = source;
#endif
                high[destination] = high_raw;
                low[destination] = low_raw;
            }
#endif
    bert_ffn::receive_private_vector<Share>(high, high_destination);
    bert_ffn::receive_private_vector<Share>(low, low_destination);
}

template <typename Share>
struct ModelShares
{
    std::vector<A<Share>> qkv_high;
    std::vector<A<Share>> qkv_low;
    std::vector<A<Share>> qkv_bias;
    std::vector<A<Share>> attention_output_high;
    std::vector<A<Share>> attention_output_low;
    std::vector<A<Share>> attention_output_bias;
    std::vector<A<Share>> attention_ln_gamma;
    std::vector<A<Share>> attention_ln_beta;
    std::vector<A<Share>> intermediate_high;
    std::vector<A<Share>> intermediate_low;
    std::vector<A<Share>> intermediate_bias;
    std::vector<A<Share>> output_high;
    std::vector<A<Share>> output_low;
    std::vector<A<Share>> output_bias;
    std::vector<A<Share>> output_ln_gamma;
    std::vector<A<Share>> output_ln_beta;

    ModelShares()
        : qkv_high(kQkvWeightCount), qkv_low(kQkvWeightCount),
          qkv_bias(kQkvBiasCount), attention_output_high(kAttentionWeightCount),
          attention_output_low(kAttentionWeightCount), attention_output_bias(kHiddenCount),
          attention_ln_gamma(kHiddenCount), attention_ln_beta(kHiddenCount),
          intermediate_high(kIntermediateWeightCount), intermediate_low(kIntermediateWeightCount),
          intermediate_bias(bert_ffn::kIntermediate), output_high(kOutputWeightCount),
          output_low(kOutputWeightCount), output_bias(kHiddenCount),
          output_ln_gamma(kHiddenCount), output_ln_beta(kHiddenCount)
    {
    }

    bert_layer::BertLayerWeights<Share> view() const
    {
        bert_layer::BertLayerWeights<Share> weights;
        weights.attention = {
            qkv_high.data(), qkv_low.data(), qkv_bias.data(),
            attention_output_high.data(), attention_output_low.data(),
            attention_output_bias.data()};
        weights.attention_layernorm_gamma = attention_ln_gamma.data();
        weights.attention_layernorm_beta = attention_ln_beta.data();
        weights.intermediate_weight_high = intermediate_high.data();
        weights.intermediate_weight_low = intermediate_low.data();
        weights.intermediate_bias = intermediate_bias.data();
        weights.output_weight_high = output_high.data();
        weights.output_weight_low = output_low.data();
        weights.output_bias = output_bias.data();
        weights.output_layernorm_gamma = output_ln_gamma.data();
        weights.output_layernorm_beta = output_ln_beta.data();
        return weights;
    }
};

template <typename Share>
ModelShares<Share> load_and_share_layer(const std::string& directory, int layer_index)
{
    if (layer_index < 0 || layer_index >= 12)
        throw std::out_of_range("BERT layer index must be in [0,11]");
    ModelShares<Share> model;

    // Concatenate [Q;K;V] in output-feature order.  Only Q is pre-scaled.
    std::vector<UINT_TYPE> qkv_owner;
#if PARTY == MODELOWNER
    if (current_phase == PHASE_LIVE)
    {
        std::vector<UINT_TYPE> query = read_owner_tensor(
            join_path(directory, layer_filename(layer_index, "q_weight")),
            static_cast<std::uint32_t>(kAttentionWeightCount));
        const std::vector<UINT_TYPE> key = read_owner_tensor(
            join_path(directory, layer_filename(layer_index, "k_weight")),
            static_cast<std::uint32_t>(kAttentionWeightCount));
        const std::vector<UINT_TYPE> value = read_owner_tensor(
            join_path(directory, layer_filename(layer_index, "v_weight")),
            static_cast<std::uint32_t>(kAttentionWeightCount));
        for (UINT_TYPE& element : query)
            element = rounded_divide_q20_by_8(element);
        qkv_owner.reserve(kQkvWeightCount);
        qkv_owner.insert(qkv_owner.end(), query.begin(), query.end());
        qkv_owner.insert(qkv_owner.end(), key.begin(), key.end());
        qkv_owner.insert(qkv_owner.end(), value.begin(), value.end());
    }
#endif
    share_private_matrix<Share>(qkv_owner, bert_attention::kQkv,
                                bert_attention::kHidden,
                                model.qkv_high, model.qkv_low);

    std::vector<UINT_TYPE> qkv_bias_owner;
#if PARTY == MODELOWNER
    if (current_phase == PHASE_LIVE)
    {
        std::vector<UINT_TYPE> query = read_owner_tensor(
            join_path(directory, layer_filename(layer_index, "q_bias")),
            bert_attention::kHidden);
        const std::vector<UINT_TYPE> key = read_owner_tensor(
            join_path(directory, layer_filename(layer_index, "k_bias")),
            bert_attention::kHidden);
        const std::vector<UINT_TYPE> value = read_owner_tensor(
            join_path(directory, layer_filename(layer_index, "v_bias")),
            bert_attention::kHidden);
        for (UINT_TYPE& element : query)
            element = rounded_divide_q20_by_8(element);
        qkv_bias_owner.reserve(kQkvBiasCount);
        qkv_bias_owner.insert(qkv_bias_owner.end(), query.begin(), query.end());
        qkv_bias_owner.insert(qkv_bias_owner.end(), key.begin(), key.end());
        qkv_bias_owner.insert(qkv_bias_owner.end(), value.begin(), value.end());
    }
#endif
    share_private_slice<Share>(qkv_bias_owner, model.qkv_bias);

    auto share_matrix_file = [&](const char* filename,
                                 int output_features,
                                 int input_features,
                                 auto& high,
                                 auto& low) {
        const std::uint32_t count = static_cast<std::uint32_t>(
            static_cast<std::size_t>(output_features) * input_features);
        const std::vector<UINT_TYPE> raw =
            read_owner_tensor(join_path(directory, filename), count);
        share_private_matrix<Share>(raw, output_features, input_features, high, low);
    };
    auto share_slice_file = [&](const char* filename, auto& destination) {
        const std::vector<UINT_TYPE> raw = read_owner_tensor(
            join_path(directory, filename),
            static_cast<std::uint32_t>(destination.size()));
        share_private_slice<Share>(raw, destination);
    };

    const std::string attention_output_weight =
        layer_filename(layer_index, "attention_output_weight");
    const std::string attention_output_bias =
        layer_filename(layer_index, "attention_output_bias");
    const std::string attention_ln_weight =
        layer_filename(layer_index, "attention_ln_weight");
    const std::string attention_ln_bias =
        layer_filename(layer_index, "attention_ln_bias");
    const std::string ffn_intermediate_weight =
        layer_filename(layer_index, "ffn_intermediate_weight");
    const std::string ffn_intermediate_bias =
        layer_filename(layer_index, "ffn_intermediate_bias");
    const std::string ffn_output_weight =
        layer_filename(layer_index, "ffn_output_weight");
    const std::string ffn_output_bias =
        layer_filename(layer_index, "ffn_output_bias");
    const std::string ffn_ln_weight = layer_filename(layer_index, "ffn_ln_weight");
    const std::string ffn_ln_bias = layer_filename(layer_index, "ffn_ln_bias");

    share_matrix_file(attention_output_weight.c_str(), bert_attention::kHidden,
                      bert_attention::kHidden, model.attention_output_high,
                      model.attention_output_low);
    share_slice_file(attention_output_bias.c_str(), model.attention_output_bias);
    share_slice_file(attention_ln_weight.c_str(), model.attention_ln_gamma);
    share_slice_file(attention_ln_bias.c_str(), model.attention_ln_beta);
    share_matrix_file(ffn_intermediate_weight.c_str(), bert_ffn::kIntermediate,
                      bert_ffn::kHidden, model.intermediate_high, model.intermediate_low);
    share_slice_file(ffn_intermediate_bias.c_str(), model.intermediate_bias);
    share_matrix_file(ffn_output_weight.c_str(), bert_ffn::kHidden,
                      bert_ffn::kIntermediate, model.output_high, model.output_low);
    share_slice_file(ffn_output_bias.c_str(), model.output_bias);
    share_slice_file(ffn_ln_weight.c_str(), model.output_ln_gamma);
    share_slice_file(ffn_ln_bias.c_str(), model.output_ln_beta);
    return model;
}

template <typename Share>
ModelShares<Share> load_and_share_layer0(const std::string& directory)
{
    return load_and_share_layer<Share>(directory, 0);
}
}  // namespace bert_layer_test

template <typename Share>
void bert_layer_test_program(DATATYPE* result)
{
    using A = bert_layer_test::A<Share>;
    constexpr int lanes = DATTYPE / BITLENGTH;
    if (process_offset != 0)
        throw std::runtime_error("FID=206 requires PROCESS_NUM=1 (a layer cannot be token-sharded)");

    const int batch = bert_ffn::positive_env_int("HPMPC_BERT_LAYER_BATCH", 2);
    const int sequence_length =
        bert_ffn::positive_env_int("HPMPC_BERT_LAYER_SEQUENCE", 128);
    if (sequence_length > bert_attention::kMaxSequenceLength)
        throw std::invalid_argument("HPMPC_BERT_LAYER_SEQUENCE must be <=128");
    const int logical_rows = batch * sequence_length;
    const int packed_rows = (logical_rows + lanes - 1) / lanes;
    const std::size_t logical_elements =
        static_cast<std::size_t>(logical_rows) * bert_layer::kHidden;
    const std::size_t packed_elements =
        static_cast<std::size_t>(packed_rows) * bert_layer::kHidden;

    const std::string weights_directory = bert_ffn::env_or_default(
        "HPMPC_BERT_LAYER_WEIGHTS_DIR", "nn/Pygeon/weights/mrpc");
    const std::string input_path = bert_ffn::env_or_default(
        "HPMPC_BERT_LAYER_INPUT", "/tmp/hpmpc_bert_layer_input_q20.bin");
    const std::string mask_path = bert_ffn::env_or_default(
        "HPMPC_BERT_LAYER_MASK", "/tmp/hpmpc_bert_layer_key_padding_mask_u8.bin");
    const std::string output_path = bert_ffn::env_or_default(
        "HPMPC_BERT_LAYER_OUTPUT", "/tmp/hpmpc_bert_layer_output_q20.bin");

    std::vector<UINT_TYPE> raw_input(logical_elements, UINT_TYPE(0));
#if PARTY == DATAOWNER
    if (current_phase == PHASE_LIVE)
        raw_input = bert_ffn::read_counted_q20(
            input_path, static_cast<std::uint32_t>(logical_elements));
#endif
    std::vector<DATATYPE> packed_input(packed_elements);
#if PARTY == DATAOWNER
    if (current_phase == PHASE_LIVE)
        for (int packed_row = 0; packed_row < packed_rows; ++packed_row)
            for (int feature = 0; feature < bert_layer::kHidden; ++feature)
            {
                alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
                for (int lane = 0; lane < lanes; ++lane)
                {
                    const int logical_row = packed_row * lanes + lane;
                    if (logical_row < logical_rows)
                        lane_values[lane] = raw_input[
                            static_cast<std::size_t>(logical_row) * bert_layer::kHidden + feature];
                }
                orthogonalize_arithmetic(
                    lane_values.data(),
                    &packed_input[static_cast<std::size_t>(packed_row) * bert_layer::kHidden + feature],
                    1);
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
#else
#error "FID=206 DATAOWNER must be 0, 1, or 2"
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

    bert_layer_test::ModelShares<Share> model =
        bert_layer_test::load_and_share_layer0<Share>(weights_directory);
    std::vector<std::uint8_t> keep_mask(
        static_cast<std::size_t>(batch) * sequence_length, std::uint8_t(1));
    if (current_phase == PHASE_LIVE)
        keep_mask = bert_layer_test::read_public_counted_u8(
            mask_path, static_cast<std::uint32_t>(keep_mask.size()));

    std::vector<A> output(packed_elements);
    bert_layer::BertLayerBlock<Share>().forward(
        input.data(), output.data(), model.view(), batch, sequence_length, keep_mask.data());

    for (auto& value : output)
        value.prepare_reveal_to_all();
    Share::communicate();
    std::vector<DATATYPE> revealed(output.size());
    for (std::size_t index = 0; index < output.size(); ++index)
        revealed[index] = output[index].complete_reveal_to_all();
    *result = revealed.front();

    if (current_phase == PHASE_LIVE)
    {
        std::vector<std::int32_t> raw_output(logical_elements);
        for (int packed_row = 0; packed_row < packed_rows; ++packed_row)
            for (int feature = 0; feature < bert_layer::kHidden; ++feature)
            {
                alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
                unorthogonalize_arithmetic(
                    &revealed[static_cast<std::size_t>(packed_row) * bert_layer::kHidden + feature],
                    lane_values.data(), 1);
                for (int lane = 0; lane < lanes; ++lane)
                {
                    const int logical_row = packed_row * lanes + lane;
                    if (logical_row < logical_rows)
                        raw_output[static_cast<std::size_t>(logical_row) * bert_layer::kHidden + feature] =
                            static_cast<std::int32_t>(lane_values[lane]);
                }
            }
        bert_ffn::write_counted_q20(output_path, raw_output);
    }
}
