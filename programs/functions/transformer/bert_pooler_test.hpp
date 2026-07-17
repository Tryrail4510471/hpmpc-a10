#pragma once

// Test-only FUNCTION_IDENTIFIER=208 harness for BertPooler and the MRPC
// two-class head. The production full-BERT graph calls BertPooler directly.
#define RESULTTYPE DATATYPE
#ifndef FUNCTION
#define FUNCTION bert_pooler_test_program
#endif

#include "../../../nn/PIGEON/architectures/bert_pooler.hpp"

// Reuse the audited counted-Q20 I/O/environment helpers from FID=202. The
// FUNCTION macro above prevents bert_ffn.hpp from selecting its own entry.
#include "../../../nn/PIGEON/architectures/bert_ffn.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bert_pooler_test
{
template <typename Share>
using A = bert_pooler::AShare<Share>;

inline std::string join_path(const std::string& directory, const char* filename)
{
    if (directory.empty())
        throw std::invalid_argument("HPMPC_BERT_POOLER_WEIGHTS_DIR must not be empty");
    return directory.back() == '/' ? directory + filename : directory + "/" + filename;
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

template <typename Share>
bert_pooler::PrivateLinear<Share> load_private_linear(const std::string& directory,
                                                      const char* weight_filename,
                                                      const char* bias_filename,
                                                      int output_features,
                                                      int input_features)
{
    const std::size_t weight_count =
        static_cast<std::size_t>(output_features) * input_features;
    if (weight_count > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("Pooler test weight tensor is too large for counted-Q20");
    const std::vector<UINT_TYPE> weight = read_owner_tensor(
        join_path(directory, weight_filename), static_cast<std::uint32_t>(weight_count));
    const std::vector<UINT_TYPE> bias = read_owner_tensor(
        join_path(directory, bias_filename), static_cast<std::uint32_t>(output_features));
    return bert_pooler::share_private_linear<Share>(
        weight, bias, output_features, input_features);
}
}  // namespace bert_pooler_test

template <typename Share>
void bert_pooler_test_program(DATATYPE* result)
{
    using A = bert_pooler_test::A<Share>;
    constexpr int lanes = DATTYPE / BITLENGTH;
    if (process_offset != 0)
        throw std::runtime_error(
            "FID=208 requires PROCESS_NUM=1 (CLS lane gathering cannot be token-sharded)");

    const int batch = bert_ffn::positive_env_int("HPMPC_BERT_POOLER_BATCH", 2);
    const int sequence_length =
        bert_ffn::positive_env_int("HPMPC_BERT_POOLER_SEQUENCE", 128);
    const std::string stage =
        bert_ffn::env_or_default("HPMPC_BERT_POOLER_STAGE", "pooler");
    if (stage != "pooler" && stage != "classifier")
        throw std::invalid_argument(
            "HPMPC_BERT_POOLER_STAGE must be pooler or classifier");

    const int logical_rows = batch * sequence_length;
    const int packed_rows = (logical_rows + lanes - 1) / lanes;
    const int packed_batch = (batch + lanes - 1) / lanes;
    const std::size_t logical_elements =
        static_cast<std::size_t>(logical_rows) * bert_pooler::kHidden;
    const std::size_t packed_elements =
        static_cast<std::size_t>(packed_rows) * bert_pooler::kHidden;
    if (logical_elements > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("Pooler test input is too large for counted-Q20");

    const std::string weights_directory = bert_ffn::env_or_default(
        "HPMPC_BERT_POOLER_WEIGHTS_DIR", "nn/Pygeon/weights/mrpc");
    const std::string input_path = bert_ffn::env_or_default(
        "HPMPC_BERT_POOLER_INPUT", "/tmp/hpmpc_bert_pooler_input_q20.bin");
    const std::string output_path = bert_ffn::env_or_default(
        "HPMPC_BERT_POOLER_OUTPUT", "/tmp/hpmpc_bert_pooler_output_q20.bin");

    // Logical input is row-major [B,L,768]. FlexNN packs successive logical
    // tokens into SIMD lanes and keeps features as matrix columns.
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
            for (int feature = 0; feature < bert_pooler::kHidden; ++feature)
            {
                alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
                for (int lane = 0; lane < lanes; ++lane)
                {
                    const int logical_row = packed_row * lanes + lane;
                    if (logical_row < logical_rows)
                        lane_values[lane] = raw_input[
                            static_cast<std::size_t>(logical_row) * bert_pooler::kHidden + feature];
                }
                orthogonalize_arithmetic(
                    lane_values.data(),
                    &packed_input[static_cast<std::size_t>(packed_row) *
                                      bert_pooler::kHidden +
                                  feature],
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
#error "FID=208 DATAOWNER must be 0, 1, or 2"
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

    // Full-export matrices are model-owner row-major [out,in]. The loader
    // performs the CPU/CUTLASS-specific transpose while splitting Q20 limbs.
    auto pooler = bert_pooler_test::load_private_linear<Share>(
        weights_directory, "pooler_weight.bin", "pooler_bias.bin",
        bert_pooler::kHidden, bert_pooler::kHidden);
    std::vector<A> pooled(static_cast<std::size_t>(packed_batch) * bert_pooler::kHidden);
    bert_pooler::BertPooler<Share> block;
    block.forward(input.data(), batch, sequence_length, pooler, pooled.data());

    int output_features = bert_pooler::kHidden;
    std::vector<A> output;
    if (stage == "pooler")
    {
        output = std::move(pooled);
    }
    else
    {
        auto classifier = bert_pooler_test::load_private_linear<Share>(
            weights_directory, "classifier_mrpc_weight.bin", "classifier_mrpc_bias.bin",
            bert_pooler::kClasses, bert_pooler::kHidden);
        output_features = bert_pooler::kClasses;
        output.resize(static_cast<std::size_t>(packed_batch) * output_features);
        block.classify(pooled.data(), packed_batch, classifier, output.data());
    }

    for (auto& value : output)
        value.prepare_reveal_to_all();
    Share::communicate();
    std::vector<DATATYPE> revealed(output.size());
    for (std::size_t index = 0; index < output.size(); ++index)
        revealed[index] = output[index].complete_reveal_to_all();
    *result = revealed.front();

    if (current_phase == PHASE_LIVE)
    {
        // Unpack batch lanes into row-major [B,768] or [B,2].
        std::vector<std::int32_t> raw_output(
            static_cast<std::size_t>(batch) * output_features);
        for (int pack = 0; pack < packed_batch; ++pack)
            for (int feature = 0; feature < output_features; ++feature)
            {
                alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
                unorthogonalize_arithmetic(
                    &revealed[static_cast<std::size_t>(pack) * output_features + feature],
                    lane_values.data(), 1);
                for (int lane = 0; lane < lanes; ++lane)
                {
                    const int example = pack * lanes + lane;
                    if (example < batch)
                        raw_output[static_cast<std::size_t>(example) * output_features + feature] =
                            static_cast<std::int32_t>(lane_values[lane]);
                }
            }
        bert_ffn::write_counted_q20(output_path, raw_output);
    }
}
