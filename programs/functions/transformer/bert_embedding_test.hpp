#pragma once

// Test-only FUNCTION_IDENTIFIER=207 harness for BertEmbedding.
//
// Security boundary:
//   * token_ids and token_type_ids are public and must be identical on all
//     three hosts.  They use counted-u32 files (uint32 count + uint32 values).
//   * the full embedding tables and private LayerNorm affine parameters are
//     read only by MODELOWNER during PHASE_LIVE.
//   * BertEmbedding selects the requested rows locally at MODELOWNER and
//     secret-shares only those selected Q12.20 rows; it never shares a full
//     vocabulary table.
#define RESULTTYPE DATATYPE
#ifndef FUNCTION
#define FUNCTION bert_embedding_test_program
#endif

#include "../../../nn/PIGEON/architectures/bert_embedding.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bert_embedding_test
{
template <typename Share>
using A = bert_embedding::AShare<Share>;

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

inline std::string join_path(const std::string& directory, const char* filename)
{
    if (directory.empty())
        throw std::invalid_argument("HPMPC_BERT_EMBEDDING_WEIGHTS_DIR must not be empty");
    return directory.back() == '/' ? directory + filename : directory + "/" + filename;
}

inline void require_no_trailing_bytes(std::ifstream& stream, const std::string& path)
{
    char extra = 0;
    stream.read(&extra, 1);
    if (stream.gcount() != 0)
        throw std::runtime_error("unexpected trailing bytes in " + path);
}

inline std::vector<std::uint32_t> read_counted_u32(const std::string& path,
                                                   std::uint32_t expected)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open public counted-u32 file: " + path);
    std::uint32_t count = 0;
    stream.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!stream || count != expected)
        throw std::runtime_error("unexpected public u32 element count in " + path);
    std::vector<std::uint32_t> values(count);
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(count) * sizeof(std::uint32_t));
    if (!stream)
        throw std::runtime_error("truncated public counted-u32 file: " + path);
    require_no_trailing_bytes(stream, path);
    return values;
}

inline std::vector<UINT_TYPE> read_counted_q20(const std::string& path,
                                               std::uint32_t expected)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open private Q20 tensor file: " + path);
    std::uint32_t count = 0;
    stream.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!stream || count != expected)
        throw std::runtime_error("unexpected Q20 element count in " + path);
    std::vector<std::int32_t> signed_values(count);
    stream.read(reinterpret_cast<char*>(signed_values.data()),
                static_cast<std::streamsize>(count) * sizeof(std::int32_t));
    if (!stream)
        throw std::runtime_error("truncated private Q20 tensor file: " + path);
    require_no_trailing_bytes(stream, path);
    std::vector<UINT_TYPE> values(count);
    for (std::uint32_t index = 0; index < count; ++index)
        values[index] = static_cast<UINT_TYPE>(signed_values[index]);
    return values;
}

inline void write_counted_q20(const std::string& path,
                              const std::vector<std::int32_t>& values)
{
    if (values.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("embedding output is too large for counted-Q20");
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
        throw std::runtime_error("cannot create embedding output file: " + path);
    const std::uint32_t count = static_cast<std::uint32_t>(values.size());
    stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size()) * sizeof(std::int32_t));
    if (!stream)
        throw std::runtime_error("failed to write embedding output file: " + path);
}

inline bert_embedding::OwnerWeights load_owner_weights(const std::string& directory)
{
    bert_embedding::OwnerWeights weights;
#if PARTY == MODELOWNER
    if (current_phase == PHASE_LIVE)
    {
        constexpr std::uint32_t word_count =
            bert_embedding::kDefaultVocabulary * bert_embedding::kHidden;
        constexpr std::uint32_t position_count =
            bert_embedding::kDefaultPositions * bert_embedding::kHidden;
        constexpr std::uint32_t token_type_count =
            bert_embedding::kDefaultTokenTypes * bert_embedding::kHidden;
        weights.word = read_counted_q20(
            join_path(directory, "embeddings_word.bin"), word_count);
        weights.position = read_counted_q20(
            join_path(directory, "embeddings_position.bin"), position_count);
        weights.token_type = read_counted_q20(
            join_path(directory, "embeddings_token_type.bin"), token_type_count);
        weights.gamma = read_counted_q20(
            join_path(directory, "embeddings_ln_weight.bin"), bert_embedding::kHidden);
        weights.beta = read_counted_q20(
            join_path(directory, "embeddings_ln_bias.bin"), bert_embedding::kHidden);
    }
#endif
    return weights;
}
}  // namespace bert_embedding_test

// Environment contract:
//   HPMPC_BERT_EMBEDDING_BATCH
//   HPMPC_BERT_EMBEDDING_SEQUENCE
//   HPMPC_BERT_EMBEDDING_WEIGHTS_DIR
//   HPMPC_BERT_EMBEDDING_TOKEN_IDS
//   HPMPC_BERT_EMBEDDING_TOKEN_TYPE_IDS
//   HPMPC_BERT_EMBEDDING_OUTPUT
template <typename Share>
void bert_embedding_test_program(DATATYPE* result)
{
    using A = bert_embedding_test::A<Share>;
    constexpr int lanes = DATTYPE / BITLENGTH;
    if (process_offset != 0)
        throw std::runtime_error("FID=207 requires PROCESS_NUM=1");

    const int batch = bert_embedding_test::positive_env_int(
        "HPMPC_BERT_EMBEDDING_BATCH", 2);
    const int sequence_length = bert_embedding_test::positive_env_int(
        "HPMPC_BERT_EMBEDDING_SEQUENCE", 128);
    const std::size_t logical_tokens = static_cast<std::size_t>(batch) * sequence_length;
    if (logical_tokens > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("embedding token count does not fit counted-u32");
    const int packed_rows = bert_embedding::packed_token_rows(batch, sequence_length);
    const std::size_t packed_elements =
        static_cast<std::size_t>(packed_rows) * bert_embedding::kHidden;

    const std::string weights_directory = bert_embedding_test::env_or_default(
        "HPMPC_BERT_EMBEDDING_WEIGHTS_DIR", "nn/Pygeon/weights/mrpc");
    const std::string token_ids_path = bert_embedding_test::env_or_default(
        "HPMPC_BERT_EMBEDDING_TOKEN_IDS", "/tmp/hpmpc_bert_embedding_token_ids_u32.bin");
    const std::string token_type_ids_path = bert_embedding_test::env_or_default(
        "HPMPC_BERT_EMBEDDING_TOKEN_TYPE_IDS",
        "/tmp/hpmpc_bert_embedding_token_type_ids_u32.bin");
    const std::string output_path = bert_embedding_test::env_or_default(
        "HPMPC_BERT_EMBEDDING_OUTPUT", "/tmp/hpmpc_bert_embedding_output_q20.bin");

    // Public ids must be provided identically to every live party.  During the
    // initialization pass only their shape matters, so zero-filled placeholders
    // avoid filesystem I/O while preserving the circuit schedule.
    std::vector<std::uint32_t> token_ids(logical_tokens, 0U);
    std::vector<std::uint32_t> token_type_ids(logical_tokens, 0U);
    if (current_phase == PHASE_LIVE)
    {
        token_ids = bert_embedding_test::read_counted_u32(
            token_ids_path, static_cast<std::uint32_t>(logical_tokens));
        token_type_ids = bert_embedding_test::read_counted_u32(
            token_type_ids_path, static_cast<std::uint32_t>(logical_tokens));
    }

    bert_embedding::OwnerWeights owner_weights =
        bert_embedding_test::load_owner_weights(weights_directory);
    std::vector<A> output(packed_elements);
    bert_embedding::BertEmbedding<Share>().forward(
        token_ids, token_type_ids, batch, sequence_length, owner_weights, output.data());

    for (auto& value : output)
        value.prepare_reveal_to_all();
    Share::communicate();
    std::vector<DATATYPE> revealed(output.size());
    for (std::size_t index = 0; index < output.size(); ++index)
        revealed[index] = output[index].complete_reveal_to_all();
    *result = revealed.front();

    if (current_phase == PHASE_LIVE)
    {
        std::vector<std::int32_t> raw_output(
            logical_tokens * bert_embedding::kHidden);
        for (int packed_row = 0; packed_row < packed_rows; ++packed_row)
            for (int feature = 0; feature < bert_embedding::kHidden; ++feature)
            {
                alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
                unorthogonalize_arithmetic(
                    &revealed[static_cast<std::size_t>(packed_row) *
                                  bert_embedding::kHidden + feature],
                    lane_values.data(), 1);
                for (int lane = 0; lane < lanes; ++lane)
                {
                    const int logical_token = packed_row * lanes + lane;
                    if (logical_token < static_cast<int>(logical_tokens))
                        raw_output[static_cast<std::size_t>(logical_token) *
                                       bert_embedding::kHidden + feature] =
                            static_cast<std::int32_t>(lane_values[lane]);
                }
            }
        bert_embedding_test::write_counted_q20(output_path, raw_output);
    }
}
