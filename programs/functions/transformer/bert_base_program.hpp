#pragma once

// Production FUNCTION_IDENTIFIER=203: complete BERT-base sequence classifier.
// The graph is intentionally fixed-role and single-process.  With DATTYPE=512
// a private two-limb encoder layer is close to 1 GiB before transient GEMM
// buffers; creating split-role/process replicas would exceed the 28 GiB nodes.
//
// 调用链（实现都在 nn/PIGEON/architectures/，本文件只负责加载和释放）：
//   BertBase::embed
//     bert_embedding.hpp    公开 token 选行，三表相加后秘密分享，再 LayerNorm
//   循环 12 次 BertBase::forward_layer_in_place
//     bert_layer.hpp        Post-LN：Attention 残差 LN1，FFN 残差 LN2
//       bert_attention.hpp  QKV、本地 lane 重排、QK^T、-16 mask、Softmax、AV、输出投影
//       bert_ffn.hpp        gemm_fixed + gelu_mpc；本层关掉 FFN 自带的残差和 LN
//   BertBase::pool_and_classify
//     bert_pooler.hpp       本地收集 CLS，768→768，分段 tanh，768→2
// 两个 logit 在 reveal_logits 之后才变回明文，argmax 不进 MPC。
// MODELOWNER 读权重，DATAOWNER 不持有模型；token id、token type、padding mask 三方公开。
#define RESULTTYPE DATATYPE
#ifndef FUNCTION
#define FUNCTION bert_base_program
#endif

#include "../../../nn/PIGEON/architectures/bert_base.hpp"

// Reuse the shape-checked counted-file and private-weight loaders exercised by
// FIDs 206-208.  FUNCTION is already owned above, so these headers only add
// helpers and do not change the selected protocol entry point.
#include "bert_embedding_test.hpp"
#include "bert_layer_test.hpp"
#include "bert_pooler_test.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bert_base_program_detail
{
inline std::string checked_task()
{
    const std::string task = bert_ffn::env_or_default("HPMPC_BERT_BASE_TASK", "mrpc");
    if (task != "mrpc" && task != "rte")
        throw std::invalid_argument("HPMPC_BERT_BASE_TASK must be mrpc or rte");
    return task;
}

template <typename Share>
bert_pooler::PrivateLinear<Share> load_pooler(const std::string& directory)
{
    return bert_pooler_test::load_private_linear<Share>(
        directory, "pooler_weight.bin", "pooler_bias.bin",
        bert_pooler::kHidden, bert_pooler::kHidden);
}

template <typename Share>
bert_pooler::PrivateLinear<Share> load_classifier(const std::string& directory,
                                                  const std::string& task)
{
    const std::string weight = "classifier_" + task + "_weight.bin";
    const std::string bias = "classifier_" + task + "_bias.bin";
    return bert_pooler_test::load_private_linear<Share>(
        directory, weight.c_str(), bias.c_str(),
        bert_pooler::kClasses, bert_pooler::kHidden);
}

template <typename Share>
void reveal_logits(std::vector<bert_base::AShare<Share>>& logits,
                   int batch_size,
                   const std::string& output_path,
                   DATATYPE* result)
{
    constexpr int lanes = DATTYPE / BITLENGTH;
    const int packed_batch = bert_base::packed_batch_rows(batch_size);
    const std::size_t expected =
        static_cast<std::size_t>(packed_batch) * bert_base::kClasses;
    if (result == nullptr || logits.size() != expected)
        throw std::invalid_argument("BERT-base logits/result has the wrong shape");

    for (auto& value : logits)
        value.prepare_reveal_to_all();
    Share::communicate();
    std::vector<DATATYPE> revealed(logits.size());
    for (std::size_t index = 0; index < logits.size(); ++index)
        revealed[index] = logits[index].complete_reveal_to_all();
    *result = revealed.front();

    if (current_phase != PHASE_LIVE)
        return;
    std::vector<std::int32_t> raw(
        static_cast<std::size_t>(batch_size) * bert_base::kClasses);
    for (int pack = 0; pack < packed_batch; ++pack)
        for (int output_feature = 0; output_feature < bert_base::kClasses;
             ++output_feature)
        {
            alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
            unorthogonalize_arithmetic(
                &revealed[static_cast<std::size_t>(pack) * bert_base::kClasses +
                          output_feature],
                lane_values.data(), 1);
            for (int lane = 0; lane < lanes; ++lane)
            {
                const int example = pack * lanes + lane;
                if (example < batch_size)
                    raw[static_cast<std::size_t>(example) * bert_base::kClasses +
                        output_feature] = static_cast<std::int32_t>(lane_values[lane]);
            }
        }
    bert_ffn::write_counted_q20(output_path, raw);
}

// Diagnostic-only boundary reconstruction.  The environment variable that
// enables this helper is read in both INIT and LIVE, so the reveal is part of
// the initialized communication schedule on every party.  Production runs
// leave HPMPC_BERT_BASE_DEBUG_DIR unset and therefore reveal only final logits.
template <typename Share>
void reveal_activation(std::vector<bert_base::AShare<Share>>& activation,
                       int batch_size,
                       int sequence_length,
                       const std::string& output_path)
{
    constexpr int lanes = DATTYPE / BITLENGTH;
    const int logical_rows = batch_size * sequence_length;
    const int packed_rows = bert_base::packed_token_rows(batch_size, sequence_length);
    const std::size_t expected =
        static_cast<std::size_t>(packed_rows) * bert_base::kHidden;
    if (activation.size() != expected)
        throw std::invalid_argument("BERT-base debug activation has the wrong shape");

    for (auto& value : activation)
        value.prepare_reveal_to_all();
    Share::communicate();
    std::vector<DATATYPE> revealed(activation.size());
    for (std::size_t index = 0; index < activation.size(); ++index)
        revealed[index] = activation[index].complete_reveal_to_all();

    if (current_phase != PHASE_LIVE)
        return;
    std::vector<std::int32_t> raw(
        static_cast<std::size_t>(logical_rows) * bert_base::kHidden);
    for (int packed_row = 0; packed_row < packed_rows; ++packed_row)
        for (int feature = 0; feature < bert_base::kHidden; ++feature)
        {
            alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
            unorthogonalize_arithmetic(
                &revealed[static_cast<std::size_t>(packed_row) * bert_base::kHidden +
                          feature],
                lane_values.data(), 1);
            for (int lane = 0; lane < lanes; ++lane)
            {
                const int logical_row = packed_row * lanes + lane;
                if (logical_row < logical_rows)
                    raw[static_cast<std::size_t>(logical_row) * bert_base::kHidden +
                        feature] = static_cast<std::int32_t>(lane_values[lane]);
            }
        }
    bert_ffn::write_counted_q20(output_path, raw);
}

inline std::string debug_boundary_path(const std::string& directory,
                                       const std::string& filename)
{
    return directory.back() == '/' ? directory + filename
                                   : directory + "/" + filename;
}
}  // namespace bert_base_program_detail

// Environment contract (all public files must be identical on all parties):
//   HPMPC_BERT_BASE_WEIGHTS_DIR      default nn/Pygeon/weights/mrpc
//   HPMPC_BERT_BASE_TASK             mrpc|rte, default mrpc
//   HPMPC_BERT_BASE_BATCH            default 1
//   HPMPC_BERT_BASE_SEQUENCE         default 128
//   HPMPC_BERT_BASE_TOKEN_IDS        counted-u32 [B,L]
//   HPMPC_BERT_BASE_TOKEN_TYPE_IDS   counted-u32 [B,L]
//   HPMPC_BERT_BASE_ATTENTION_MASK   counted-u8  [B,L], 1=keep, 0=padding
//   HPMPC_BERT_BASE_OUTPUT           counted-Q20 [B,2]
//   HPMPC_BERT_BASE_DEBUG_DIR        optional; reveals embedding/layer boundaries
template <typename Share>
void bert_base_program(DATATYPE* result)
{
    using A = bert_base::AShare<Share>;
    if (process_offset != 0)
        throw std::runtime_error(
            "FID=203 requires PROCESS_NUM=1,SPLITROLES=0; encoder state is not sharded");

    const int batch = bert_ffn::positive_env_int("HPMPC_BERT_BASE_BATCH", 1);
    const int sequence_length =
        bert_ffn::positive_env_int("HPMPC_BERT_BASE_SEQUENCE", 128);
    const int packed_rows = bert_base::packed_token_rows(batch, sequence_length);
    const std::size_t logical_tokens = static_cast<std::size_t>(batch) * sequence_length;
    if (logical_tokens > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("BERT-base public token count exceeds counted-file format");

    const std::string task = bert_base_program_detail::checked_task();
    const std::string weights_directory = bert_ffn::env_or_default(
        "HPMPC_BERT_BASE_WEIGHTS_DIR", "nn/Pygeon/weights/mrpc");
    const std::string token_ids_path = bert_ffn::env_or_default(
        "HPMPC_BERT_BASE_TOKEN_IDS", "/tmp/hpmpc_bert_base_token_ids_u32.bin");
    const std::string token_type_ids_path = bert_ffn::env_or_default(
        "HPMPC_BERT_BASE_TOKEN_TYPE_IDS",
        "/tmp/hpmpc_bert_base_token_type_ids_u32.bin");
    const std::string attention_mask_path = bert_ffn::env_or_default(
        "HPMPC_BERT_BASE_ATTENTION_MASK",
        "/tmp/hpmpc_bert_base_attention_mask_u8.bin");
    const std::string output_path = bert_ffn::env_or_default(
        "HPMPC_BERT_BASE_OUTPUT", "/tmp/hpmpc_bert_base_logits_q20.bin");
    const std::string debug_directory =
        bert_ffn::env_or_default("HPMPC_BERT_BASE_DEBUG_DIR", "");

    // Token and type ids are public.  Initialization requires only dimensions,
    // while each live party independently reads the same counted files.
    std::vector<std::uint32_t> token_ids(logical_tokens, 0U);
    std::vector<std::uint32_t> token_type_ids(logical_tokens, 0U);
    std::vector<std::uint8_t> keep_mask(logical_tokens, std::uint8_t(1));
    if (current_phase == PHASE_LIVE)
    {
        token_ids = bert_embedding_test::read_counted_u32(
            token_ids_path, static_cast<std::uint32_t>(logical_tokens));
        token_type_ids = bert_embedding_test::read_counted_u32(
            token_type_ids_path, static_cast<std::uint32_t>(logical_tokens));
        keep_mask = bert_layer_test::read_public_counted_u8(
            attention_mask_path, static_cast<std::uint32_t>(logical_tokens));
    }

    bert_base::BertBase<Share> graph;
    std::vector<A> activation;

    // Scope the 23.4M-entry owner embedding table so it is released before
    // the first private encoder layer is allocated.  BertEmbedding shares only
    // selected rows and private LayerNorm affine parameters.
    {
        bert_embedding::OwnerWeights embedding_weights =
            bert_embedding_test::load_owner_weights(weights_directory);
        activation = graph.embed(token_ids, token_type_ids, batch, sequence_length,
                                 embedding_weights);
    }
    if (activation.size() != static_cast<std::size_t>(packed_rows) * bert_base::kHidden)
        throw std::runtime_error("BERT-base embedding produced an invalid layout");
    if (!debug_directory.empty())
        bert_base_program_detail::reveal_activation<Share>(
            activation, batch, sequence_length,
            bert_base_program_detail::debug_boundary_path(
                debug_directory, "boundary_00_embedding_q20.bin"));

    // Critical memory invariant: allocate/share one layer, run it, swap the
    // activation, then destroy that layer's high/low weights before loading
    // the next.  Never retain a vector<ModelShares> for all twelve layers.
    // load_and_share_layer 按层把 QKV、Attention 输出、FFN 权重拆成 high/low 分享；
    // layer 是循环局部变量，本轮 forward_layer_in_place 返回后即释放。
    for (int layer_index = 0; layer_index < bert_base::kLayers; ++layer_index)
    {
        bert_layer_test::ModelShares<Share> layer =
            bert_layer_test::load_and_share_layer<Share>(
                weights_directory, layer_index);
        graph.forward_layer_in_place(
            activation, layer.view(), batch, sequence_length, keep_mask.data());
        if (!debug_directory.empty())
            bert_base_program_detail::reveal_activation<Share>(
                activation, batch, sequence_length,
                bert_base_program_detail::debug_boundary_path(
                    debug_directory,
                    "boundary_" + std::to_string(layer_index + 1) + "_q20.bin"));
    }

    // Load these two small private Linears only after all encoder-layer
    // weights have been destroyed.  Separate task bundles select separate
    // fine-tuned backbones and heads via WEIGHTS_DIR+TASK.
    const auto pooler =
        bert_base_program_detail::load_pooler<Share>(weights_directory);
    const auto classifier =
        bert_base_program_detail::load_classifier<Share>(weights_directory, task);
    std::vector<A> logits = graph.pool_and_classify(
        activation, batch, sequence_length, pooler, classifier);

    // Reconstruct exactly two logits per sample.  Argmax is intentionally
    // plaintext; classification does not require an MPC Softmax.
    bert_base_program_detail::reveal_logits<Share>(
        logits, batch, output_path, result);
}
