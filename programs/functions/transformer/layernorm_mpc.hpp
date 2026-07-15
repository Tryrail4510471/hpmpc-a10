#pragma once

#include "../../../datatypes/Additive_Share.hpp"
#include "../../../datatypes/float_fixed_converter.hpp"
#include "../comparisons.hpp"
#include "../exact_truncation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define RESULTTYPE DATATYPE
#ifndef FUNCTION
#define FUNCTION layernorm_mpc_program
#endif

namespace layernorm_mpc
{
constexpr int kHiddenSize = 768;
constexpr int kNewtonIterations = 2;
constexpr int kLutEntries = 8;
constexpr FLOATTYPE kEpsilon = FLOATTYPE(1.0e-5);

template <typename Share>
using AShare = Additive_Share<DATATYPE, Share>;

inline UINT_TYPE fixed(FLOATTYPE value)
{
    return FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(value);
}

// Exact Q12.20 multiplication. Splitting each operand into two 10-bit limbs
// prevents the raw 32-bit ring product from discarding the high bits needed by
// a conventional (a*b)>>20 implementation. Every product is truncated before
// it is consumed by the next polynomial/Newton operation.
template <typename Share>
void multiply_fixed(const AShare<Share>* lhs, const AShare<Share>* rhs, AShare<Share>* out, int len)
{
    using A = AShare<Share>;
    constexpr int limb_bits = FRACTIONAL / 2;
    constexpr UINT_TYPE limb_base = UINT_TYPE(1) << limb_bits;

    std::vector<A> high_lhs(lhs, lhs + len);
    std::vector<A> high_rhs(rhs, rhs + len);
    trunc_exact_in_place(high_lhs.data(), len, limb_bits);
    trunc_exact_in_place(high_rhs.data(), len, limb_bits);

    std::vector<A> low_lhs(len);
    std::vector<A> low_rhs(len);
    for (int i = 0; i < len; ++i)
    {
        low_lhs[i] = lhs[i] - high_lhs[i].mult_public(limb_base);
        low_rhs[i] = rhs[i] - high_rhs[i].mult_public(limb_base);
    }

    std::vector<A> cross(len);
    for (int i = 0; i < len; ++i)
    {
        out[i] = high_lhs[i].prepare_dot(high_rhs[i]);
        cross[i] = high_lhs[i].prepare_dot(low_rhs[i]) + low_lhs[i].prepare_dot(high_rhs[i]);
        out[i].mask_and_send_dot_without_trunc();
        cross[i].mask_and_send_dot_without_trunc();
    }
    Share::communicate();
    for (int i = 0; i < len; ++i)
    {
        out[i].complete_mult_without_trunc();
        cross[i].complete_mult_without_trunc();
    }
    trunc_exact_in_place(cross.data(), len, limb_bits);
    for (int i = 0; i < len; ++i)
        out[i] = out[i] + cross[i];
}

template <typename Share>
void multiply_public_fixed(const AShare<Share>* input, UINT_TYPE constant, AShare<Share>* out, int len)
{
    using A = AShare<Share>;
    constexpr int limb_bits = FRACTIONAL / 2;
    constexpr UINT_TYPE limb_base = UINT_TYPE(1) << limb_bits;
    const INT_TYPE signed_constant = static_cast<INT_TYPE>(constant);
    const INT_TYPE constant_high = signed_constant >> limb_bits;
    const INT_TYPE constant_low = signed_constant - constant_high * static_cast<INT_TYPE>(limb_base);

    std::vector<A> input_high(input, input + len);
    trunc_exact_in_place(input_high.data(), len, limb_bits);
    std::vector<A> cross(len);
    for (int i = 0; i < len; ++i)
    {
        const A input_low = input[i] - input_high[i].mult_public(limb_base);
        cross[i] = input_high[i].mult_public(static_cast<UINT_TYPE>(constant_low)) +
                   input_low.mult_public(static_cast<UINT_TYPE>(constant_high));
        out[i] = input_high[i].mult_public(static_cast<UINT_TYPE>(constant_high));
    }
    trunc_exact_in_place(cross.data(), len, limb_bits);
    for (int i = 0; i < len; ++i)
        out[i] = out[i] + cross[i];
}

template <typename Share>
void multiply_public_fixed_vector(const AShare<Share>* input,
                                  const std::vector<UINT_TYPE>& constants,
                                  AShare<Share>* out,
                                  int packed_rows,
                                  int width)
{
    using A = AShare<Share>;
    constexpr int limb_bits = FRACTIONAL / 2;
    constexpr UINT_TYPE limb_base = UINT_TYPE(1) << limb_bits;
    const int elements = packed_rows * width;
    std::vector<A> input_high(input, input + elements);
    std::vector<A> cross(elements);
    trunc_exact_in_place(input_high.data(), elements, limb_bits);
    for (int row = 0; row < packed_rows; ++row)
        for (int col = 0; col < width; ++col)
        {
            const int index = row * width + col;
            const INT_TYPE signed_constant = static_cast<INT_TYPE>(constants[col]);
            const INT_TYPE constant_high = signed_constant >> limb_bits;
            const INT_TYPE constant_low =
                signed_constant - constant_high * static_cast<INT_TYPE>(limb_base);
            const A input_low = input[index] - input_high[index].mult_public(limb_base);
            cross[index] = input_high[index].mult_public(static_cast<UINT_TYPE>(constant_low)) +
                           input_low.mult_public(static_cast<UINT_TYPE>(constant_high));
            out[index] = input_high[index].mult_public(static_cast<UINT_TYPE>(constant_high));
        }
    trunc_exact_in_place(cross.data(), elements, limb_bits);
    for (int i = 0; i < elements; ++i)
        out[i] = out[i] + cross[i];
}

template <typename Share>
void multiply_bit(const AShare<Share>* bit, const AShare<Share>* value, AShare<Share>* out, int len)
{
    // bit is an arithmetic 0/1 share, so bit*value already has Q12.20 scale.
    for (int i = 0; i < len; ++i)
    {
        out[i] = bit[i].prepare_dot(value[i]);
        out[i].mask_and_send_dot_without_trunc();
    }
    Share::communicate();
    for (int i = 0; i < len; ++i)
        out[i].complete_mult_without_trunc();
}

// Eight magnitude buckets provide a public-constant seed. The secret bucket
// bits are obtained from the existing LTZ/A2B comparison and returned as
// arithmetic 0/1 shares. This is the secure equivalent of decoding a small
// LUT index; directly shifting an additive share is not valid.
template <typename Share>
void inverse_sqrt_seed(const AShare<Share>* z, AShare<Share>* seed, int rows)
{
    using A = AShare<Share>;
    constexpr std::array<FLOATTYPE, kLutEntries - 1> thresholds = {
        FLOATTYPE(0.25), FLOATTYPE(0.5), FLOATTYPE(0.75), FLOATTYPE(1.0),
        FLOATTYPE(1.5),  FLOATTYPE(2.0), FLOATTYPE(4.0)};
    // Seeds are 1/sqrt(bucket midpoint), rounded to Q12.20. The intervals are
    // deliberately narrow around variance 1, the operating range of BERT.
    constexpr std::array<FLOATTYPE, kLutEntries> seeds = {
        FLOATTYPE(2.000000000), FLOATTYPE(1.632993162), FLOATTYPE(1.264911064),
        FLOATTYPE(1.069044968), FLOATTYPE(0.894427191), FLOATTYPE(0.755928946),
        FLOATTYPE(0.577350269), FLOATTYPE(0.353553391)};

    std::vector<A> comparisons((kLutEntries - 1) * rows);
    std::vector<A> less((kLutEntries - 1) * rows);
    for (int k = 0; k < kLutEntries - 1; ++k)
        for (int row = 0; row < rows; ++row)
            comparisons[k * rows + row] = z[row] - A(fixed(thresholds[k]));
    pack_additive<0, BITLENGTH>(comparisons.data(),
                                less.data(),
                                (kLutEntries - 1) * rows,
                                LTZ<0, BITLENGTH, Share, DATATYPE>);

    std::vector<A> ge((kLutEntries - 1) * rows);
    std::vector<A> bucket(kLutEntries * rows);
    for (int k = 0; k < kLutEntries - 1; ++k)
        for (int row = 0; row < rows; ++row)
            ge[k * rows + row] = A(UINT_TYPE(1)) - less[k * rows + row];
    for (int row = 0; row < rows; ++row)
    {
        bucket[row] = less[row];
        for (int k = 1; k < kLutEntries - 1; ++k)
            bucket[k * rows + row] = ge[(k - 1) * rows + row] - ge[k * rows + row];
        bucket[(kLutEntries - 1) * rows + row] = ge[(kLutEntries - 2) * rows + row];
    }

    for (int row = 0; row < rows; ++row)
    {
        seed[row] = A(UINT_TYPE(0));
        for (int k = 0; k < kLutEntries; ++k)
            seed[row] = seed[row] + bucket[k * rows + row].mult_public(fixed(seeds[k]));
    }
}

template <typename Share>
void inverse_sqrt(const AShare<Share>* z, AShare<Share>* reciprocal_sqrt, int rows)
{
    using A = AShare<Share>;
    inverse_sqrt_seed<Share>(z, reciprocal_sqrt, rows);

    std::vector<A> r2(rows);
    std::vector<A> zr2(rows);
    std::vector<A> correction(rows);
    std::vector<A> next(rows);
    for (int iteration = 0; iteration < kNewtonIterations; ++iteration)
    {
        multiply_fixed<Share>(reciprocal_sqrt, reciprocal_sqrt, r2.data(), rows);
        multiply_fixed<Share>(z, r2.data(), zr2.data(), rows);
        for (int row = 0; row < rows; ++row)
            correction[row] = A(fixed(FLOATTYPE(3))) - zr2[row];
        multiply_fixed<Share>(reciprocal_sqrt, correction.data(), next.data(), rows);
        trunc_exact_in_place(next.data(), rows, 1);  // divide by the public constant 2
        std::copy(next.begin(), next.end(), reciprocal_sqrt);
    }
}

template <typename Share>
void layernorm_mpc_forward(const AShare<Share>* input,
                           const std::vector<UINT_TYPE>& gamma,
                           const std::vector<UINT_TYPE>& beta,
                           AShare<Share>* output,
                           int packed_rows,
                           int hidden_size = kHiddenSize)
{
    using A = AShare<Share>;
    static_assert(BITLENGTH == 32, "layernorm_mpc requires BITLENGTH=32");
    static_assert(FRACTIONAL == 20, "layernorm_mpc requires FRACTIONAL=20");
    static_assert(TRUNC_APPROACH == 2, "layernorm_mpc requires exact truncation (TRUNC_APPROACH=2)");
    if (packed_rows <= 0 || hidden_size <= 0 || hidden_size > kHiddenSize)
        throw std::invalid_argument("layernorm_mpc_forward expects 0 < hidden_size <= 768");
    if (gamma.size() != static_cast<std::size_t>(hidden_size) ||
        beta.size() != static_cast<std::size_t>(hidden_size))
        throw std::invalid_argument("LayerNorm gamma/beta length must equal hidden_size");

    const int elements = packed_rows * hidden_size;
    const UINT_TYPE reciprocal_hidden = fixed(FLOATTYPE(1) / hidden_size);
    std::vector<A> mean(packed_rows);
    if (hidden_size == kHiddenSize)
    {
        // 768 = 3*256. Averaging each triplet before the reduction keeps a
        // constant input such as 5.0 inside the signed Q12.20 ring. The final
        // division by 256 is an exact arithmetic right shift.
        constexpr int groups = kHiddenSize / 3;
        std::vector<A> triplet_sums(packed_rows * groups);
        std::vector<A> triplet_means(packed_rows * groups);
        for (int row = 0; row < packed_rows; ++row)
            for (int group = 0; group < groups; ++group)
            {
                const int base = row * hidden_size + group * 3;
                triplet_sums[row * groups + group] = input[base] + input[base + 1] + input[base + 2];
            }
        multiply_public_fixed<Share>(triplet_sums.data(), fixed(FLOATTYPE(1) / FLOATTYPE(3)),
                                     triplet_means.data(), packed_rows * groups);
        for (int row = 0; row < packed_rows; ++row)
        {
            mean[row] = A(UINT_TYPE(0));
            for (int group = 0; group < groups; ++group)
                mean[row] = mean[row] + triplet_means[row * groups + group];
        }
        trunc_exact_in_place(mean.data(), packed_rows, 8);
    }
    else
    {
        std::vector<A> sums(packed_rows, A(UINT_TYPE(0)));
        for (int row = 0; row < packed_rows; ++row)
            for (int col = 0; col < hidden_size; ++col)
                sums[row] = sums[row] + input[row * hidden_size + col];
        multiply_public_fixed<Share>(sums.data(), reciprocal_hidden, mean.data(), packed_rows);
    }

    std::vector<A> x2(elements);
    multiply_fixed<Share>(input, input, x2.data(), elements);
    // Scale square terms before reduction. For variance around 3, summing 768
    // Q12.20 squares first can exceed the signed 32-bit ring range.
    std::vector<A> scaled_x2(elements);
    multiply_public_fixed<Share>(x2.data(), reciprocal_hidden, scaled_x2.data(), elements);
    std::vector<A> ex2(packed_rows, A(UINT_TYPE(0)));
    for (int row = 0; row < packed_rows; ++row)
        for (int col = 0; col < hidden_size; ++col)
            ex2[row] = ex2[row] + scaled_x2[row * hidden_size + col];
    std::vector<A> mean2(packed_rows);
    multiply_fixed<Share>(mean.data(), mean.data(), mean2.data(), packed_rows);
    std::vector<A> raw_variance(packed_rows);
    for (int row = 0; row < packed_rows; ++row)
        raw_variance[row] = ex2[row] - mean2[row];

    // E[x^2]-mean^2 can be a few fixed-point LSBs negative for a constant
    // vector. Clamp that representation error to zero before adding epsilon.
    std::vector<A> negative(packed_rows);
    pack_additive<0, BITLENGTH>(
        raw_variance.data(), negative.data(), packed_rows, LTZ<0, BITLENGTH, Share, DATATYPE>);
    std::vector<A> nonnegative_bit(packed_rows);
    for (int row = 0; row < packed_rows; ++row)
        nonnegative_bit[row] = A(UINT_TYPE(1)) - negative[row];
    std::vector<A> nonnegative_variance(packed_rows);
    multiply_bit<Share>(nonnegative_bit.data(), raw_variance.data(), nonnegative_variance.data(), packed_rows);
    std::vector<A> variance_eps(packed_rows);
    for (int row = 0; row < packed_rows; ++row)
        variance_eps[row] = nonnegative_variance[row] + A(fixed(kEpsilon));

    std::vector<A> reciprocal_sqrt(packed_rows);
    inverse_sqrt<Share>(variance_eps.data(), reciprocal_sqrt.data(), packed_rows);

    std::vector<A> centered(elements);
    std::vector<A> broadcast_r(elements);
    for (int row = 0; row < packed_rows; ++row)
        for (int col = 0; col < hidden_size; ++col)
        {
            const int index = row * hidden_size + col;
            centered[index] = input[index] - mean[row];
            broadcast_r[index] = reciprocal_sqrt[row];
        }
    std::vector<A> normalized(elements);
    multiply_fixed<Share>(centered.data(), broadcast_r.data(), normalized.data(), elements);
    multiply_public_fixed_vector<Share>(normalized.data(), gamma, output, packed_rows, hidden_size);
    for (int row = 0; row < packed_rows; ++row)
        for (int col = 0; col < hidden_size; ++col)
            output[row * hidden_size + col] = output[row * hidden_size + col] + A(beta[col]);
}

struct TensorHeader
{
    std::uint32_t batch;
    std::uint32_t sequence;
    std::uint32_t hidden;
};

struct TensorPartHeader
{
    TensorHeader tensor;
    std::uint32_t logical_row_begin;
    std::uint32_t logical_row_count;
};

struct TensorData
{
    TensorHeader header;
    std::vector<float> input;
    std::vector<float> gamma;
    std::vector<float> beta;
};

inline const char* env_or_default(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : value;
}

inline int positive_env_int(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr)
        return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max())
        throw std::runtime_error(std::string(name) + " must be a positive integer");
    return static_cast<int>(parsed);
}

inline void validate_header(const TensorHeader& header)
{
    if (header.batch == 0 || header.sequence == 0 || header.hidden == 0 || header.hidden > kHiddenSize)
        throw std::runtime_error("LayerNorm dimensions must be positive and hidden_size must be <= 768");
}

inline TensorHeader read_header(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open LayerNorm input: " + path);
    TensorHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in)
        throw std::runtime_error("cannot read LayerNorm input header: " + path);
    validate_header(header);
    return header;
}

inline TensorData read_input_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open LayerNorm input: " + path);
    TensorData tensor{};
    in.read(reinterpret_cast<char*>(&tensor.header), sizeof(tensor.header));
    if (!in)
        throw std::runtime_error("cannot read LayerNorm input header: " + path);
    validate_header(tensor.header);
    const std::size_t elements = static_cast<std::size_t>(tensor.header.batch) * tensor.header.sequence *
                                 tensor.header.hidden;
    tensor.input.resize(elements);
    tensor.gamma.resize(tensor.header.hidden);
    tensor.beta.resize(tensor.header.hidden);
    in.read(reinterpret_cast<char*>(tensor.input.data()), static_cast<std::streamsize>(elements * sizeof(float)));
    in.read(reinterpret_cast<char*>(tensor.gamma.data()),
            static_cast<std::streamsize>(tensor.gamma.size() * sizeof(float)));
    in.read(reinterpret_cast<char*>(tensor.beta.data()),
            static_cast<std::streamsize>(tensor.beta.size() * sizeof(float)));
    if (!in || in.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("LayerNorm input payload has the wrong byte length");
    return tensor;
}

inline void write_output_file(const std::string& path, const TensorHeader& header, const std::vector<float>& values)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        throw std::runtime_error("cannot open LayerNorm output: " + path);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out)
        throw std::runtime_error("failed to write LayerNorm output: " + path);
}

inline void write_output_part(const std::string& path,
                              const TensorHeader& tensor,
                              std::uint32_t logical_row_begin,
                              std::uint32_t logical_row_count,
                              const std::vector<float>& values)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        throw std::runtime_error("cannot open LayerNorm output part: " + path);
    const TensorPartHeader header{tensor, logical_row_begin, logical_row_count};
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out)
        throw std::runtime_error("failed to write LayerNorm output part: " + path);
}
}  // namespace layernorm_mpc

template <typename Share>
void layernorm_mpc_program(DATATYPE* result)
{
    using A = layernorm_mpc::AShare<Share>;
    constexpr int lanes = DATTYPE / BITLENGTH;
    const std::string input_path =
        layernorm_mpc::env_or_default("HPMPC_LAYERNORM_INPUT", "/tmp/hpmpc_layernorm_input.bin");
    const std::string output_path =
        layernorm_mpc::env_or_default("HPMPC_LAYERNORM_OUTPUT", "/tmp/hpmpc_layernorm_output.bin");
    const layernorm_mpc::TensorHeader header = layernorm_mpc::read_header(input_path);
    const int logical_rows = static_cast<int>(header.batch * header.sequence);
    const int hidden = static_cast<int>(header.hidden);
    const int total_workers = layernorm_mpc::positive_env_int("HPMPC_LAYERNORM_TOTAL_WORKERS", 1);
    const int worker = process_offset;
    if (worker < 0 || worker >= total_workers)
        throw std::runtime_error("process_offset is outside HPMPC_LAYERNORM_TOTAL_WORKERS");
    const int worker_row_begin = logical_rows * worker / total_workers;
    const int worker_row_end = logical_rows * (worker + 1) / total_workers;
    const int worker_rows = worker_row_end - worker_row_begin;
    if (worker_rows <= 0)
        throw std::runtime_error("too many LayerNorm workers for the number of logical rows");
    const int packed_rows = (worker_rows + lanes - 1) / lanes;
    const int elements = packed_rows * hidden;

    layernorm_mpc::TensorData tensor;
    if (current_phase == PHASE_LIVE)
        tensor = layernorm_mpc::read_input_file(input_path);
    std::vector<UINT_TYPE> gamma(hidden, layernorm_mpc::fixed(FLOATTYPE(1)));
    std::vector<UINT_TYPE> beta(hidden, layernorm_mpc::fixed(FLOATTYPE(0)));
    if (current_phase == PHASE_LIVE)
        for (int col = 0; col < hidden; ++col)
        {
            gamma[col] = layernorm_mpc::fixed(static_cast<FLOATTYPE>(tensor.gamma[col]));
            beta[col] = layernorm_mpc::fixed(static_cast<FLOATTYPE>(tensor.beta[col]));
        }

    std::vector<DATATYPE> packed_input(elements);
#if PARTY == 0
    if (current_phase == PHASE_LIVE)
        for (int local_packed_row = 0; local_packed_row < packed_rows; ++local_packed_row)
        {
            for (int col = 0; col < hidden; ++col)
            {
                alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
                for (int lane = 0; lane < lanes; ++lane)
                {
                    const int logical_row = worker_row_begin + local_packed_row * lanes + lane;
                    if (logical_row < worker_row_end)
                        lane_values[lane] = layernorm_mpc::fixed(static_cast<FLOATTYPE>(
                            tensor.input[static_cast<std::size_t>(logical_row) * hidden + col]));
                }
                orthogonalize_arithmetic(
                    lane_values.data(), &packed_input[local_packed_row * hidden + col], 1);
            }
        }
#endif

    std::vector<A> shared_input(elements);
    for (int i = 0; i < elements; ++i)
        shared_input[i].template prepare_receive_from<P_0>(packed_input[i]);
    Share::communicate();
    for (int i = 0; i < elements; ++i)
        shared_input[i].template complete_receive_from<P_0>();

    std::vector<A> shared_output(elements);
    layernorm_mpc::layernorm_mpc_forward<Share>(
        shared_input.data(), gamma, beta, shared_output.data(), packed_rows, hidden);

    for (int i = 0; i < elements; ++i)
        shared_output[i].prepare_reveal_to_all();
    Share::communicate();
    std::vector<DATATYPE> revealed(elements);
    for (int i = 0; i < elements; ++i)
        revealed[i] = shared_output[i].complete_reveal_to_all();
    *result = revealed[0];

    if (current_phase == PHASE_LIVE)
    {
        std::vector<float> output(static_cast<std::size_t>(worker_rows) * hidden);
        for (int local_packed_row = 0; local_packed_row < packed_rows; ++local_packed_row)
        {
            for (int col = 0; col < hidden; ++col)
            {
                alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
                unorthogonalize_arithmetic(
                    &revealed[local_packed_row * hidden + col], lane_values.data(), 1);
                for (int lane = 0; lane < lanes; ++lane)
                {
                    const int logical_row = worker_row_begin + local_packed_row * lanes + lane;
                    if (logical_row < worker_row_end)
                        output[static_cast<std::size_t>(logical_row - worker_row_begin) * hidden + col] =
                            FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::ufixed_to_float(
                                lane_values[lane]);
                }
            }
        }
        if (total_workers == 1)
            layernorm_mpc::write_output_file(output_path, header, output);
        else
            layernorm_mpc::write_output_part(output_path + ".part-" + std::to_string(worker) + "-of-" +
                                                 std::to_string(total_workers) + ".bin",
                                             header,
                                             static_cast<std::uint32_t>(worker_row_begin),
                                             static_cast<std::uint32_t>(worker_rows),
                                             output);
    }
}
