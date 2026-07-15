#pragma once

#include "../../../datatypes/Additive_Share.hpp"
#include "../../../datatypes/float_fixed_converter.hpp"
#include "../comparisons.hpp"
#include "../exact_truncation.hpp"
#include "../max_min.hpp"

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
#define FUNCTION softmax_mpc_program
#endif

#ifndef SOFTMAX_BATCH
#define SOFTMAX_BATCH 2
#endif
#ifndef SOFTMAX_HEADS
#define SOFTMAX_HEADS 12
#endif
#ifndef SOFTMAX_QUERY_LEN
#define SOFTMAX_QUERY_LEN 128
#endif
#ifndef SOFTMAX_KEY_LEN
#define SOFTMAX_KEY_LEN 128
#endif

namespace transformer_mpc
{
constexpr int kExpRangeReductionBits = 4;
constexpr int kExpSquarings = 1 << kExpRangeReductionBits;
constexpr int kReciprocalIterations = 2;
constexpr int kReciprocalBuckets = 8;  // highest set bit for sums in [1, 128]

template <typename Share>
using AShare = Additive_Share<DATATYPE, Share>;

inline UINT_TYPE fixed(FLOATTYPE value)
{
    return FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(value);
}

template <typename Share>
void multiply_fixed(const AShare<Share>* lhs, const AShare<Share>* rhs, AShare<Share>* out, int len)
{
    using A = AShare<Share>;
    constexpr int limb_bits = FRACTIONAL / 2;
    constexpr UINT_TYPE limb_base = UINT_TYPE(1) << limb_bits;

    // Full-precision Q12.20 product in a 32-bit ring. For raw integers a,b:
    // a=ah*2^10+al and b=bh*2^10+bl. Then
    // (a*b)>>20 = ah*bh + (ah*bl+al*bh)>>10, up to <1 raw LSB.
    // Every secure product below fits in 32 bits for this protocol's ranges.
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
void multiply_bit_fixed(const AShare<Share>* bit, const AShare<Share>* value, AShare<Share>* out, int len)
{
    // `bit` is an arithmetic 0/1, not a Q12.20 value. Therefore bit*value
    // already has Q12.20 scale and must not be truncated.
    for (int i = 0; i < len; ++i)
    {
        out[i] = bit[i].prepare_dot(value[i]);
        out[i].mask_and_send_dot_without_trunc();
    }
    Share::communicate();
    for (int i = 0; i < len; ++i)
        out[i].complete_mult_without_trunc();
}

template <typename Share>
void divide_power_of_two(const AShare<Share>* in, int denominator, AShare<Share>* out, int len)
{
#if TRUNC_APPROACH == 2
    if (denominator <= 0 || (denominator & (denominator - 1)) != 0)
        throw std::invalid_argument("divide_power_of_two requires a positive power of two");
    int bits = 0;
    for (int value = denominator; value > 1; value >>= 1)
        ++bits;
    for (int i = 0; i < len; ++i)
        out[i] = in[i];
    trunc_exact_in_place(out, len, bits);
#else
    for (int i = 0; i < len; ++i)
        out[i] = in[i].prepare_div_exp2(denominator);
    Share::communicate();
    for (int i = 0; i < len; ++i)
        out[i].complete_public_mult_fixed();
#endif
}

// Secure negative exponential. delta must be x - row_max, hence delta <= 0.
// Values below -16*ln(2) are masked to zero. Remaining values are divided by
// 16, evaluated with 1+y+y^2/2+y^3/6 on y in [-ln(2),0], then squared four
// times to recover exp(delta).
template <typename Share>
void negative_exp(const AShare<Share>* delta, AShare<Share>* out, int len)
{
    using A = AShare<Share>;
    const UINT_TYPE one_fixed = fixed(FLOATTYPE(1));
    const UINT_TYPE lower_fixed = fixed(FLOATTYPE(-kExpSquarings) * std::log(FLOATTYPE(2)));

    std::vector<A> above_lower(len);
    std::vector<A> below_lower(len);
    std::vector<A> active(len);
    std::vector<A> clipped(len);
    for (int i = 0; i < len; ++i)
        above_lower[i] = delta[i] - A(lower_fixed);

    // LTZ returns arithmetic 0/1 (not fixed-point). This is the B-path
    // comparison, implemented by get_msb_range/A2B in the existing code.
    pack_additive<0, BITLENGTH>(
        above_lower.data(), below_lower.data(), len, LTZ<0, BITLENGTH, Share, DATATYPE>);

    // Reuse the LTZ result instead of invoking a second DReLU/MSB path:
    // max(delta, lower) = lower + (delta>=lower)*(delta-lower).
    for (int i = 0; i < len; ++i)
        active[i] = A(UINT_TYPE(1)) - below_lower[i];
    multiply_bit_fixed<Share>(active.data(), above_lower.data(), clipped.data(), len);
    for (int i = 0; i < len; ++i)
        clipped[i] = clipped[i] + A(lower_fixed);

    std::vector<A> y(len);
    divide_power_of_two<Share>(clipped.data(), kExpSquarings, y.data(), len);

    std::vector<A> y2(len);
    std::vector<A> y3(len);
    multiply_fixed<Share>(y.data(), y.data(), y2.data(), len);
    multiply_fixed<Share>(y2.data(), y.data(), y3.data(), len);

    std::vector<A> half_y2(len);
    divide_power_of_two<Share>(y2.data(), 2, half_y2.data(), len);
    std::vector<A> one_sixth(len, A(fixed(FLOATTYPE(1) / FLOATTYPE(6))));
    std::vector<A> sixth_y3(len);
    multiply_fixed<Share>(y3.data(), one_sixth.data(), sixth_y3.data(), len);
    for (int i = 0; i < len; ++i)
        out[i] = A(one_fixed) + y[i] + half_y2[i] + sixth_y3[i];

    std::vector<A> squared(len);
    for (int round = 0; round < kExpRangeReductionBits; ++round)
    {
        multiply_fixed<Share>(out, out, squared.data(), len);
        std::copy(squared.begin(), squared.end(), out);
    }

    // Apply the same arithmetic bit so the clipped tail becomes exactly zero.
    multiply_bit_fixed<Share>(active.data(), out, squared.data(), len);
    std::copy(squared.begin(), squared.end(), out);
}

// Lookup-based reciprocal seed followed by exactly two Newton-Raphson rounds.
// The highest-bit bucket k selects a linear seed on s in [2^k,2^(k+1)):
// r0=(48/17)2^-(k+1) - (32/17)s*2^-2(k+1).
template <typename Share>
void reciprocal_1_to_128(const AShare<Share>* sums, AShare<Share>* reciprocal, int rows)
{
    using A = AShare<Share>;
    const UINT_TYPE two_fixed = fixed(FLOATTYPE(2));

    std::vector<A> comparisons(kReciprocalBuckets * rows);
    std::vector<A> less_than(kReciprocalBuckets * rows);
    for (int k = 0; k < kReciprocalBuckets; ++k)
    {
        const UINT_TYPE threshold = fixed(std::ldexp(FLOATTYPE(1), k));
        for (int row = 0; row < rows; ++row)
            comparisons[k * rows + row] = sums[row] - A(threshold);
    }
    pack_additive<0, BITLENGTH>(comparisons.data(),
                                less_than.data(),
                                kReciprocalBuckets * rows,
                                LTZ<0, BITLENGTH, Share, DATATYPE>);

    std::vector<A> ge(kReciprocalBuckets * rows);
    std::vector<A> bucket(kReciprocalBuckets * rows);
    for (int k = 0; k < kReciprocalBuckets; ++k)
        for (int row = 0; row < rows; ++row)
            ge[k * rows + row] = A(UINT_TYPE(1)) - less_than[k * rows + row];

    for (int k = 0; k < kReciprocalBuckets - 1; ++k)
        for (int row = 0; row < rows; ++row)
            bucket[k * rows + row] = ge[k * rows + row] - ge[(k + 1) * rows + row];
    for (int row = 0; row < rows; ++row)
        bucket[(kReciprocalBuckets - 1) * rows + row] = ge[(kReciprocalBuckets - 1) * rows + row];

    std::vector<A> beta(rows, A(UINT_TYPE(0)));
    for (int row = 0; row < rows; ++row)
        reciprocal[row] = A(UINT_TYPE(0));
    for (int k = 0; k < kReciprocalBuckets; ++k)
    {
        const FLOATTYPE scale = std::ldexp(FLOATTYPE(1), -(k + 1));
        const UINT_TYPE alpha_k = fixed((FLOATTYPE(48) / FLOATTYPE(17)) * scale);
        const UINT_TYPE beta_k = fixed((FLOATTYPE(32) / FLOATTYPE(17)) * scale * scale);
        for (int row = 0; row < rows; ++row)
        {
            reciprocal[row] = reciprocal[row] + bucket[k * rows + row].mult_public(alpha_k);
            beta[row] = beta[row] + bucket[k * rows + row].mult_public(beta_k);
        }
    }
    std::vector<A> beta_times_sum(rows);
    multiply_fixed<Share>(beta.data(), sums, beta_times_sum.data(), rows);
    for (int row = 0; row < rows; ++row)
        reciprocal[row] = reciprocal[row] - beta_times_sum[row];

    std::vector<A> sum_times_r(rows);
    std::vector<A> correction(rows);
    std::vector<A> next(rows);
    for (int iteration = 0; iteration < kReciprocalIterations; ++iteration)
    {
        multiply_fixed<Share>(sums, reciprocal, sum_times_r.data(), rows);
        for (int row = 0; row < rows; ++row)
            correction[row] = A(two_fixed) - sum_times_r[row];
        multiply_fixed<Share>(reciprocal, correction.data(), next.data(), rows);
        std::copy(next.begin(), next.end(), reciprocal);
    }
}

// `packed_rows` counts Additive_Share rows. Each share contains
// DATTYPE/BITLENGTH SIMD lanes, so this directly supports DATTYPE=512.
template <typename Share>
void softmax_mpc_forward(const AShare<Share>* input,
                         AShare<Share>* output,
                         int packed_rows,
                         int row_width = SOFTMAX_KEY_LEN)
{
    using A = AShare<Share>;
    static_assert(BITLENGTH == 32, "softmax_mpc requires BITLENGTH=32");
    static_assert(FRACTIONAL == 20, "softmax_mpc requires FRACTIONAL=20");
    static_assert(TRUNC_APPROACH == 2, "softmax_mpc requires exact truncation (TRUNC_APPROACH=2)");

    if (packed_rows <= 0 || row_width <= 0 || row_width > 128)
        throw std::invalid_argument("softmax_mpc_forward expects positive dimensions and row_width <= 128");

    const int elements = packed_rows * row_width;
    std::vector<A> row_max(packed_rows);
    max_min_sint<0, BITLENGTH>(input, row_width, row_max.data(), packed_rows, true);

    std::vector<A> centered(elements);
    for (int row = 0; row < packed_rows; ++row)
        for (int col = 0; col < row_width; ++col)
            centered[row * row_width + col] = input[row * row_width + col] - row_max[row];

    std::vector<A> exponentials(elements);
    negative_exp<Share>(centered.data(), exponentials.data(), elements);

    std::vector<A> sums(packed_rows, A(UINT_TYPE(0)));
    for (int row = 0; row < packed_rows; ++row)
        for (int col = 0; col < row_width; ++col)
            sums[row] = sums[row] + exponentials[row * row_width + col];

    std::vector<A> reciprocal(packed_rows);
    reciprocal_1_to_128<Share>(sums.data(), reciprocal.data(), packed_rows);

    std::vector<A> broadcast_reciprocal(elements);
    for (int row = 0; row < packed_rows; ++row)
        for (int col = 0; col < row_width; ++col)
            broadcast_reciprocal[row * row_width + col] = reciprocal[row];
    multiply_fixed<Share>(exponentials.data(), broadcast_reciprocal.data(), output, elements);
}

struct TensorHeader
{
    std::uint32_t batch;
    std::uint32_t heads;
    std::uint32_t query_len;
    std::uint32_t key_len;
};

struct TensorData
{
    TensorHeader header;
    std::vector<float> values;
};

struct TensorPartHeader
{
    TensorHeader tensor;
    std::uint32_t logical_row_begin;
    std::uint32_t logical_row_count;
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

inline void validate_tensor_header(const TensorHeader& header)
{
    if (header.batch == 0 || header.heads == 0 || header.query_len == 0 || header.key_len == 0 ||
        header.key_len > 128)
        throw std::runtime_error("softmax tensor dimensions must be positive and key_len must be <= 128");
}

inline TensorHeader read_tensor_header(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open softmax input: " + path);
    TensorHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in)
        throw std::runtime_error("cannot read softmax input header: " + path);
    validate_tensor_header(header);
    return header;
}

inline TensorData read_input_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open softmax input: " + path);
    TensorData tensor{};
    in.read(reinterpret_cast<char*>(&tensor.header), sizeof(tensor.header));
    if (!in)
        throw std::runtime_error("cannot read softmax input header: " + path);
    validate_tensor_header(tensor.header);
    const TensorHeader& header = tensor.header;
    const std::size_t count = static_cast<std::size_t>(header.batch) * header.heads * header.query_len * header.key_len;
    tensor.values.resize(count);
    in.read(reinterpret_cast<char*>(tensor.values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!in || in.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("softmax input payload has the wrong byte length");
    return tensor;
}

inline void write_output_file(const std::string& path, const TensorHeader& header, const std::vector<float>& values)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        throw std::runtime_error("cannot open softmax output: " + path);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out)
        throw std::runtime_error("failed to write softmax output: " + path);
}

inline void write_output_part(const std::string& path,
                              const TensorHeader& tensor,
                              std::uint32_t logical_row_begin,
                              std::uint32_t logical_row_count,
                              const std::vector<float>& values)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        throw std::runtime_error("cannot open softmax output part: " + path);
    const TensorPartHeader header{tensor, logical_row_begin, logical_row_count};
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out)
        throw std::runtime_error("failed to write softmax output part: " + path);
}
}  // namespace transformer_mpc

// Standalone FUNCTION_IDENTIFIER=200 entry point. File I/O is intentionally a
// single-process verification harness; the core softmax_mpc_forward API remains
// SIMD- and split-role-compatible for integration into a Transformer layer.
template <typename Share>
void softmax_mpc_program(DATATYPE* result)
{
    using A = transformer_mpc::AShare<Share>;
    constexpr int lanes = DATTYPE / BITLENGTH;
    const std::string input_path =
        transformer_mpc::env_or_default("HPMPC_SOFTMAX_INPUT", "/tmp/hpmpc_softmax_input.bin");
    const std::string output_path =
        transformer_mpc::env_or_default("HPMPC_SOFTMAX_OUTPUT", "/tmp/hpmpc_softmax_output.bin");
    const transformer_mpc::TensorHeader header = transformer_mpc::read_tensor_header(input_path);
    const int logical_rows = static_cast<int>(header.batch * header.heads * header.query_len);
    const int row_width = static_cast<int>(header.key_len);
    const int total_packed_rows = (logical_rows + lanes - 1) / lanes;
    const int total_workers = transformer_mpc::positive_env_int("HPMPC_SOFTMAX_TOTAL_WORKERS", 1);
    const int worker = process_offset;
    if (worker < 0 || worker >= total_workers)
        throw std::runtime_error("process_offset is outside HPMPC_SOFTMAX_TOTAL_WORKERS");
    const int packed_row_begin = total_packed_rows * worker / total_workers;
    const int packed_row_end = total_packed_rows * (worker + 1) / total_workers;
    const int packed_rows = packed_row_end - packed_row_begin;
    if (packed_rows <= 0)
        throw std::runtime_error("too many Softmax workers for the number of SIMD-packed rows");
    const int packed_elements = packed_rows * row_width;

    std::vector<float> plaintext;
#if PARTY == 0
    if (current_phase == PHASE_LIVE)
        plaintext = transformer_mpc::read_input_file(input_path).values;
#endif

    std::vector<DATATYPE> packed_input(packed_elements);
#if PARTY == 0
    if (current_phase == PHASE_LIVE)
    {
        for (int local_packed_row = 0; local_packed_row < packed_rows; ++local_packed_row)
        {
            const int global_packed_row = packed_row_begin + local_packed_row;
            for (int col = 0; col < row_width; ++col)
            {
                alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
                for (int lane = 0; lane < lanes; ++lane)
                {
                    const int logical_row = global_packed_row * lanes + lane;
                    if (logical_row < logical_rows)
                    {
                        const float value = plaintext[static_cast<std::size_t>(logical_row) * row_width + col];
                        lane_values[lane] = transformer_mpc::fixed(static_cast<FLOATTYPE>(value));
                    }
                }
                orthogonalize_arithmetic(
                    lane_values.data(), &packed_input[local_packed_row * row_width + col], 1);
            }
        }
    }
#endif

    std::vector<A> shared_input(packed_elements);
    for (int i = 0; i < packed_elements; ++i)
        shared_input[i].template prepare_receive_from<P_0>(packed_input[i]);
    Share::communicate();
    for (int i = 0; i < packed_elements; ++i)
        shared_input[i].template complete_receive_from<P_0>();

    std::vector<A> shared_output(packed_elements);
    transformer_mpc::softmax_mpc_forward<Share>(
        shared_input.data(), shared_output.data(), packed_rows, row_width);

    for (int i = 0; i < packed_elements; ++i)
        shared_output[i].prepare_reveal_to_all();
    Share::communicate();

    std::vector<DATATYPE> revealed(packed_elements);
    for (int i = 0; i < packed_elements; ++i)
        revealed[i] = shared_output[i].complete_reveal_to_all();
    *result = revealed[0];

    if (current_phase == PHASE_LIVE)
    {
        const int logical_row_begin = packed_row_begin * lanes;
        const int logical_row_end = std::min(logical_rows, packed_row_end * lanes);
        const int local_logical_rows = logical_row_end - logical_row_begin;
        std::vector<float> output(static_cast<std::size_t>(local_logical_rows) * row_width);
        for (int local_packed_row = 0; local_packed_row < packed_rows; ++local_packed_row)
        {
            const int global_packed_row = packed_row_begin + local_packed_row;
            for (int col = 0; col < row_width; ++col)
            {
                alignas(sizeof(DATATYPE)) std::array<UINT_TYPE, lanes> lane_values{};
                unorthogonalize_arithmetic(
                    &revealed[local_packed_row * row_width + col], lane_values.data(), 1);
                for (int lane = 0; lane < lanes; ++lane)
                {
                    const int logical_row = global_packed_row * lanes + lane;
                    if (logical_row < logical_rows)
                    {
                        output[static_cast<std::size_t>(logical_row - logical_row_begin) * row_width + col] =
                            FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::ufixed_to_float(
                                lane_values[lane]);
                    }
                }
            }
        }
        if (total_workers == 1)
            transformer_mpc::write_output_file(output_path, header, output);
        else
            transformer_mpc::write_output_part(output_path + ".part-" + std::to_string(worker) + "-of-" +
                                                   std::to_string(total_workers) + ".bin",
                                               header,
                                               static_cast<std::uint32_t>(logical_row_begin),
                                               static_cast<std::uint32_t>(local_logical_rows),
                                               output);
    }
}
