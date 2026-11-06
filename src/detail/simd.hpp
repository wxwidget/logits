#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

#if LOGITLAB_SIMD && (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#include <immintrin.h>
#define LOGITLAB_AVX2 __attribute__((target("avx2,fma")))
#define LOGITLAB_HAS_AVX2 1
#else
#define LOGITLAB_HAS_AVX2 0
#endif

namespace logitlab::detail::simd {
inline bool enabled() {
#if LOGITLAB_HAS_AVX2
  static const bool available = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                                std::getenv("LOGITLAB_DISABLE_SIMD") == nullptr;
  return available;
#else
  return false;
#endif
}
#if LOGITLAB_HAS_AVX2
LOGITLAB_AVX2 inline bool copy_indices32(const std::int32_t* source, std::size_t* output,
                                         std::size_t size) {
  std::size_t j = 0;
  for (; j + 7 < size; j += 8) {
    const auto packet = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source + j));
    // A sign bit in any lane denotes a negative index. Widen four lanes at a
    // time into the owned 64-bit size_t storage; never narrow a valid index.
    if (_mm256_movemask_ps(_mm256_castsi256_ps(packet))) return false;
    const auto low = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(packet));
    const auto high = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(packet, 1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + j), low);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + j + 4), high);
  }
  for (; j < size; ++j) {
    if (source[j] < 0) return false;
    output[j] = static_cast<std::size_t>(source[j]);
  }
  return true;
}
LOGITLAB_AVX2 inline bool all_finite(const double* values, std::size_t size) {
  const auto sign = _mm256_set1_pd(-0.0);
  const auto maximum = _mm256_set1_pd(std::numeric_limits<double>::max());
  std::size_t j = 0;
  for (; j + 3 < size; j += 4) {
    const auto absolute = _mm256_andnot_pd(sign, _mm256_loadu_pd(values + j));
    // Ordered comparison rejects NaNs as well as either signed infinity.
    if (_mm256_movemask_pd(_mm256_cmp_pd(absolute, maximum, _CMP_LE_OQ)) != 15) return false;
  }
  for (; j < size; ++j)
    if (!std::isfinite(values[j])) return false;
  return true;
}
LOGITLAB_AVX2 inline double sum(__m256d value) {
  alignas(32) double lanes[4];
  _mm256_store_pd(lanes, value);
  return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
}
__attribute__((noinline, cold)) inline bool opposing_product_overflows(const double* x,
                                                                       const double* y,
                                                                       std::size_t size) {
  bool positive = false, negative = false;
  for (std::size_t j = 0; j < size; ++j) {
    const double product = x[j] * y[j];
    positive |= product == std::numeric_limits<double>::infinity();
    negative |= product == -std::numeric_limits<double>::infinity();
  }
  return positive && negative;
}
LOGITLAB_AVX2 inline double dense_dot(const double* x, const double* y, std::size_t size) {
  __m256d a = _mm256_setzero_pd(), b = a, c = a, d = a;
  std::size_t j = 0;
  for (; j + 15 < size; j += 16) {
    a = _mm256_fmadd_pd(_mm256_loadu_pd(x + j), _mm256_loadu_pd(y + j), a);
    b = _mm256_fmadd_pd(_mm256_loadu_pd(x + j + 4), _mm256_loadu_pd(y + j + 4), b);
    c = _mm256_fmadd_pd(_mm256_loadu_pd(x + j + 8), _mm256_loadu_pd(y + j + 8), c);
    d = _mm256_fmadd_pd(_mm256_loadu_pd(x + j + 12), _mm256_loadu_pd(y + j + 12), d);
  }
  double result = sum(_mm256_add_pd(_mm256_add_pd(a, b), _mm256_add_pd(c, d)));
  for (; j < size; ++j) result += x[j] * y[j];
  // Clang may fuse a tail multiply/add: fma(-2, DBL_MAX, +Inf) stays
  // infinite where separate overflowing products cancel to NaN. Preserve
  // rejection of that undefined score; ordinary finite dots skip this scan.
  if (__builtin_expect(std::isinf(result), 0) && opposing_product_overflows(x, y, size))
    return std::numeric_limits<double>::quiet_NaN();
  return result;
}
LOGITLAB_AVX2 inline void dense_add(const double* x, double scale, double* out, std::size_t size) {
  const auto alpha = _mm256_set1_pd(scale);
  std::size_t j = 0;
  for (; j + 3 < size; j += 4)
    _mm256_storeu_pd(out + j,
                     _mm256_fmadd_pd(alpha, _mm256_loadu_pd(x + j), _mm256_loadu_pd(out + j)));
  for (; j < size; ++j) out[j] += scale * x[j];
}
LOGITLAB_AVX2 inline void dense_moments(const double* x, double residual, double curvature,
                                        double* gradient, double* diagonal, std::size_t size) {
  const auto r = _mm256_set1_pd(residual), h = _mm256_set1_pd(curvature);
  std::size_t j = 0;
  for (; j + 3 < size; j += 4) {
    const auto values = _mm256_loadu_pd(x + j);
    _mm256_storeu_pd(gradient + j, _mm256_fmadd_pd(r, values, _mm256_loadu_pd(gradient + j)));
    // Multiply h*x first: h==0 must not turn a finite huge x into 0*Inf.
    _mm256_storeu_pd(diagonal + j, _mm256_fmadd_pd(_mm256_mul_pd(h, values), values,
                                                   _mm256_loadu_pd(diagonal + j)));
  }
  for (; j < size; ++j) {
    gradient[j] += residual * x[j];
    diagonal[j] += curvature * x[j] * x[j];
  }
}
#endif
}  // namespace logitlab::detail::simd
