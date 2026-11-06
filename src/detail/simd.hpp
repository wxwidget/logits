#pragma once

#include <cmath>
#include <cstddef>
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
