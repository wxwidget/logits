#pragma once

#include <cmath>

#include "detail/simd.hpp"

// glibc's vector math ABI is optional. No fast-math flags or reduced-precision
// approximations are used; unsupported platforms retain standard scalar math.
namespace logitlab::detail::vector_math {
inline bool available() {
#if LOGITLAB_HAVE_VECMATH && LOGITLAB_HAS_AVX2
  return simd::enabled();
#else
  return false;
#endif
}
#if LOGITLAB_HAVE_VECMATH && LOGITLAB_HAS_AVX2
extern "C" LOGITLAB_AVX2 __m256d _ZGVdN4v_exp(__m256d);
extern "C" LOGITLAB_AVX2 __m256d _ZGVdN4v_log1p(__m256d);
LOGITLAB_AVX2 inline void exp4(const double* input, double* output) {
  _mm256_storeu_pd(output, _ZGVdN4v_exp(_mm256_loadu_pd(input)));
}
LOGITLAB_AVX2 inline void log1p4(const double* input, double* output) {
  _mm256_storeu_pd(output, _ZGVdN4v_log1p(_mm256_loadu_pd(input)));
}
#else
inline void exp4(const double* input, double* output) {
  for (std::size_t i = 0; i < 4; ++i) output[i] = std::exp(input[i]);
}
inline void log1p4(const double* input, double* output) {
  for (std::size_t i = 0; i < 4; ++i) output[i] = std::log1p(input[i]);
}
#endif
}  // namespace logitlab::detail::vector_math
