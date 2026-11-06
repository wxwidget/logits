#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "absl/types/span.h"
#include "detail/simd.hpp"

namespace logitlab::detail {
inline void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
inline bool finite(absl::Span<const double> values) {
#if LOGITLAB_HAS_AVX2
  if (simd::enabled()) return simd::all_finite(values.data(), values.size());
#endif
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}
inline bool zero_coefficients(absl::Span<const double> parameters, std::size_t features) {
  return std::all_of(parameters.begin(), parameters.begin() + features,
                     [](double value) { return value == 0.; });
}
// Independent accumulators expose instruction-level parallelism without
// fast-math. Reassociation changes rounding, never the convergence criterion.
inline double dot(absl::Span<const double> a, absl::Span<const double> b) {
  double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  std::size_t j = 0;
  for (; j + 3 < a.size(); j += 4) {
    s0 += a[j] * b[j];
    s1 += a[j + 1] * b[j + 1];
    s2 += a[j + 2] * b[j + 2];
    s3 += a[j + 3] * b[j + 3];
  }
  double result = (s0 + s1) + (s2 + s3);
  for (; j < a.size(); ++j) result += a[j] * b[j];
  return result;
}
inline double norm(absl::Span<const double> values) {
  const double square = dot(values, values);
  if (std::isnan(square)) return square;
  if (square > 0 && std::isfinite(square)) return std::sqrt(square);
  // Rare overflow/underflow path; ordinary optimizer vectors use the fast sum.
  double scale = 0;
  for (double value : values) scale = std::max(scale, std::abs(value));
  if (scale == 0 || !std::isfinite(scale)) return scale;
  double scaled = 0;
  for (double value : values) {
    const double ratio = value / scale;
    scaled += ratio * ratio;
  }
  return scale * std::sqrt(scaled);
}
inline double sigmoid(double x) {
  const double t = std::exp(-std::abs(x));
  return x >= 0 ? 1 / (1 + t) : t / (1 + t);
}
inline double softplus(double x) { return std::max(x, 0.) + std::log1p(std::exp(-std::abs(x))); }
}  // namespace logitlab::detail
