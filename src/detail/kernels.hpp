#pragma once

#include <vector>

#include "detail/numeric.hpp"
#include "detail/simd.hpp"
#include "logitlab/dataset.hpp"

namespace logitlab::detail {
struct DenseRow {
  const double* values;
  std::size_t size;
  double dot(absl::Span<const double> parameters) const {
#if LOGITLAB_HAS_AVX2

    if (simd::enabled()) return simd::dense_dot(values, parameters.data(), size);
#endif
    return detail::dot({values, size}, parameters.subspan(0, size));
  }
  void add(double scale, absl::Span<double> output) const {
#if LOGITLAB_HAS_AVX2

    if (simd::enabled()) {
      simd::dense_add(values, scale, output.data(), size);
      return;
    }
#endif
    for (std::size_t j = 0; j < size; ++j) output[j] += scale * values[j];
  }
  void add_diagonal(double scale, absl::Span<double> output) const {
    for (std::size_t j = 0; j < size; ++j) output[j] += scale * values[j] * values[j];
  }
  void add_moments(double residual, double curvature, absl::Span<double> gradient,
                   absl::Span<double> diagonal) const {
#if LOGITLAB_HAS_AVX2

    if (simd::enabled()) {
      simd::dense_moments(values, residual, curvature, gradient.data(), diagonal.data(), size);
      return;
    }
#endif
    for (std::size_t j = 0; j < size; ++j) {
      gradient[j] += residual * values[j];
      diagonal[j] += curvature * values[j] * values[j];
    }
  }
};
struct SparseRow {
  const double* values;
  const std::size_t* indices;
  std::size_t size;
  double dot(absl::Span<const double> parameters) const {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    std::size_t j = 0;
    for (; j + 3 < size; j += 4) {
      s0 += values[j] * parameters[indices[j]];
      s1 += values[j + 1] * parameters[indices[j + 1]];
      s2 += values[j + 2] * parameters[indices[j + 2]];
      s3 += values[j + 3] * parameters[indices[j + 3]];
    }
    double result = (s0 + s1) + (s2 + s3);
    for (; j < size; ++j) result += values[j] * parameters[indices[j]];
    return result;
  }
  void add(double scale, absl::Span<double> output) const {
    // Canonical CSR has unique indices: each packet can load before storing.
    double* out = output.data();
    std::size_t j = 0;
    for (; j + 3 < size; j += 4) {
      const auto a = indices[j], b = indices[j + 1], c = indices[j + 2], d = indices[j + 3];
      const double ga = out[a], gb = out[b], gc = out[c], gd = out[d];
      out[a] = ga + scale * values[j];
      out[b] = gb + scale * values[j + 1];
      out[c] = gc + scale * values[j + 2];
      out[d] = gd + scale * values[j + 3];
    }
    for (; j < size; ++j) out[indices[j]] += scale * values[j];
  }
  void add_diagonal(double scale, absl::Span<double> output) const {
    for (std::size_t j = 0; j < size; ++j) output[indices[j]] += scale * values[j] * values[j];
  }
  void add_moments(double residual, double curvature, absl::Span<double> gradient,
                   absl::Span<double> diagonal) const {
    // Canonical indices are distinct. Loading both output packets first keeps
    // the gradient and diagonal chains independent on irregular sparse rows.
    std::size_t j = 0;
    for (; j + 3 < size; j += 4) {
      const auto a = indices[j], b = indices[j + 1], c = indices[j + 2], d = indices[j + 3];
      const double xa = values[j], xb = values[j + 1], xc = values[j + 2], xd = values[j + 3];
      const double ga = gradient[a], gb = gradient[b], gc = gradient[c], gd = gradient[d];
      const double da = diagonal[a], db = diagonal[b], dc = diagonal[c], dd = diagonal[d];
      gradient[a] = ga + residual * xa;
      gradient[b] = gb + residual * xb;
      gradient[c] = gc + residual * xc;
      gradient[d] = gd + residual * xd;
      diagonal[a] = da + curvature * xa * xa;
      diagonal[b] = db + curvature * xb * xb;
      diagonal[c] = dc + curvature * xc * xc;
      diagonal[d] = dd + curvature * xd * xd;
    }
    for (; j < size; ++j) {
      gradient[indices[j]] += residual * values[j];
      diagonal[indices[j]] += curvature * values[j] * values[j];
    }
  }
};
struct DenseRows {
  const Dataset& data;
  DenseRow operator[](std::size_t i) const {
    return {data.values.data() + i * data.cols, data.cols};
  }
};
struct SparseRows {
  const Dataset& data;
  SparseRow operator[](std::size_t i) const {
    const auto begin = data.offsets[i], count = data.offsets[i + 1] - begin;
    // Empty CSR storage can have null data pointers; never add even zero to them.
    return {count ? data.values.data() + begin : nullptr,
            count ? data.indices.data() + begin : nullptr, count};
  }
};
template <class F>
decltype(auto) with_rows(const Dataset& data, F&& operation) {
  // Dispatch once per pass; row kernels inside the loop have no storage branch.
  if (data.sparse()) return operation(SparseRows{data});
  return operation(DenseRows{data});
}
template <class Curvature>
inline void hessian_vector(const Dataset& data, const Curvature& curvature,
                           absl::Span<const double> vector, double l2, bool intercept,
                           absl::Span<double> output, double damping = 0.) {
  std::fill(output.begin(), output.end(), 0.);
  with_rows(data, [&](const auto& rows) {
    for (std::size_t i = 0; i < data.rows; ++i) {
      if (curvature[i] == 0) continue;
      const auto row = rows[i];
      const double value = curvature[i] * (row.dot(vector) + (intercept ? vector.back() : 0.));
      row.add(value, output);
      if (intercept) output.back() += value;
    }
  });
  for (std::size_t j = 0; j < data.cols; ++j) output[j] += (l2 + damping) * vector[j];
  if (intercept) output.back() += damping * vector.back();
}
}  // namespace logitlab::detail
