#pragma once
#include <cstddef>
#include <vector>

#include "absl/types/span.h"

namespace logitlab {
// Owned float64 row-major dense or canonical CSR. Target domain is validated
// by the entry point: binary for logistic, real for linear, family-specific for GLM.
// CSR indices must be sorted and unique. Entry points validate before computation.
struct Dataset {
  std::size_t rows = 0, cols = 0;
  std::vector<double> values;
  std::vector<std::size_t> offsets, indices;
  std::vector<double> labels;
  bool sparse() const { return !offsets.empty(); }
  void validate(bool require_labels = true, bool binary_labels = true) const;
  // Low-level row kernels require a validated dataset and correctly sized spans.
  double dot(std::size_t row, absl::Span<const double> weights) const;
  void add(std::size_t row, double scale, absl::Span<double> out) const;
};
// A row visit is one objective/gradient, batch, curvature or HVP traversal.
// It is a work proxy, not a FLOP count; parameter-vector work is separate.
struct Work {
  std::size_t full_evaluations = 0, batch_gradients = 0, hessian_vector_products = 0;
  std::size_t row_visits = 0, parameter_updates = 0;
};
}  // namespace logitlab
