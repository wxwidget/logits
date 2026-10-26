#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
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
struct Options {
  std::string solver = "lbfgs";
  std::size_t max_epochs = 100, batch_size = 256, history_size = 10;
  double learning_rate = 0.05, l2 = 1e-4, tolerance = 1e-6;
  std::uint64_t seed = 42;
  bool fit_intercept = true;
};
// A row visit is one objective/gradient, batch, curvature or HVP traversal.
// It is a work proxy, not a FLOP count; parameter-vector work is separate.
struct Work {
  std::size_t full_evaluations = 0, batch_gradients = 0, hessian_vector_products = 0;
  std::size_t row_visits = 0, parameter_updates = 0;
};
struct Iteration {
  std::size_t epoch;
  double loss, gradient_norm, seconds;
  Work work;
};
struct Result {
  std::vector<double> weights;
  double intercept = 0;
  bool converged = false;
  std::string status;
  std::vector<Iteration> history;
  Work work;
};
double sigmoid(double x);
// Mean binary cross-entropy + l2/2 * ||w||^2; intercept is not penalized.
// Parameters are [weights..., intercept] even when fit_intercept is false.
// The output gradient must not alias parameters. Invalid inputs throw invalid_argument.
double objective(const Dataset&, absl::Span<const double>, const Options&,
                 absl::Span<double> gradient);
Result fit(const Dataset&, const Options& = {});
std::vector<double> predict(const Dataset&, const Result&);
}  // namespace logitlab
