#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/types/span.h"

namespace logitlab {
// Owned float64 row-major dense or canonical CSR. Labels must be 0 or 1.
// CSR indices must be sorted and unique. Entry points validate before computation.
struct Dataset {
  std::size_t rows = 0, cols = 0;
  std::vector<double> values;
  std::vector<std::size_t> offsets, indices;
  std::vector<double> labels;
  bool sparse() const { return !offsets.empty(); }
  void validate(bool require_labels = true) const;
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
struct Iteration {
  std::size_t epoch;
  double loss, gradient_norm, seconds;
};
struct Result {
  std::vector<double> weights;
  double intercept = 0;
  bool converged = false;
  std::string status;
  std::vector<Iteration> history;
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
