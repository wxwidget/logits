#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "logitlab/dataset.hpp"

namespace logitlab {
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
