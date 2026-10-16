#pragma once
#include <limits>

#include "logitlab/logistic_regression.hpp"

namespace logitlab {
struct LinearOptions {
  // auto chooses diagonally preconditioned CG for smooth unconstrained fits,
  // otherwise monotone restarted FISTA. CG does not support L1 or bounds.
  std::string solver = "auto";
  std::size_t max_iter = 1000;
  double l1 = 0, l2 = 0, tolerance = 1e-7, learning_rate = 1;
  bool fit_intercept = true, positive = false;
  // Empty vectors mean unbounded; otherwise exactly cols elements.
  // Equal finite endpoints fix a coefficient.
  std::vector<double> lower_bounds, upper_bounds;
  double intercept_lower = -std::numeric_limits<double>::infinity();
  double intercept_upper = std::numeric_limits<double>::infinity();
};
struct LinearIteration {
  std::size_t iteration;
  double loss, stationarity_norm, seconds;
  Work work;
};
struct LinearResult {
  std::vector<double> weights;
  double intercept = 0;
  bool converged = false;
  std::string solver, status;
  std::vector<LinearIteration> history;
  Work work;
};
// Weighted squared loss / (2 sum(weights)) + l1 ||w||_1 + l2/2 ||w||^2.
// Empty weights mean equal weights. Intercept is never penalized.
// Returns full objective; output gradient contains ONLY the smooth part.
// Parameters must be finite and feasible; output must not alias parameters.
double linear_objective(const Dataset&, absl::Span<const double>, const LinearOptions&,
                        absl::Span<double> gradient, absl::Span<const double> sample_weight = {});
LinearResult fit_linear(const Dataset&, const LinearOptions& = {},
                        absl::Span<const double> sample_weight = {});
std::vector<double> predict_linear(const Dataset&, const LinearResult&);
}  // namespace logitlab
