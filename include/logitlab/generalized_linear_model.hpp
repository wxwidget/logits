#pragma once

#include "logitlab/logistic_regression.hpp"

namespace logitlab {
struct GLMOptions {
  // Canonical pairs only: gaussian/identity, binomial/logit, poisson/log.
  std::string family = "gaussian", link = "auto";
  std::size_t max_iter = 100, max_cg_iter = 200;
  double l2 = 0, tolerance = 1e-7;
  bool fit_intercept = true;
};
struct GLMIteration {
  std::size_t iteration;
  double loss, gradient_norm, seconds;
  Work work;
};
struct GLMResult {
  std::vector<double> weights;
  double intercept = 0;
  bool converged = false;
  std::string family, link, status;
  std::vector<GLMIteration> history;
  Work work;
};
// Weighted mean canonical negative log likelihood, omitting response-only
// constants, plus l2/2 ||w||^2. Intercept is unpenalized. Binomial responses
// may be proportions in [0,1]; Poisson responses must be nonnegative.
// Optional row weights are normalized; offset enters eta = Xw+b+offset.
// Zero-weight rows are not evaluated. Gradient must not alias parameters.
double glm_objective(const Dataset&, absl::Span<const double>, const GLMOptions&,
                     absl::Span<double> gradient, absl::Span<const double> sample_weight = {},
                     absl::Span<const double> offset = {});
GLMResult fit_glm(const Dataset&, const GLMOptions& = {},
                  absl::Span<const double> sample_weight = {},
                  absl::Span<const double> offset = {});
// Default prediction is the conditional mean; linear=true returns eta.
// Prediction offsets must be supplied anew; training offsets are not retained.
std::vector<double> predict_glm(const Dataset&, const GLMResult&,
                                absl::Span<const double> offset = {}, bool linear = false);
// Weighted MEAN unit deviance, evaluated stably from eta (not rounded means).
double glm_deviance(const Dataset&, const GLMResult&, absl::Span<const double> sample_weight = {},
                    absl::Span<const double> offset = {});
}  // namespace logitlab
