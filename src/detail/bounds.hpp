#pragma once
#include "detail/numeric.hpp"
#include "logitlab/linear_regression.hpp"

namespace logitlab::detail {
// A separable coefficient box, independent of objective and optimizer state.
struct CoefficientBox {
  std::vector<double> lower, upper;
  bool constrained = false;
  CoefficientBox(std::size_t features, const LinearOptions& options)
      : lower(features + 1, -INFINITY), upper(features + 1, INFINITY) {
    require(options.lower_bounds.empty() || options.lower_bounds.size() == features,
            "invalid lower bounds shape");
    require(options.upper_bounds.empty() || options.upper_bounds.size() == features,
            "invalid upper bounds shape");
    initialize_features(features, options);
    lower.back() = options.intercept_lower;
    upper.back() = options.intercept_upper;
    validate_intervals(features, options.fit_intercept);
    require(options.fit_intercept || (lower.back() <= 0 && upper.back() >= 0),
            "disabled intercept must allow zero");
    if (!options.fit_intercept) lower.back() = upper.back() = 0;
  }
  double clamp(double value, std::size_t feature) const {
    return std::max(lower[feature], std::min(upper[feature], value));
  }

 private:
  void initialize_features(std::size_t features, const LinearOptions& options) {
    for (std::size_t j = 0; j < features; ++j) {
      if (!options.lower_bounds.empty()) lower[j] = options.lower_bounds[j];
      if (!options.upper_bounds.empty()) upper[j] = options.upper_bounds[j];
      if (options.positive) lower[j] = std::max(lower[j], 0.);
    }
  }
  void validate_intervals(std::size_t features, bool intercept) {
    for (std::size_t j = 0; j < lower.size(); ++j) {
      require(!std::isnan(lower[j]) && !std::isnan(upper[j]) && lower[j] <= upper[j] &&
                  lower[j] != INFINITY && upper[j] != -INFINITY,
              "invalid coefficient bounds");
      if (j < features || intercept)
        constrained |= std::isfinite(lower[j]) || std::isfinite(upper[j]);
    }
  }
};
}  // namespace logitlab::detail
