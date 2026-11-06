#pragma once

#include <numeric>
#include <vector>

#include "detail/numeric.hpp"

namespace logitlab::detail {
// Relative observation mass shared by linear and GLM objectives. An absent
// weight vector stays allocation-free; zero-mass rows must skip arithmetic.
class ObservationWeights {
 public:
  ObservationWeights(std::size_t rows, absl::Span<const double> weights) : uniform_(1. / rows) {
    require(weights.empty() || weights.size() == rows, "invalid sample weight shape");
    if (weights.empty()) return;
    double maximum = 0;
    for (double value : weights) {
      require(std::isfinite(value) && value >= 0, "sample weights must be finite and nonnegative");
      maximum = std::max(maximum, value);
    }
    require(maximum > 0, "sample weights must have positive mass");
    values_.reserve(rows);
    for (double value : weights) values_.push_back(value / maximum);
    const double mass = std::accumulate(values_.begin(), values_.end(), 0.);
    for (double& value : values_) value /= mass;
  }
  double operator[](std::size_t row) const { return values_.empty() ? uniform_ : values_[row]; }

 private:
  double uniform_;
  std::vector<double> values_;
};
}  // namespace logitlab::detail
