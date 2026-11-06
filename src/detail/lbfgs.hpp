#pragma once
#include <vector>

#include "detail/numeric.hpp"

namespace logitlab::detail {
// Limited-memory secant history owns only curvature pairs; dataset traversal
// and line-search policy belong to the trainer, not this algebra component.
class LBFGSHistory {
 public:
  explicit LBFGSHistory(std::size_t capacity) : capacity_(capacity) { alpha_.reserve(capacity); }
  void apply(std::vector<double>& direction) {
    if (steps_.empty()) return;
    alpha_.resize(steps_.size());
    for (std::size_t k = steps_.size(); k-- > 0;) {
      alpha_[k] = dot(steps_[k], direction) / dot(steps_[k], changes_[k]);
      for (std::size_t j = 0; j < direction.size(); ++j) direction[j] -= alpha_[k] * changes_[k][j];
    }
    const double scale =
        dot(steps_.back(), changes_.back()) / dot(changes_.back(), changes_.back());
    for (double& value : direction) value *= scale;
    for (std::size_t k = 0; k < steps_.size(); ++k) {
      const double beta = dot(changes_[k], direction) / dot(steps_[k], changes_[k]);
      for (std::size_t j = 0; j < direction.size(); ++j)
        direction[j] += steps_[k][j] * (alpha_[k] - beta);
    }
  }
  void update(absl::Span<const double> before, absl::Span<const double> after,
              absl::Span<const double> gradient, absl::Span<const double> next_gradient) {
    std::vector<double> step(before.size()), change(before.size());
    for (std::size_t j = 0; j < step.size(); ++j) {
      step[j] = after[j] - before[j];
      change[j] = next_gradient[j] - gradient[j];
    }
    if (!(dot(step, change) > 1e-12 * norm(step) * norm(change))) return;
    if (steps_.size() == capacity_) {
      steps_.erase(steps_.begin());
      changes_.erase(changes_.begin());
    }
    steps_.push_back(std::move(step));
    changes_.push_back(std::move(change));
  }

 private:
  std::size_t capacity_;
  std::vector<std::vector<double>> steps_, changes_;
  std::vector<double> alpha_;
};
}  // namespace logitlab::detail
