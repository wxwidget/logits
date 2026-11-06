#pragma once
// Private adapter for immutable Dataset objects owned by the Python binding.
#include "logitlab/linear_regression.hpp"
namespace logitlab::detail {
LinearResult fit_linear_validated(const Dataset&, const LinearOptions&, absl::Span<const double>);
std::vector<double> predict_linear_validated(const Dataset&, const LinearResult&);
}  // namespace logitlab::detail
