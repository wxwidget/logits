#pragma once
// Private adapter for immutable Dataset objects owned by the Python binding.
#include "logitlab/logistic_regression.hpp"
namespace logitlab::detail {
Result fit_logistic_validated(const Dataset&, const Options&);
std::vector<double> predict_logistic_validated(const Dataset&, const Result&);
}  // namespace logitlab::detail
