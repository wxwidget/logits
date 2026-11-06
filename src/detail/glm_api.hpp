#pragma once
// Private adapter for immutable Dataset objects owned by the Python binding.
#include "logitlab/generalized_linear_model.hpp"
namespace logitlab::detail {
GLMResult fit_glm_validated(const Dataset&, const GLMOptions&, absl::Span<const double>,
                            absl::Span<const double>);
std::vector<double> predict_glm_validated(const Dataset&, const GLMResult&,
                                          absl::Span<const double>, bool);
}  // namespace logitlab::detail
