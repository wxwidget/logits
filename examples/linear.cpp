#include <cmath>
#include <iostream>

#include "logitlab/linear_regression.hpp"

int main() {
  logitlab::Dataset data;
  data.rows = 6;
  data.cols = 2;
  data.values = {-2, 1, -1, -1, 0, 2, 1, -2, 2, 0, 3, 1};
  data.labels = {-1.5, 1.5, 2, 6, 7, 8.5};  // 2*x0 - .5*x1 + 3
  auto model = logitlab::fit_linear(data);
  auto predictions = logitlab::predict_linear(data, model);
  if (!model.converged) return 1;
  for (std::size_t i = 0; i < data.rows; ++i)
    if (std::abs(predictions[i] - data.labels[i]) > 1e-5) return 1;
  logitlab::LinearOptions constrained;
  constrained.lower_bounds = {0, 0};
  constrained.upper_bounds = {1, 0};  // Bound x0; fix x1 exactly.
  auto bounded = logitlab::fit_linear(data, constrained);
  if (!bounded.converged || bounded.weights[0] < 0 || bounded.weights[0] > 1 ||
      bounded.weights[1] != 0)
    return 1;
  std::cout << "OLS: " << model.status << "; constrained: " << bounded.status << '\n';
}
