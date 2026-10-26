#include <iostream>

#include "logitlab/generalized_linear_model.hpp"

int main() {
  logitlab::Dataset data;
  data.rows = 8;
  data.cols = 1;
  data.values = {-1.5, -1., -.5, 0., .5, 1., 1.5, 2.};
  data.labels = {0, 1, 1, 2, 1, 3, 4, 5};
  logitlab::GLMOptions options;
  options.family = "poisson";
  options.link = "log";
  options.l2 = .01;
  options.tolerance = 1e-8;
  const auto model = logitlab::fit_glm(data, options);
  std::cout << model.family << '/' << model.link << " status=" << model.status
            << " mean deviance=" << logitlab::glm_deviance(data, model) << '\n';
  for (double mean : logitlab::predict_glm(data, model)) std::cout << mean << ' ';
  std::cout << '\n';
  return model.converged ? 0 : 1;
}
