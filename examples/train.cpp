#include <iostream>

#include "logitlab/logistic_regression.hpp"

int main() {
  logitlab::Dataset data;
  data.rows = 4;
  data.cols = 2;
  data.values = {-2., 0., -1., 1., 1., 0., 2., 1.};
  data.labels = {0., 0., 1., 1.};
  logitlab::Options options;
  options.l2 = 0.1;
  const auto model = logitlab::fit(data, options);
  const auto probabilities = logitlab::predict(data, model);
  std::cout << "status=" << model.status << " loss=" << model.history.back().loss << '\n';
  for (auto probability : probabilities) std::cout << probability << '\n';
  return model.converged ? 0 : 1;
}
