#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "logitlab/generalized_linear_model.hpp"

using namespace logitlab;
void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template <class F>
void rejects(F action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "invalid GLM input accepted");
}
int main() {
  try {
    Dataset data;
    data.rows = 6;
    data.cols = 2;
    data.values = {-2, 1, -1, -1, 0, 2, 1, -2, 2, 0, .5, 1};
    std::vector<double> weights = {1, 0, 2, 3, 1, 2}, offset = {.1, -.3, 0, .2, .1, -.2};
    for (const auto& family : {"gaussian", "binomial", "poisson"}) {
      GLMOptions options;
      options.family = family;
      options.l2 = .2;
      options.tolerance = 1e-9;
      data.labels = std::string(family) == "binomial" ? std::vector<double>{0, 1, .2, .8, 1, .3}
                                                      : std::vector<double>{0, 1, 2, 3, 4, 1};
      std::vector<double> theta = {.3, -.2, .7}, gradient(3), scratch(3);
      glm_objective(data, theta, options, absl::MakeSpan(gradient), weights, offset);
      for (std::size_t j = 0; j < theta.size(); ++j) {
        const double old = theta[j];
        theta[j] = old + 1e-6;
        const double hi =
            glm_objective(data, theta, options, absl::MakeSpan(scratch), weights, offset);
        theta[j] = old - 1e-6;
        const double lo =
            glm_objective(data, theta, options, absl::MakeSpan(scratch), weights, offset);
        theta[j] = old;
        check(std::abs(gradient[j] - (hi - lo) / 2e-6) < 1e-8, "GLM gradient mismatch");
      }
      const auto model = fit_glm(data, options, weights, offset);
      check(model.converged, "GLM failed to converge");
      check(model.history.back().gradient_norm <= options.tolerance, "invalid GLM stop");
      check(model.work.hessian_vector_products > 0, "missing GLM curvature work");
      check(predict_glm(data, model, offset).size() == data.rows, "prediction shape");
      check(glm_deviance(data, model, weights, offset) >= 0, "negative GLM deviance");
      options.link = "unsupported";
      rejects([&] { fit_glm(data, options); });
    }
    GLMOptions options;
    rejects([&] { fit_glm(data, options, std::vector<double>(6, 0)); });
    rejects([&] { fit_glm(data, options, {}, {offset.data(), 2}); });
    options.family = "poisson";
    data.labels[0] = -1;
    rejects([&] { fit_glm(data, options); });
    options.family = "binomial";
    data.labels[0] = 2;
    rejects([&] { fit_glm(data, options); });
    std::cout << "Generalized linear model checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
