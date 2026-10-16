#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "logitlab/linear_regression.hpp"

using namespace logitlab;
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class F>
void rejects(F action) {
  bool caught = false;
  try {
    action();
  } catch (const std::invalid_argument&) {
    caught = true;
  }
  check(caught, "invalid input accepted");
}
int main() {
  try {
    Dataset d;
    d.rows = 5;
    d.cols = 2;
    d.values = {-2, 1, -1, -1, 0, 2, 1, -2, 2, 0};
    d.labels = {-2.3, -.8, 2.4, 2.7, 5.2};
    LinearOptions o;
    o.l2 = .2;
    std::vector<double> w = {.3, -.2, .7}, g(3), scratch(3), weights = {1, 0, 2, 3, 1};
    linear_objective(d, w, o, absl::MakeSpan(g), weights);
    for (std::size_t j = 0; j < w.size(); ++j) {
      auto old = w[j];
      w[j] = old + 1e-6;
      auto hi = linear_objective(d, w, o, absl::MakeSpan(scratch), weights);
      w[j] = old - 1e-6;
      auto lo = linear_objective(d, w, o, absl::MakeSpan(scratch), weights);
      w[j] = old;
      check(std::abs(g[j] - (hi - lo) / 2e-6) < 1e-8, "linear gradient mismatch");
    }
    for (const auto& solver : {"cg", "proximal-gd", "fista"}) {
      o.solver = solver;
      o.tolerance = 1e-9;
      o.max_iter = 3000;
      auto model = fit_linear(d, o, weights);
      check(model.converged, "linear fit did not converge");
      check(model.history.back().stationarity_norm <= o.tolerance, "invalid stop");
      check(model.work.full_evaluations >= model.history.size(), "missing work");
      check(predict_linear(d, model).size() == d.rows, "prediction shape");
    }
    o.solver = "fista";
    o.l1 = .1;
    o.positive = true;
    auto constrained = fit_linear(d, o);
    for (auto v : constrained.weights) check(v >= 0, "negative constrained coefficient");
    o.lower_bounds = {1, 0};
    o.upper_bounds = {1, .2};
    auto fixed = fit_linear(d, o);
    check(fixed.weights[0] == 1 && fixed.weights[1] <= .2, "box/fixed constraint broken");
    o.solver = "cg";
    rejects([&] { fit_linear(d, o); });
    o = LinearOptions{};
    auto malformed = d;
    malformed.labels[0] = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { fit_linear(malformed, o); });
    rejects([&] { fit_linear(d, o, std::vector<double>(5, 0)); });
    rejects([&] { fit_linear(d, o, {weights.data(), 2}); });
    o.lower_bounds = {2};
    o.upper_bounds = {1};
    rejects([&] { fit_linear(d, o); });
    o = LinearOptions{};
    o.l1 = -1;
    rejects([&] { linear_objective(d, w, o, absl::MakeSpan(g)); });
    o = LinearOptions{};
    w[0] = std::numeric_limits<double>::infinity();
    rejects([&] { linear_objective(d, w, o, absl::MakeSpan(g)); });
    Dataset constant;
    constant.rows = 7;
    constant.cols = 1;
    constant.values = {0, 1, 2, 3, 4, 5, 6};
    constant.labels.assign(7, 2.);
    auto constant_fit = fit_linear(constant);
    check(constant_fit.converged && constant_fit.intercept == 2., "rounded constant intercept");
    std::cout << "Linear regression checks passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
