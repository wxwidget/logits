#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "logitlab/logistic_regression.hpp"
using namespace logitlab;
void check(bool x, const char* m) {
  if (!x) throw std::runtime_error(m);
}
template <class F>
void rejects(F action, const char* message) {
  bool rejected = false;
  try {
    action();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, message);
}
int main() {
  try {
    Dataset d;
    d.rows = 6;
    d.cols = 2;
    d.values = {-3, 1, -2, 1, -1, 1, 1, 1, 2, 1, 3, 1};
    d.labels = {0, 0, 0, 1, 1, 1};
    d.validate();
    Options o;
    o.l2 = 0.1;
    std::vector<double> w = {0.3, -0.2, 0.1}, g(3), scratch(3);
    objective(d, w, o, absl::MakeSpan(g));
    for (std::size_t j = 0; j < 3; ++j) {
      double old = w[j];
      w[j] = old + 1e-6;
      double hi = objective(d, w, o, absl::MakeSpan(scratch));
      w[j] = old - 1e-6;
      double lo = objective(d, w, o, absl::MakeSpan(scratch));
      w[j] = old;
      check(std::abs(g[j] - (hi - lo) / 2e-6) < 1e-7, "gradient mismatch");
    }
    Dataset s = d;
    s.values.clear();
    s.offsets = {0};
    for (std::size_t i = 0; i < d.rows; ++i) {
      for (std::size_t j = 0; j < d.cols; ++j) {
        s.values.push_back(d.values[i * d.cols + j]);
        s.indices.push_back(j);
      }
      s.offsets.push_back(s.values.size());
    }
    s.validate();
    check(std::abs(objective(d, w, o, absl::MakeSpan(g)) -
                   objective(s, w, o, absl::MakeSpan(scratch))) < 1e-12,
          "CSR objective mismatch");
    for (const auto& solver : {"gd", "sgd", "adam", "lbfgs", "newton-cg"}) {
      o.solver = solver;
      o.max_epochs = 200;
      o.learning_rate = 0.1;
      o.batch_size = 3;
      auto model = fit(d, o);
      check(model.history.front().work.full_evaluations == 1, "initial work not counted");
      check(model.history.front().work.row_visits == d.rows, "initial traversal not counted");
      check(model.work.row_visits >= d.rows * model.work.full_evaluations,
            "work count omits objective traversals");
      check(model.history.back().loss < model.history.front().loss, "loss failed to improve");
      auto p = predict(d, model);
      for (std::size_t i = 0; i < p.size(); ++i)
        check((p[i] >= 0.5) == (d.labels[i] == 1), "classification mismatch");
      auto sparse_model = fit(s, o);
      for (std::size_t j = 0; j < d.cols; ++j)
        check(std::abs(model.weights[j] - sparse_model.weights[j]) < 1e-10, "CSR solver mismatch");
      std::cout << solver << " loss=" << model.history.back().loss << " status=" << model.status
                << '\n';
    }
    check(sigmoid(1000) == 1 && sigmoid(-1000) == 0, "unstable sigmoid");
    Options failed_options;
    failed_options.solver = "gd";
    failed_options.learning_rate = 1e300;
    failed_options.max_epochs = 1;
    auto failed_model = fit(d, failed_options);
    check(failed_model.status == "line_search_failed", "expected failed search");
    check(failed_model.work.full_evaluations == 41, "rejected search evaluations not counted");
    check(failed_model.work.parameter_updates == 0, "rejected steps counted as updates");
    d.labels[0] = 2;
    bool rejected = false;
    try {
      fit(d, o);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check(rejected, "invalid labels accepted");
    d.labels[0] = 0;
    auto malformed = d;
    malformed.labels.clear();
    rejects([&] { objective(malformed, w, o, absl::MakeSpan(g)); }, "unvalidated objective labels");
    malformed = d;
    malformed.indices = {0};
    rejects([&] { malformed.validate(); }, "orphan CSR indices accepted");
    malformed = d;
    malformed.rows = std::numeric_limits<std::size_t>::max();
    rejects([&] { malformed.validate(); }, "row overflow accepted");
    auto short_gradient = std::vector<double>(2);
    rejects([&] { objective(d, w, o, absl::MakeSpan(short_gradient)); }, "bad gradient shape");
    w[0] = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { objective(d, w, o, absl::MakeSpan(g)); }, "NaN parameter accepted");
    w[0] = 0.3;
    o.l2 = -1;
    rejects([&] { objective(d, w, o, absl::MakeSpan(g)); }, "negative evaluator L2 accepted");
    o.l2 = 0.1;
    o.fit_intercept = false;
    double no_intercept = objective(d, w, o, absl::MakeSpan(g));
    check(g.back() == 0, "disabled intercept has gradient");
    w.back() = 900;
    check(objective(d, w, o, absl::MakeSpan(g)) == no_intercept, "disabled intercept changes loss");
    auto model = fit(d, o);
    auto inference = d;
    inference.labels.clear();
    check(predict(inference, model).size() == d.rows, "inference wrongly requires labels");
    model.weights[0] = std::numeric_limits<double>::infinity();
    rejects([&] { predict(inference, model); }, "non-finite model accepted");
    Dataset extreme;
    extreme.rows = 1;
    extreme.cols = 2;
    extreme.values = {2., -2.};
    Result overflow_model;
    overflow_model.weights = {std::numeric_limits<double>::max(),
                              std::numeric_limits<double>::max()};
    bool undefined_rejected = false;
    try {
      predict(extreme, overflow_model);
    } catch (const std::runtime_error&) {
      undefined_rejected = true;
    }
    check(undefined_rejected, "undefined prediction score accepted");
    std::cout << "Core checks passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
