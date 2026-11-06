#include <chrono>
#include <random>

#include "detail/families.hpp"
#include "detail/kernels.hpp"
#include "detail/lbfgs.hpp"
#include "detail/logistic_api.hpp"
#include "detail/validation.hpp"
#include "logitlab/logistic_regression.hpp"

namespace logitlab {
double sigmoid(double value) { return detail::sigmoid(value); }
namespace {
using detail::dot;
using detail::norm;
using detail::require;

void validate_options(const Options& options) {
  require(options.max_epochs > 0 && options.batch_size > 0 && options.history_size > 0,
          "iteration, batch and history sizes must be positive");
  require(std::isfinite(options.learning_rate) && options.learning_rate > 0 &&
              std::isfinite(options.l2) && options.l2 >= 0 && std::isfinite(options.tolerance) &&
              options.tolerance > 0,
          "invalid optimization options");
  require(options.solver == "gd" || options.solver == "sgd" || options.solver == "adam" ||
              options.solver == "lbfgs" || options.solver == "newton-cg",
          "unknown solver");
}

double evaluate(const Dataset& data, absl::Span<const double> parameters, const Options& options,
                absl::Span<double> gradient, absl::Span<double> curvature = {}) {
  std::fill(gradient.begin(), gradient.end(), 0.);
  double loss = 0;
  const double inverse_rows = 1. / data.rows;
  const bool zero_features = detail::zero_coefficients(parameters, data.cols);
  detail::with_rows(data, [&](const auto& rows) {
    for (std::size_t begin = 0; begin < data.rows; begin += detail::kResponseBatch) {
      const auto count = std::min(detail::kResponseBatch, data.rows - begin);
      detail::ResponseBatch batch;
      for (std::size_t lane = 0; lane < count; ++lane) {
        const auto i = begin + lane;
        batch.eta[lane] = (zero_features ? 0. : rows[i].dot(parameters)) +
                          (options.fit_intercept ? parameters.back() : 0.);
        batch.target[lane] = data.labels[i];
      }
      detail::Binomial{}.batch(batch, count);
      for (std::size_t lane = 0; lane < count; ++lane) {
        const auto i = begin + lane;
        const auto& moment = batch.values[lane];
        loss += moment.loss;
        if (!curvature.empty()) curvature[i] = moment.variance * inverse_rows;
        const double residual = moment.residual * inverse_rows;
        rows[i].add(residual, gradient);
        if (options.fit_intercept) gradient.back() += residual;
      }
    }
  });
  loss /= data.rows;
  for (std::size_t j = 0; j < data.cols; ++j) {
    loss += .5 * options.l2 * parameters[j] * parameters[j];
    gradient[j] += options.l2 * parameters[j];
  }
  return loss;
}

// One trainer owns the optimization state and work accounting. Each step has
// exactly one responsibility; all row storage details stay in shared kernels.
class LogisticTrainer {
 public:
  LogisticTrainer(const Dataset& data, const Options& options)
      : d(data),
        o(options),
        n(d.cols + 1),
        stochastic(o.solver == "sgd" || o.solver == "adam"),
        adaptive(o.solver == "adam"),
        w(n),
        g(n),
        direction(n),
        candidate(n),
        next_g(n),
        curvature(o.solver == "newton-cg" ? d.rows : 0),
        next_curvature(curvature.size()),
        m(adaptive ? n : 0),
        v(adaptive ? n : 0),
        order(stochastic ? d.rows : 0),
        rng(o.seed),
        history(o.history_size),
        start(std::chrono::steady_clock::now()) {
    std::iota(order.begin(), order.end(), 0);
    result.status = "max_epochs";
    loss = objective_at(w, absl::MakeSpan(g));
  }
  Result run() {
    record(0);
    for (std::size_t epoch = 1; epoch <= o.max_epochs && norm(g) > o.tolerance; ++epoch) {
      if (stochastic)
        stochastic_step();
      else if (!batch_step()) {
        result.status = "line_search_failed";
        break;
      }
      record(epoch);
    }
    if (norm(g) <= o.tolerance) {
      result.converged = true;
      result.status = "converged";
    }
    result.intercept = o.fit_intercept ? w.back() : 0.;
    w.pop_back();
    result.weights = std::move(w);
    return std::move(result);
  }

 private:
  double objective_at(absl::Span<const double> parameters, absl::Span<double> gradient,
                      bool trial = false) {
    ++result.work.full_evaluations;
    result.work.row_visits += d.rows;
    return evaluate(d, parameters, o, gradient, absl::MakeSpan(trial ? next_curvature : curvature));
  }
  void record(std::size_t epoch) {
    const double gradient_norm = norm(g);
    if (!std::isfinite(loss) || !std::isfinite(gradient_norm))
      throw std::runtime_error("non-finite objective; scale features or reduce learning rate");
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.history.push_back({epoch, loss, gradient_norm, seconds, result.work});
  }
  void update_parameters() {
    ++updates;
    const double correction1 = adaptive ? 1 - std::pow(.9, updates) : 1;
    const double correction2 = adaptive ? 1 - std::pow(.999, updates) : 1;
    for (std::size_t j = 0; j < n; ++j) {
      const double gradient = next_g[j] + (j < d.cols ? o.l2 * w[j] : 0.);
      if (adaptive) {
        m[j] = .9 * m[j] + .1 * gradient;
        v[j] = .999 * v[j] + .001 * gradient * gradient;
        w[j] -= o.learning_rate * (m[j] / correction1) / (std::sqrt(v[j] / correction2) + 1e-8);
      } else
        w[j] -= o.learning_rate * gradient;
    }
  }
  void stochastic_step() {
    std::shuffle(order.begin(), order.end(), rng);
    detail::with_rows(d, [&](const auto& rows) {
      for (std::size_t begin = 0; begin < d.rows;) {
        const auto count = std::min(o.batch_size, d.rows - begin);
        const double inverse_count = 1. / count;
        ++result.work.batch_gradients;
        ++result.work.parameter_updates;
        result.work.row_visits += count;
        std::fill(next_g.begin(), next_g.end(), 0.);
        for (std::size_t pack = begin; pack < begin + count; pack += detail::kResponseBatch) {
          const auto size = std::min(detail::kResponseBatch, begin + count - pack);
          detail::ResponseBatch batch;
          for (std::size_t lane = 0; lane < size; ++lane) {
            const auto i = order[pack + lane];
            batch.eta[lane] = rows[i].dot(w) + (o.fit_intercept ? w.back() : 0.);
            batch.target[lane] = d.labels[i];
          }
          detail::Binomial{}.batch(batch, size);
          for (std::size_t lane = 0; lane < size; ++lane) {
            const double residual = batch.values[lane].residual * inverse_count;
            rows[order[pack + lane]].add(residual, absl::MakeSpan(next_g));
            if (o.fit_intercept) next_g.back() += residual;
          }
        }
        update_parameters();
        begin += count;
      }
    });
    loss = objective_at(w, absl::MakeSpan(g));
  }
  void newton_direction() {
    std::vector<double> r = g, p = g, hp(n), solution(n, 0.);
    double rr = dot(r, r), initial = rr;
    // Tighten the inner residual as the full gradient falls; solving a distant
    // Newton model to fixed 1e-4 relative precision wastes data passes.
    const double forcing = std::min(.5, std::sqrt(norm(g)));
    // Inner accuracy below half the final tolerance cannot improve acceptance;
    // the outer loop still checks the actual complete objective gradient.
    const double floor = .5 * o.tolerance;
    const double target = std::max(floor * floor, initial * forcing * forcing);
    for (std::size_t k = 0; k < std::min<std::size_t>(n, 50); ++k) {
      ++result.work.hessian_vector_products;
      result.work.row_visits += d.rows;
      detail::hessian_vector(d, curvature, p, o.l2, o.fit_intercept, absl::MakeSpan(hp), 1e-10);
      const double denominator = dot(p, hp);
      if (denominator <= 0) break;
      const double alpha = rr / denominator;
      for (std::size_t j = 0; j < n; ++j) {
        solution[j] += alpha * p[j];
        r[j] -= alpha * hp[j];
      }
      const double next = dot(r, r);
      if (next <= target) break;
      for (std::size_t j = 0; j < n; ++j) p[j] = r[j] + next / rr * p[j];
      rr = next;
    }
    direction.swap(solution);
  }
  double descent_direction() {
    direction = g;
    if (o.solver == "lbfgs")
      history.apply(direction);
    else if (o.solver == "newton-cg")
      newton_direction();
    for (double& value : direction) value = -value;
    double slope = dot(g, direction);
    if (!std::isfinite(slope) || slope >= 0) {
      direction = g;
      for (double& value : direction) value = -value;
      slope = -dot(g, g);
    }
    return slope;
  }
  bool line_search(double slope, double& next_loss) {
    double step = o.solver == "gd" ? o.learning_rate : 1.;
    for (int attempt = 0; attempt < 40; ++attempt) {
      for (std::size_t j = 0; j < n; ++j) candidate[j] = w[j] + step * direction[j];
      next_loss = objective_at(candidate, absl::MakeSpan(next_g), true);
      if (std::isfinite(next_loss) && next_loss <= loss + 1e-4 * step * slope) return true;
      step *= .5;
    }
    return false;
  }
  bool batch_step() {
    const double slope = descent_direction();
    double next_loss = 0;
    if (!line_search(slope, next_loss)) return false;
    if (o.solver == "lbfgs") history.update(w, candidate, g, next_g);
    w.swap(candidate);
    g.swap(next_g);
    curvature.swap(next_curvature);
    loss = next_loss;
    ++result.work.parameter_updates;
    return true;
  }
  const Dataset& d;
  const Options& o;
  std::size_t n;
  bool stochastic, adaptive;
  std::vector<double> w, g, direction, candidate, next_g, curvature, next_curvature, m, v;
  std::vector<std::size_t> order;
  std::mt19937_64 rng;
  detail::LBFGSHistory history;
  std::chrono::steady_clock::time_point start;
  std::size_t updates = 0;
  Result result;
  double loss = 0;
};
}  // namespace

double objective(const Dataset& data, absl::Span<const double> parameters, const Options& options,
                 absl::Span<double> gradient) {
  data.validate();
  require(std::isfinite(options.l2) && options.l2 >= 0, "l2 must be finite and nonnegative");
  require(parameters.size() == data.cols + 1 && gradient.size() == parameters.size(),
          "invalid parameter shape");
  require(detail::finite(parameters), "parameters must be finite");
  return evaluate(data, parameters, options, gradient);
}
Result fit(const Dataset& data, const Options& options) {
  data.validate();
  validate_options(options);
  return LogisticTrainer(data, options).run();
}
std::vector<double> predict_impl(const Dataset& data, const Result& model, bool known_storage) {
  if (!known_storage) data.validate(false);
  require(model.weights.size() == data.cols, "feature count mismatch");
  require(std::isfinite(model.intercept), "non-finite intercept");
  require(detail::finite(model.weights), "non-finite weights");
  std::vector<double> output(data.rows);
  detail::with_rows(data, [&](const auto& rows) {
    for (std::size_t i = 0; i < data.rows; ++i) {
      const double score = rows[i].dot(model.weights) + model.intercept;
      if (std::isnan(score)) throw std::runtime_error("undefined prediction score; scale features");
      output[i] = detail::sigmoid(score);
    }
  });
  return output;
}
Result detail::fit_logistic_validated(const Dataset& data, const Options& options) {
  validate_targets(data, true);
  validate_options(options);
  return LogisticTrainer(data, options).run();
}
std::vector<double> predict(const Dataset& data, const Result& model) {
  return predict_impl(data, model, false);
}
std::vector<double> detail::predict_logistic_validated(const Dataset& data, const Result& model) {
  return predict_impl(data, model, true);
}
}  // namespace logitlab
