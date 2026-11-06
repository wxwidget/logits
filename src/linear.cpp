#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "detail/bounds.hpp"
#include "detail/kernels.hpp"
#include "detail/linear_api.hpp"
#include "detail/validation.hpp"
#include "detail/weights.hpp"
#include "logitlab/linear_regression.hpp"

namespace logitlab {
namespace {
using detail::dot;
using detail::norm;
using detail::require;
void validate_options(const LinearOptions& o) {
  require(o.max_iter > 0 && std::isfinite(o.tolerance) && o.tolerance > 0 &&
              std::isfinite(o.learning_rate) && o.learning_rate > 0 && std::isfinite(o.l1) &&
              o.l1 >= 0 && std::isfinite(o.l2) && o.l2 >= 0,
          "invalid linear optimization options");
  require(
      o.solver == "auto" || o.solver == "cg" || o.solver == "proximal-gd" || o.solver == "fista",
      "unknown linear solver");
}
struct Problem {
  const Dataset& data;
  const LinearOptions& options;
  detail::ObservationWeights probabilities;
  detail::CoefficientBox bounds;
  bool constrained;
  Problem(const Dataset& d, const LinearOptions& o, absl::Span<const double> weights,
          bool known_storage = false)
      : data(detail::checked_data(d, false, known_storage)),
        options(o),
        probabilities(d.rows, weights),
        bounds(d.cols, o),
        constrained(bounds.constrained) {
    validate_options(o);
    require(o.solver != "cg" || (!constrained && o.l1 == 0), "CG requires no L1 or bounds");
  }
  double weight(std::size_t i) const { return probabilities[i]; }
  double clamp(double value, std::size_t j) const { return bounds.clamp(value, j); }
  double prox(double value, double step, std::size_t j) const {
    const double threshold = j < data.cols ? step * options.l1 : 0;
    return clamp(std::copysign(std::max(0., std::abs(value) - threshold), value), j);
  }
  double penalty(absl::Span<const double> w) const {
    double result = 0;
    for (std::size_t j = 0; j < data.cols; ++j) result += options.l1 * std::abs(w[j]);
    return result;
  }
  double evaluate(absl::Span<const double> w, absl::Span<double> g) const {
    std::fill(g.begin(), g.end(), 0);
    double loss = 0;
    const bool zero_features = detail::zero_coefficients(w, data.cols);
    detail::with_rows(data, [&](const auto& rows) {
      for (std::size_t i = 0; i < data.rows; ++i) {
        const double q = weight(i);
        if (q == 0) continue;
        const auto row = rows[i];
        const double residual = (zero_features ? 0. : row.dot(w)) + w.back() - data.labels[i];
        loss += .5 * q * residual * residual;
        row.add(q * residual, g);
        if (options.fit_intercept) g.back() += q * residual;
      }
    });
    for (std::size_t j = 0; j < data.cols; ++j) {
      loss += .5 * options.l2 * w[j] * w[j];
      g[j] += options.l2 * w[j];
    }
    return loss + penalty(w);
  }
  double stationarity(absl::Span<const double> w, absl::Span<const double> g) const {
    if (!constrained && options.l1 == 0) return norm(g);
    double square = 0;
    for (std::size_t j = 0; j < w.size(); ++j) {
      double residual = w[j] - prox(w[j] - g[j], 1., j);
      square += residual * residual;
    }
    return std::sqrt(square);
  }
};
}  // namespace

double linear_objective(const Dataset& d, absl::Span<const double> w, const LinearOptions& o,
                        absl::Span<double> g, absl::Span<const double> weights) {
  Problem problem(d, o, weights);
  require(w.size() == d.cols + 1 && g.size() == w.size(), "invalid parameter shape");
  for (std::size_t j = 0; j < w.size(); ++j)
    require(
        std::isfinite(w[j]) && w[j] >= problem.bounds.lower[j] && w[j] <= problem.bounds.upper[j],
        "parameters must be finite and feasible");
  return problem.evaluate(w, g);
}

namespace {
class LinearTrainer {
 public:
  LinearTrainer(const Dataset& data, const LinearOptions& options, absl::Span<const double> weights,
                bool known_storage = false)
      : problem(data, options, weights, known_storage),
        d(data),
        o(options),
        n(d.cols + 1),
        w(n),
        g(n),
        next(n),
        next_g(n),
        anchor(n),
        anchor_g(n),
        diagonal(n, o.l2),
        direction(n),
        hp(n),
        z(n),
        step(o.learning_rate) {
    result.solver =
        o.solver == "auto" ? (problem.constrained || o.l1 > 0 ? "fista" : "cg") : o.solver;
    result.status = "max_iter";
    initialize_parameters();
    start = std::chrono::steady_clock::now();
    loss = evaluate(w, absl::MakeSpan(g));
    if (result.solver == "cg") initialize_cg();
    record(0);
    anchor = w;
    anchor_g = g;
  }
  LinearResult run() {
    for (std::size_t iteration = 1; iteration <= o.max_iter && !result.converged; ++iteration) {
      const bool accepted = result.solver == "cg" ? cg_step(iteration) : proximal_step(iteration);
      if (!accepted) break;
      ++result.work.parameter_updates;
      record(iteration);
    }
    if (result.converged) result.status = "converged";
    result.intercept = w.back();
    w.pop_back();
    result.weights = std::move(w);
    return std::move(result);
  }

 private:
  void initialize_parameters() {
    for (std::size_t j = 0; j < n; ++j) w[j] = problem.clamp(0., j);
    if (!o.fit_intercept) return;
    double mean = 0, first = 0;
    bool seen = false, constant = true;
    for (std::size_t i = 0; i < d.rows; ++i) {
      const double q = problem.weight(i);
      if (q == 0) continue;
      mean += q * d.labels[i];
      if (!seen) {
        first = d.labels[i];
        seen = true;
      } else
        constant &= d.labels[i] == first;
    }
    // Preserve exact constants instead of introducing a rounded weighted sum.
    w.back() = problem.clamp(constant ? first : mean, d.cols);
  }
  double evaluate(absl::Span<const double> parameters, absl::Span<double> gradient) {
    ++result.work.full_evaluations;
    result.work.row_visits += d.rows;
    return problem.evaluate(parameters, gradient);
  }
  void record(std::size_t iteration) {
    const double stationarity = problem.stationarity(w, g);
    if (!std::isfinite(loss) || !std::isfinite(norm(g)) || !std::isfinite(stationarity))
      throw std::runtime_error("non-finite linear objective; scale features and targets");
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.history.push_back({iteration, loss, stationarity, seconds, result.work});
    result.converged = stationarity <= o.tolerance;
  }
  void initialize_cg() {
    diagonal.back() = o.fit_intercept ? 1. : 0.;
    detail::with_rows(d, [&](const auto& rows) {
      for (std::size_t i = 0; i < d.rows; ++i) {
        if (problem.weight(i) != 0)
          rows[i].add_diagonal(problem.weight(i), absl::MakeSpan(diagonal));
      }
    });
    result.work.row_visits += d.rows;
    for (std::size_t j = 0; j < n; ++j) {
      if (!(diagonal[j] > 0) || !std::isfinite(diagonal[j])) diagonal[j] = 1.;
      direction[j] = z[j] = -g[j] / diagonal[j];
    }
  }
  bool cg_step(std::size_t iteration) {
    const double rz = -dot(g, z);
    detail::hessian_vector(d, problem.probabilities, direction, o.l2, o.fit_intercept,
                           absl::MakeSpan(hp));
    ++result.work.hessian_vector_products;
    result.work.row_visits += d.rows;
    const double curvature = dot(direction, hp);
    if (!(curvature > 0) || !std::isfinite(curvature) || !std::isfinite(rz)) {
      result.status = "numerical_breakdown";
      return false;
    }
    const double alpha = rz / curvature;
    for (std::size_t j = 0; j < n; ++j) {
      next[j] = w[j] + alpha * direction[j];
      next_g[j] = g[j] + alpha * hp[j];
    }
    // Quadratic recurrence avoids a redundant data pass. Refresh periodically,
    // on budget exhaustion, and BEFORE certifying convergence to bound drift.
    double new_loss = loss + alpha * dot(g, direction) + .5 * alpha * alpha * curvature;
    if (iteration % 20 == 0 || iteration == o.max_iter || norm(next_g) <= o.tolerance)
      new_loss = evaluate(next, absl::MakeSpan(next_g));
    for (std::size_t j = 0; j < n; ++j) z[j] = -next_g[j] / diagonal[j];
    const double beta = -dot(next_g, z) / rz;
    for (std::size_t j = 0; j < n; ++j) direction[j] = z[j] + beta * direction[j];
    w.swap(next);
    g.swap(next_g);
    loss = new_loss;
    return true;
  }
  bool proximal_backtrack(double& candidate_loss) {
    for (int attempt = 0; attempt < 64; ++attempt) {
      double squared = 0;
      for (std::size_t j = 0; j < n; ++j) {
        next[j] = problem.prox(anchor[j] - step * anchor_g[j], step, j);
        const double delta = next[j] - anchor[j];
        squared += delta * delta;
      }
      candidate_loss = evaluate(next, absl::MakeSpan(next_g));
      double curvature = 0;
      for (std::size_t j = 0; j < n; ++j)
        curvature += (next[j] - anchor[j]) * (next_g[j] - anchor_g[j]);
      if (std::isfinite(candidate_loss) && std::isfinite(curvature) &&
          curvature <= squared / step * (1 + 1e-12))
        return true;
      step *= .5;
    }
    return false;
  }
  void accept_proximal(double candidate_loss) {
    const double next_momentum = (1 + std::sqrt(1 + 4 * momentum * momentum)) / 2;
    const double ratio = (momentum - 1) / next_momentum;
    for (std::size_t j = 0; j < n; ++j) {
      anchor[j] = next[j] + ratio * (next[j] - w[j]);
      // The smooth gradient is affine, so the same extrapolation is exact.
      anchor_g[j] = next_g[j] + ratio * (next_g[j] - g[j]);
    }
    momentum = next_momentum;
    w.swap(next);
    g.swap(next_g);
    loss = candidate_loss;
    step = std::min(o.learning_rate, step * 1.25);
  }
  bool proximal_step(std::size_t iteration) {
    for (int restart = 0; restart < 2; ++restart) {
      if (result.solver == "proximal-gd" || restart) {
        anchor = w;
        anchor_g = g;
        momentum = 1;
      } else if (iteration % 20 == 0)
        evaluate(anchor, absl::MakeSpan(anchor_g));
      double candidate_loss = 0;
      if (!proximal_backtrack(candidate_loss)) continue;
      if (candidate_loss > loss + 1e-14 * (1 + std::abs(loss))) continue;
      accept_proximal(candidate_loss);
      return true;
    }
    result.status = "line_search_failed";
    return false;
  }
  Problem problem;
  const Dataset& d;
  const LinearOptions& o;
  std::size_t n;
  LinearResult result;
  std::vector<double> w, g, next, next_g, anchor, anchor_g, diagonal, direction, hp, z;
  std::chrono::steady_clock::time_point start;
  double loss = 0, momentum = 1, step;
};
}  // namespace
LinearResult fit_linear(const Dataset& data, const LinearOptions& options,
                        absl::Span<const double> weights) {
  return LinearTrainer(data, options, weights).run();
}

std::vector<double> predict_impl(const Dataset& d, const LinearResult& model, bool known_storage) {
  if (!known_storage) d.validate(false);
  require(model.weights.size() == d.cols, "feature count mismatch");
  require(std::isfinite(model.intercept), "non-finite intercept");
  for (double value : model.weights) require(std::isfinite(value), "non-finite weights");
  std::vector<double> output(d.rows);
  for (std::size_t i = 0; i < d.rows; ++i) {
    output[i] = d.dot(i, model.weights) + model.intercept;
    if (!std::isfinite(output[i])) throw std::runtime_error("non-finite linear prediction");
  }
  return output;
}
LinearResult detail::fit_linear_validated(const Dataset& data, const LinearOptions& options,
                                          absl::Span<const double> weights) {
  return LinearTrainer(data, options, weights, true).run();
}
std::vector<double> predict_linear(const Dataset& data, const LinearResult& model) {
  return predict_impl(data, model, false);
}
std::vector<double> detail::predict_linear_validated(const Dataset& data,
                                                     const LinearResult& model) {
  return predict_impl(data, model, true);
}
}  // namespace logitlab
