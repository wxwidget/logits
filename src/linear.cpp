#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "logitlab/linear_regression.hpp"

namespace logitlab {
namespace {
void require(bool value, const char* message) {
  if (!value) throw std::invalid_argument(message);
}
double dot(absl::Span<const double> x, absl::Span<const double> y) {
  return std::inner_product(x.begin(), x.end(), y.begin(), 0.);
}
double norm(absl::Span<const double> x) { return std::sqrt(dot(x, x)); }
struct Problem {
  const Dataset& data;
  const LinearOptions& options;
  std::vector<double> probabilities, lower, upper;
  bool constrained = false;
  Problem(const Dataset& d, const LinearOptions& o, absl::Span<const double> weights)
      : data(d), options(o) {
    d.validate(true, false);
    lower.assign(d.cols + 1, -INFINITY);
    upper.assign(d.cols + 1, INFINITY);
    require(o.max_iter > 0 && std::isfinite(o.tolerance) && o.tolerance > 0 &&
                std::isfinite(o.learning_rate) && o.learning_rate > 0 && std::isfinite(o.l1) &&
                o.l1 >= 0 && std::isfinite(o.l2) && o.l2 >= 0,
            "invalid linear optimization options");
    require(
        o.solver == "auto" || o.solver == "cg" || o.solver == "proximal-gd" || o.solver == "fista",
        "unknown linear solver");
    require(o.lower_bounds.empty() || o.lower_bounds.size() == d.cols,
            "invalid lower bounds shape");
    require(o.upper_bounds.empty() || o.upper_bounds.size() == d.cols,
            "invalid upper bounds shape");
    for (std::size_t j = 0; j < d.cols; ++j) {
      if (!o.lower_bounds.empty()) lower[j] = o.lower_bounds[j];
      if (!o.upper_bounds.empty()) upper[j] = o.upper_bounds[j];
      if (o.positive) lower[j] = std::max(lower[j], 0.);
    }
    lower.back() = o.intercept_lower;
    upper.back() = o.intercept_upper;
    for (std::size_t j = 0; j < lower.size(); ++j) {
      require(!std::isnan(lower[j]) && !std::isnan(upper[j]) && lower[j] <= upper[j] &&
                  lower[j] != INFINITY && upper[j] != -INFINITY,
              "invalid coefficient bounds");
      if (j < d.cols || o.fit_intercept)
        constrained |= std::isfinite(lower[j]) || std::isfinite(upper[j]);
    }
    require(o.fit_intercept || (lower.back() <= 0 && upper.back() >= 0),
            "disabled intercept must allow zero");
    if (!o.fit_intercept) lower.back() = upper.back() = 0;
    require(o.solver != "cg" || (!constrained && o.l1 == 0), "CG requires no L1 or bounds");
    require(weights.empty() || weights.size() == d.rows, "invalid sample weight shape");
    if (!weights.empty()) {
      double maximum = 0;
      for (double weight : weights) {
        require(std::isfinite(weight) && weight >= 0,
                "sample weights must be finite and nonnegative");
        maximum = std::max(maximum, weight);
      }
      require(maximum > 0, "sample weights must have positive mass");
      double total = 0;
      for (double weight : weights) total += weight / maximum;
      probabilities.reserve(weights.size());
      for (double weight : weights) probabilities.push_back((weight / maximum) / total);
    }
  }
  double weight(std::size_t i) const {
    return probabilities.empty() ? 1. / data.rows : probabilities[i];
  }
  double clamp(double value, std::size_t j) const {
    return std::max(lower[j], std::min(upper[j], value));
  }
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
    for (std::size_t i = 0; i < data.rows; ++i) {
      if (weight(i) == 0) continue;
      double residual = data.dot(i, w) + w[data.cols] - data.labels[i];
      loss += .5 * weight(i) * residual * residual;
      data.add(i, weight(i) * residual, g);
      if (options.fit_intercept) g[data.cols] += weight(i) * residual;
    }
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
    require(std::isfinite(w[j]) && w[j] >= problem.lower[j] && w[j] <= problem.upper[j],
            "parameters must be finite and feasible");
  return problem.evaluate(w, g);
}

LinearResult fit_linear(const Dataset& d, const LinearOptions& o,
                        absl::Span<const double> weights) {
  Problem problem(d, o, weights);
  const auto n = d.cols + 1;
  LinearResult result;
  result.solver =
      o.solver == "auto" ? (problem.constrained || o.l1 > 0 ? "fista" : "cg") : o.solver;
  result.status = "max_iter";
  std::vector<double> w(n), g(n), next(n), next_g(n), anchor(n), anchor_g(n), diagonal(n, o.l2),
      direction(n), hp(n), z(n);
  for (std::size_t j = 0; j < n; ++j) w[j] = problem.clamp(0., j);
  if (o.fit_intercept) {
    double mean = 0, first = 0;
    bool seen = false, constant = true;
    for (std::size_t i = 0; i < d.rows; ++i) {
      const auto probability = problem.weight(i);
      if (probability == 0) continue;
      mean += probability * d.labels[i];
      if (!seen) {
        first = d.labels[i];
        seen = true;
      } else {
        constant = constant && d.labels[i] == first;
      }
    }
    // Preserve a constant target exactly, rather than accumulating a rounded
    // normalized sum. Finite R² uses an exact-prediction policy on constants.
    w.back() = problem.clamp(constant ? first : mean, d.cols);
  }
  auto start = std::chrono::steady_clock::now();
  auto evaluate = [&](absl::Span<const double> parameters, absl::Span<double> gradient) {
    ++result.work.full_evaluations;
    result.work.row_visits += d.rows;
    return problem.evaluate(parameters, gradient);
  };
  double loss = evaluate(w, absl::MakeSpan(g));
  auto record = [&](std::size_t iteration) {
    const auto stationarity = problem.stationarity(w, g);
    if (!std::isfinite(loss) || !std::isfinite(norm(g)) || !std::isfinite(stationarity))
      throw std::runtime_error("non-finite linear objective; scale features and targets");
    result.history.push_back(
        {iteration, loss, stationarity,
         std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
         result.work});
    result.converged = stationarity <= o.tolerance;
  };
  if (result.solver == "cg") {
    diagonal.back() = o.fit_intercept ? 1. : 0.;
    // Jacobi preconditioning streams data once; never form X'X.
    for (std::size_t i = 0; i < d.rows; ++i) {
      if (problem.weight(i) == 0) continue;
      if (d.sparse()) {
        for (auto k = d.offsets[i]; k < d.offsets[i + 1]; ++k)
          diagonal[d.indices[k]] += problem.weight(i) * d.values[k] * d.values[k];
      } else {
        for (std::size_t j = 0; j < d.cols; ++j) {
          auto value = d.values[i * d.cols + j];
          diagonal[j] += problem.weight(i) * value * value;
        }
      }
    }
    result.work.row_visits += d.rows;
    for (std::size_t j = 0; j < n; ++j) {
      if (!(diagonal[j] > 0) || !std::isfinite(diagonal[j])) diagonal[j] = 1.;
      direction[j] = z[j] = -g[j] / diagonal[j];
    }
  }
  record(0);
  anchor = w;
  double momentum = 1, step = o.learning_rate;
  for (std::size_t iteration = 1; iteration <= o.max_iter && !result.converged; ++iteration) {
    if (result.solver == "cg") {
      const double rz = -dot(g, z);
      std::fill(hp.begin(), hp.end(), 0);
      for (std::size_t i = 0; i < d.rows; ++i) {
        if (problem.weight(i) == 0) continue;
        const double a = problem.weight(i) * (d.dot(i, direction) + direction.back());
        d.add(i, a, absl::MakeSpan(hp));
        if (o.fit_intercept) hp.back() += a;
      }
      for (std::size_t j = 0; j < d.cols; ++j) hp[j] += o.l2 * direction[j];
      ++result.work.hessian_vector_products;
      result.work.row_visits += d.rows;
      const double curvature = dot(direction, hp);
      if (!(curvature > 0) || !std::isfinite(curvature) || !std::isfinite(rz)) {
        result.status = "numerical_breakdown";
        break;
      }
      const double alpha = rz / curvature;
      for (std::size_t j = 0; j < n; ++j) next[j] = w[j] + alpha * direction[j];
      const double new_loss = evaluate(next, absl::MakeSpan(next_g));
      for (std::size_t j = 0; j < n; ++j) z[j] = -next_g[j] / diagonal[j];
      const double beta = -dot(next_g, z) / rz;
      for (std::size_t j = 0; j < n; ++j) direction[j] = z[j] + beta * direction[j];
      w.swap(next);
      g.swap(next_g);
      loss = new_loss;
    } else {
      bool accepted = false;
      // One retry from the last feasible point implements a monotone restart.
      for (int restart = 0; restart < 2 && !accepted; ++restart) {
        if (result.solver == "proximal-gd" || restart) {
          anchor = w;
          momentum = 1;
        }
        evaluate(anchor, absl::MakeSpan(anchor_g));
        double candidate_loss = 0;
        for (int backtrack = 0; backtrack < 64; ++backtrack) {
          double squared = 0;
          for (std::size_t j = 0; j < n; ++j) {
            next[j] = problem.prox(anchor[j] - step * anchor_g[j], step, j);
            const double delta = next[j] - anchor[j];
            squared += delta * delta;
          }
          candidate_loss = evaluate(next, absl::MakeSpan(next_g));
          // Quadratic majorization without subtracting nearly equal objectives:
          // delta' H delta <= ||delta||² / step. The smooth gradient is affine.
          double curvature = 0;
          for (std::size_t j = 0; j < n; ++j)
            curvature += (next[j] - anchor[j]) * (next_g[j] - anchor_g[j]);
          if (std::isfinite(candidate_loss) && std::isfinite(curvature) &&
              curvature <= squared / step * (1 + 1e-12)) {
            accepted = true;
            break;
          }
          step *= .5;
        }
        if (accepted && candidate_loss > loss + 1e-14 * (1 + std::abs(loss))) {
          accepted = false;
          continue;
        }
        if (accepted) {
          const double new_momentum = (1 + std::sqrt(1 + 4 * momentum * momentum)) / 2;
          for (std::size_t j = 0; j < n; ++j)
            anchor[j] = next[j] + (momentum - 1) / new_momentum * (next[j] - w[j]);
          momentum = new_momentum;
          w.swap(next);
          g.swap(next_g);
          loss = candidate_loss;
          step = std::min(o.learning_rate, step * 1.25);
        }
      }
      if (!accepted) {
        result.status = "line_search_failed";
        break;
      }
    }
    ++result.work.parameter_updates;
    record(iteration);
  }
  if (result.converged) result.status = "converged";
  result.intercept = w.back();
  w.pop_back();
  result.weights = std::move(w);
  return result;
}

std::vector<double> predict_linear(const Dataset& d, const LinearResult& model) {
  d.validate(false);
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
}  // namespace logitlab
