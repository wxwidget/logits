#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

#include "logitlab/logistic_regression.hpp"

namespace logitlab {
namespace {
double inner(absl::Span<const double> a, absl::Span<const double> b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}
double norm(absl::Span<const double> a) { return std::sqrt(inner(a, a)); }
void require(bool b, const char* message) {
  if (!b) throw std::invalid_argument(message);
}
}  // namespace
void Dataset::validate(bool labels_required, bool binary_labels) const {
  require(rows > 0 && cols > 0, "data must have positive dimensions");
  require(rows < std::numeric_limits<std::size_t>::max(), "row count overflow");
  require(cols < std::numeric_limits<std::size_t>::max(), "column count overflow");
  if (sparse()) {
    require(offsets.size() == rows + 1 && offsets.front() == 0 && offsets.back() == values.size(),
            "invalid CSR offsets");
    require(indices.size() == values.size(), "invalid CSR indices");
    for (std::size_t i = 0; i < rows; ++i) {
      require(offsets[i] <= offsets[i + 1] && offsets[i + 1] <= values.size(), "invalid CSR row");
      for (auto k = offsets[i]; k < offsets[i + 1]; ++k) {
        require(indices[k] < cols, "CSR index out of range");
        require(k == offsets[i] || indices[k - 1] < indices[k],
                "CSR indices must be sorted and unique");
      }
    }
  } else {
    require(indices.empty(), "dense data must not contain CSR indices");
    require(rows <= std::numeric_limits<std::size_t>::max() / cols && values.size() == rows * cols,
            "invalid dense shape");
  }
  for (double v : values) require(std::isfinite(v), "features must be finite");
  if (labels_required) {
    require(labels.size() == rows, "label count does not match rows");
    for (double y : labels) {
      require(std::isfinite(y), "targets must be finite");
      if (binary_labels) require(y == 0 || y == 1, "labels must be 0 or 1");
    }
  }
}
double Dataset::dot(std::size_t row, absl::Span<const double> w) const {
  double s = 0;
  if (sparse())
    for (auto k = offsets[row]; k < offsets[row + 1]; ++k) s += values[k] * w[indices[k]];
  else
    for (std::size_t j = 0; j < cols; ++j) s += values[row * cols + j] * w[j];
  return s;
}
void Dataset::add(std::size_t row, double scale, absl::Span<double> g) const {
  if (sparse())
    for (auto k = offsets[row]; k < offsets[row + 1]; ++k) g[indices[k]] += scale * values[k];
  else
    for (std::size_t j = 0; j < cols; ++j) g[j] += scale * values[row * cols + j];
}
double sigmoid(double x) {
  if (x >= 0) return 1 / (1 + std::exp(-x));
  double e = std::exp(x);
  return e / (1 + e);
}
namespace {
double objective_unchecked(const Dataset& d, absl::Span<const double> w, const Options& o,
                           absl::Span<double> g) {
  require(w.size() == d.cols + 1 && g.size() == w.size(), "invalid parameter shape");
  std::fill(g.begin(), g.end(), 0);
  double f = 0;
  for (std::size_t i = 0; i < d.rows; ++i) {
    double z = d.dot(i, w) + (o.fit_intercept ? w[d.cols] : 0);
    // Avoid inf-inf cancellation for large, correctly classified margins.
    double margin = d.labels[i] == 1 ? -z : z;
    f += std::max(margin, 0.0) + std::log1p(std::exp(-std::abs(margin)));
    double residual = (sigmoid(z) - d.labels[i]) / d.rows;
    d.add(i, residual, g);
    if (o.fit_intercept) g[d.cols] += residual;
  }
  f /= d.rows;
  for (std::size_t j = 0; j < d.cols; ++j) {
    f += 0.5 * o.l2 * w[j] * w[j];
    g[j] += o.l2 * w[j];
  }
  return f;
}
}  // namespace

double objective(const Dataset& d, absl::Span<const double> w, const Options& o,
                 absl::Span<double> g) {
  d.validate();
  require(std::isfinite(o.l2) && o.l2 >= 0, "l2 must be finite and nonnegative");
  require(w.size() == d.cols + 1 && g.size() == w.size(), "invalid parameter shape");
  for (double value : w) require(std::isfinite(value), "parameters must be finite");
  return objective_unchecked(d, w, o, g);
}

Result fit(const Dataset& d, const Options& o) {
  d.validate();
  require(o.max_epochs > 0 && o.batch_size > 0 && o.history_size > 0,
          "iteration, batch and history sizes must be positive");
  require(std::isfinite(o.learning_rate) && o.learning_rate > 0 && std::isfinite(o.l2) &&
              o.l2 >= 0 && std::isfinite(o.tolerance) && o.tolerance > 0,
          "invalid optimization options");
  require(o.solver == "gd" || o.solver == "sgd" || o.solver == "adam" || o.solver == "lbfgs" ||
              o.solver == "newton-cg",
          "unknown solver");
  const auto n = d.cols + 1;
  const bool stochastic = o.solver == "sgd" || o.solver == "adam";
  const bool adaptive = o.solver == "adam";
  std::vector<double> w(n, 0), g(n), direction(n), candidate(n), next_g(n);
  std::vector<double> m(adaptive ? n : 0, 0), v(adaptive ? n : 0, 0);
  std::vector<std::vector<double>> ss, ys;
  std::vector<std::size_t> order(stochastic ? d.rows : 0);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 rng(o.seed);
  std::size_t updates = 0;
  auto start = std::chrono::steady_clock::now();
  Result result;
  result.status = "max_epochs";
  auto evaluate = [&](absl::Span<const double> parameters, absl::Span<double> gradient) {
    ++result.work.full_evaluations;
    result.work.row_visits += d.rows;
    return objective_unchecked(d, parameters, o, gradient);
  };
  double f = evaluate(w, absl::MakeSpan(g));
  auto record = [&](std::size_t epoch) {
    if (!std::isfinite(f) || !std::isfinite(norm(g)))
      throw std::runtime_error("non-finite objective; scale features or reduce learning rate");
    result.history.push_back(
        {epoch, f, norm(g),
         std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
         result.work});
  };
  record(0);
  for (std::size_t epoch = 1; epoch <= o.max_epochs; ++epoch) {
    if (norm(g) <= o.tolerance) {
      result.converged = true;
      result.status = "converged";
      break;
    }
    if (stochastic) {
      std::shuffle(order.begin(), order.end(), rng);
      for (std::size_t begin = 0; begin < d.rows;) {
        auto count = std::min(o.batch_size, d.rows - begin);
        ++result.work.batch_gradients;
        ++result.work.parameter_updates;
        result.work.row_visits += count;
        std::fill(next_g.begin(), next_g.end(), 0);
        for (std::size_t k = begin; k < begin + count; ++k) {
          auto i = order[k];
          double z = d.dot(i, w) + (o.fit_intercept ? w[d.cols] : 0);
          double r = (sigmoid(z) - d.labels[i]) / count;
          d.add(i, r, absl::MakeSpan(next_g));
          if (o.fit_intercept) next_g[d.cols] += r;
        }
        ++updates;
        const double correction1 = adaptive ? 1 - std::pow(0.9, updates) : 1;
        const double correction2 = adaptive ? 1 - std::pow(0.999, updates) : 1;
        for (std::size_t j = 0; j < n; ++j) {
          double a = next_g[j] + (j < d.cols ? o.l2 * w[j] : 0);
          if (o.solver == "adam") {
            m[j] = 0.9 * m[j] + 0.1 * a;
            v[j] = 0.999 * v[j] + 0.001 * a * a;
            w[j] -= o.learning_rate * (m[j] / correction1) / (std::sqrt(v[j] / correction2) + 1e-8);
          } else
            w[j] -= o.learning_rate * a;
        }
        begin += count;
      }
      f = evaluate(w, absl::MakeSpan(g));
    } else {
      direction = g;
      if (o.solver == "lbfgs" && !ss.empty()) {
        std::vector<double> alpha(ss.size());
        for (std::size_t k = ss.size(); k-- > 0;) {
          alpha[k] = inner(ss[k], direction) / inner(ss[k], ys[k]);
          for (std::size_t j = 0; j < n; ++j) direction[j] -= alpha[k] * ys[k][j];
        }
        double scale = inner(ss.back(), ys.back()) / inner(ys.back(), ys.back());
        for (auto& x : direction) x *= scale;
        for (std::size_t k = 0; k < ss.size(); ++k) {
          double beta = inner(ys[k], direction) / inner(ss[k], ys[k]);
          for (std::size_t j = 0; j < n; ++j) direction[j] += ss[k][j] * (alpha[k] - beta);
        }
      } else if (o.solver == "newton-cg") {
        // Hessian-vector products: no dense p-by-p Hessian allocation.
        std::vector<double> curvature(d.rows), r = g, p = g, hp(n), solution(n, 0);
        result.work.row_visits += d.rows;
        for (std::size_t i = 0; i < d.rows; ++i) {
          double q = sigmoid(d.dot(i, w) + (o.fit_intercept ? w[d.cols] : 0));
          curvature[i] = q * (1 - q) / d.rows;
        }
        double rr = inner(r, r), initial = rr;
        for (std::size_t k = 0; k < std::min<std::size_t>(n, 50); ++k) {
          ++result.work.hessian_vector_products;
          result.work.row_visits += d.rows;
          std::fill(hp.begin(), hp.end(), 0);
          for (std::size_t i = 0; i < d.rows; ++i) {
            double a = curvature[i] * (d.dot(i, p) + (o.fit_intercept ? p[d.cols] : 0));
            d.add(i, a, absl::MakeSpan(hp));
            if (o.fit_intercept) hp[d.cols] += a;
          }
          for (std::size_t j = 0; j < n; ++j)
            hp[j] += (j < d.cols ? o.l2 : 0) * p[j] + 1e-10 * p[j];
          double denom = inner(p, hp);
          if (denom <= 0) break;
          double a = rr / denom;
          for (std::size_t j = 0; j < n; ++j) {
            solution[j] += a * p[j];
            r[j] -= a * hp[j];
          }
          double next = inner(r, r);
          if (next <= initial * 1e-8) break;
          for (std::size_t j = 0; j < n; ++j) p[j] = r[j] + next / rr * p[j];
          rr = next;
        }
        direction = std::move(solution);
      }
      for (auto& x : direction) x = -x;
      double slope = inner(g, direction);
      if (!std::isfinite(slope) || slope >= 0) {
        direction = g;
        for (auto& x : direction) x = -x;
        slope = -inner(g, g);
      }
      double step = o.solver == "gd" ? o.learning_rate : 1;
      double next_f = 0;
      bool accepted = false;
      for (int ls = 0; ls < 40; ++ls) {
        for (std::size_t j = 0; j < n; ++j) candidate[j] = w[j] + step * direction[j];
        next_f = evaluate(candidate, absl::MakeSpan(next_g));
        if (std::isfinite(next_f) && next_f <= f + 1e-4 * step * slope) {
          accepted = true;
          break;
        }
        step *= 0.5;
      }
      if (!accepted) {
        result.status = "line_search_failed";
        break;
      }
      if (o.solver == "lbfgs") {
        std::vector<double> s(n), y(n);
        for (std::size_t j = 0; j < n; ++j) {
          s[j] = candidate[j] - w[j];
          y[j] = next_g[j] - g[j];
        }
        if (inner(s, y) > 1e-12 * norm(s) * norm(y)) {
          if (ss.size() == o.history_size) {
            ss.erase(ss.begin());
            ys.erase(ys.begin());
          }
          ss.push_back(std::move(s));
          ys.push_back(std::move(y));
        }
      }
      w.swap(candidate);
      ++result.work.parameter_updates;
      g.swap(next_g);
      f = next_f;
    }
    record(epoch);
  }
  if (norm(g) <= o.tolerance) {
    result.converged = true;
    result.status = "converged";
  }
  result.intercept = o.fit_intercept ? w.back() : 0;
  w.pop_back();
  result.weights = std::move(w);
  return result;
}
std::vector<double> predict(const Dataset& d, const Result& model) {
  d.validate(false);
  require(model.weights.size() == d.cols, "feature count mismatch");
  require(std::isfinite(model.intercept), "non-finite intercept");
  for (double w : model.weights) require(std::isfinite(w), "non-finite weights");
  std::vector<double> out(d.rows);
  for (std::size_t i = 0; i < d.rows; ++i) {
    const double score = d.dot(i, model.weights) + model.intercept;
    if (std::isnan(score)) throw std::runtime_error("undefined prediction score; scale features");
    out[i] = sigmoid(score);
  }
  return out;
}
}  // namespace logitlab
