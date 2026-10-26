#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "logitlab/generalized_linear_model.hpp"

namespace logitlab {
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
enum class Family { gaussian, binomial, poisson };
Family family(const std::string& name) {
  if (name == "gaussian") return Family::gaussian;
  if (name == "binomial") return Family::binomial;
  if (name == "poisson") return Family::poisson;
  throw std::invalid_argument("family must be gaussian, binomial or poisson");
}
std::string link(Family f) {
  return f == Family::gaussian ? "identity" : f == Family::binomial ? "logit" : "log";
}
double dot(absl::Span<const double> a, absl::Span<const double> b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.);
}
double norm(absl::Span<const double> a) {
  double result = 0;
  for (double value : a) result = std::hypot(result, value);
  return result;
}
bool finite(absl::Span<const double> a) {
  return std::all_of(a.begin(), a.end(), [](double value) { return std::isfinite(value); });
}
double softplus(double eta) { return std::max(eta, 0.) + std::log1p(std::exp(-std::abs(eta))); }
void validate_offset(absl::Span<const double> offset, std::size_t rows) {
  require(offset.empty() || (offset.size() == rows && finite(offset)),
          "offset must be finite with one value per row");
}
struct Problem {
  const Dataset& data;
  GLMOptions options;
  Family distribution;
  absl::Span<const double> offset;
  std::vector<double> q;
  Problem(const Dataset& d, GLMOptions o, absl::Span<const double> weights,
          absl::Span<const double> offsets)
      : data(d), options(std::move(o)), distribution(family(options.family)), offset(offsets) {
    d.validate(true, false);
    require(options.link == "auto" || options.link == link(distribution),
            "unsupported family/link pair; canonical links only");
    require(options.max_iter > 0 && options.max_cg_iter > 0 && std::isfinite(options.l2) &&
                options.l2 >= 0 && std::isfinite(options.tolerance) && options.tolerance > 0,
            "invalid GLM optimization options");
    validate_offset(offset, d.rows);
    require(weights.empty() || weights.size() == d.rows, "invalid sample weight shape");
    q.assign(d.rows, 1.);
    if (!weights.empty()) {
      require(finite(weights) && std::all_of(weights.begin(), weights.end(),
                                             [](double value) { return value >= 0; }),
              "sample weights must be finite and nonnegative");
      const double maximum = *std::max_element(weights.begin(), weights.end());
      require(maximum > 0, "sample weights must have positive mass");
      for (std::size_t i = 0; i < d.rows; ++i) q[i] = weights[i] / maximum;
    }
    const double mass = std::accumulate(q.begin(), q.end(), 0.);
    for (std::size_t i = 0; i < d.rows; ++i) {
      q[i] /= mass;
      if (distribution == Family::binomial)
        require(d.labels[i] >= 0 && d.labels[i] <= 1, "binomial responses must be in [0,1]");
      if (distribution == Family::poisson)
        require(d.labels[i] >= 0, "poisson responses must be nonnegative");
    }
  }
  double eta(std::size_t i, absl::Span<const double> theta) const {
    return data.dot(i, theta.subspan(0, data.cols)) + theta.back() +
           (offset.empty() ? 0 : offset[i]);
  }
  // Objective, gradient, Hessian row factors and Jacobi diagonal share one pass.
  double evaluate(absl::Span<const double> theta, std::vector<double>& gradient,
                  std::vector<double>& curvature, std::vector<double>& diagonal, Work& work) const {
    gradient.assign(theta.size(), 0.);
    diagonal.assign(theta.size(), 0.);
    curvature.assign(data.rows, 0.);
    double objective = 0;
    ++work.full_evaluations;
    for (std::size_t i = 0; i < data.rows; ++i) {
      ++work.row_visits;
      if (q[i] == 0) continue;
      const double z = eta(i, theta), y = data.labels[i];
      if (!std::isfinite(z)) return INFINITY;
      double loss, residual, variance;
      if (distribution == Family::gaussian) {
        residual = z - y;
        loss = .5 * residual * residual;
        variance = 1.;
      } else if (distribution == Family::binomial) {
        loss = y * softplus(-z) + (1 - y) * softplus(z);
        residual = z >= 0 ? (1 - y) - sigmoid(-z) : sigmoid(z) - y;
        const double t = std::exp(-std::abs(z));
        variance = t / ((1 + t) * (1 + t));
      } else {
        const double mean = std::exp(z);
        loss = mean - y * z;
        residual = mean - y;
        variance = mean;
      }
      objective += q[i] * loss;
      const double r = q[i] * residual, h = q[i] * variance;
      data.add(i, r, absl::MakeSpan(gradient).subspan(0, data.cols));
      curvature[i] = h;
      if (data.sparse()) {
        for (std::size_t k = data.offsets[i]; k < data.offsets[i + 1]; ++k)
          diagonal[data.indices[k]] += h * data.values[k] * data.values[k];
      } else {
        for (std::size_t j = 0; j < data.cols; ++j) {
          const double x = data.values[i * data.cols + j];
          diagonal[j] += h * x * x;
        }
      }
      if (options.fit_intercept) {
        gradient.back() += r;
        diagonal.back() += h;
      }
    }
    for (std::size_t j = 0; j < data.cols; ++j) {
      objective += .5 * options.l2 * theta[j] * theta[j];
      gradient[j] += options.l2 * theta[j];
      diagonal[j] += options.l2;
    }
    return objective;
  }
  void hvp(absl::Span<const double> vector, absl::Span<const double> curvature,
           std::vector<double>& out, Work& work) const {
    out.assign(vector.size(), 0.);
    for (std::size_t i = 0; i < data.rows; ++i) {
      if (curvature[i] == 0) continue;
      const double value =
          curvature[i] * (data.dot(i, vector.subspan(0, data.cols)) + vector.back());
      data.add(i, value, absl::MakeSpan(out).subspan(0, data.cols));
      if (options.fit_intercept) out.back() += value;
    }
    for (std::size_t j = 0; j < data.cols; ++j) out[j] += options.l2 * vector[j];
    ++work.hessian_vector_products;
    work.row_visits += data.rows;
  }
  std::vector<double> initial() const {
    std::vector<double> theta(data.cols + 1, 0.);
    if (!options.fit_intercept) return theta;
    double mean = 0, average_offset = 0;
    bool constant = true;
    double first = 0;
    bool have_first = false;
    for (std::size_t i = 0; i < data.rows; ++i) {
      if (q[i] == 0) continue;
      mean += q[i] * data.labels[i];
      if (!offset.empty()) average_offset += q[i] * offset[i];
      const double value = data.labels[i] - (offset.empty() ? 0 : offset[i]);
      if (!have_first) {
        first = value;
        have_first = true;
      } else
        constant &= value == first;
    }
    if (distribution == Family::gaussian)
      theta.back() = constant ? first : mean - average_offset;
    else if (distribution == Family::binomial) {
      mean = std::clamp(mean, 1e-3, 1 - 1e-3);
      theta.back() = std::log(mean) - std::log1p(-mean) - average_offset;
    } else {
      double maximum_offset = -INFINITY, maximum_y = 0;
      for (std::size_t i = 0; i < data.rows; ++i)
        if (q[i] > 0) {
          maximum_offset = std::max(maximum_offset, offset.empty() ? 0. : offset[i]);
          maximum_y = std::max(maximum_y, data.labels[i]);
        }
      double response = 0, mass = 0;
      for (std::size_t i = 0; i < data.rows; ++i)
        if (q[i] > 0) {
          mass += q[i] * std::exp((offset.empty() ? 0 : offset[i]) - maximum_offset);
          if (maximum_y > 0) response += q[i] * (data.labels[i] / maximum_y);
        }
      const double log_mean =
          maximum_y > 0 ? std::log(maximum_y) + std::log(response) : std::log(1e-3);
      theta.back() = (log_mean - std::log(mass)) - maximum_offset;
    }
    return theta;
  }
};
std::vector<double> direction(const Problem& problem, absl::Span<const double> gradient,
                              absl::Span<const double> curvature, absl::Span<const double> diagonal,
                              Work& work) {
  const std::size_t size = gradient.size();
  std::vector<double> result(size, 0.), residual(size), z(size), p(size), hp(size);
  for (std::size_t j = 0; j < size; ++j) {
    residual[j] = -gradient[j];
    z[j] = residual[j] / std::max(diagonal[j], 1e-12);
  }
  p = z;
  double rz = dot(residual, z);
  const double stop = std::min(.5, std::sqrt(norm(gradient))) * norm(gradient);
  for (std::size_t k = 0; k < std::min(size, problem.options.max_cg_iter); ++k) {
    problem.hvp(p, curvature, hp, work);
    const double php = dot(p, hp);
    if (!(php > 0) || !std::isfinite(php) || !std::isfinite(rz)) break;
    const double alpha = rz / php;
    for (std::size_t j = 0; j < size; ++j) {
      result[j] += alpha * p[j];
      residual[j] -= alpha * hp[j];
    }
    if (norm(residual) <= stop) break;
    for (std::size_t j = 0; j < size; ++j) z[j] = residual[j] / std::max(diagonal[j], 1e-12);
    const double next = dot(residual, z);
    if (!(rz > 0)) break;
    for (std::size_t j = 0; j < size; ++j) p[j] = z[j] + next / rz * p[j];
    rz = next;
  }
  if (!finite(result) || !(dot(gradient, result) < 0))
    for (std::size_t j = 0; j < size; ++j) result[j] = -gradient[j] / std::max(diagonal[j], 1e-12);
  return result;
}
void validate_model(const Dataset& data, const GLMResult& model, absl::Span<const double> offset) {
  data.validate(false, false);
  require(
      model.weights.size() == data.cols && finite(model.weights) && std::isfinite(model.intercept),
      "invalid GLM model or prediction feature count");
  require(model.link == link(family(model.family)), "invalid GLM model family/link");
  validate_offset(offset, data.rows);
}
}  // namespace

double glm_objective(const Dataset& data, absl::Span<const double> theta, const GLMOptions& options,
                     absl::Span<double> gradient, absl::Span<const double> weights,
                     absl::Span<const double> offset) {
  Problem problem(data, options, weights, offset);
  require(theta.size() == data.cols + 1 && gradient.size() == theta.size() && finite(theta),
          "invalid GLM parameter shape or values");
  require(options.fit_intercept || theta.back() == 0, "disabled intercept must be zero");
  std::vector<double> g, h, diagonal;
  Work work;
  const double value = problem.evaluate(theta, g, h, diagonal, work);
  if (!std::isfinite(value) || !finite(g))
    throw std::runtime_error("GLM objective overflow; scale inputs");
  std::copy(g.begin(), g.end(), gradient.begin());
  return value;
}

GLMResult fit_glm(const Dataset& data, const GLMOptions& options, absl::Span<const double> weights,
                  absl::Span<const double> offset) {
  Problem problem(data, options, weights, offset);
  GLMResult result;
  result.family = options.family;
  result.link = link(problem.distribution);
  result.status = "max_iter";
  auto theta = problem.initial();
  std::vector<double> gradient, curvature, diagonal;
  const auto start = std::chrono::steady_clock::now();
  double loss = problem.evaluate(theta, gradient, curvature, diagonal, result.work);
  if (!finite(theta) || !std::isfinite(loss) || !finite(gradient) || !finite(diagonal))
    throw std::runtime_error("GLM initialization overflow; scale features, targets or offsets");
  auto record = [&](std::size_t iteration) {
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.history.push_back({iteration, loss, norm(gradient), seconds, result.work});
  };
  record(0);
  for (std::size_t iteration = 1; iteration <= options.max_iter; ++iteration) {
    if (norm(gradient) <= options.tolerance) {
      result.converged = true;
      break;
    }
    const auto step_direction = direction(problem, gradient, curvature, diagonal, result.work);
    const double slope = dot(gradient, step_direction);
    if (!(slope < 0) || !std::isfinite(slope)) {
      result.status = "numerical_breakdown";
      break;
    }
    std::vector<double> candidate(theta.size()), new_gradient, new_curvature, new_diagonal;
    double step = 1., new_loss = INFINITY;
    bool accepted = false;
    for (std::size_t attempt = 0; attempt < 60; ++attempt) {
      for (std::size_t j = 0; j < theta.size(); ++j)
        candidate[j] = theta[j] + step * step_direction[j];
      new_loss =
          problem.evaluate(candidate, new_gradient, new_curvature, new_diagonal, result.work);
      if (finite(candidate) && std::isfinite(new_loss) && finite(new_gradient) &&
          finite(new_diagonal) && new_loss <= loss + 1e-4 * step * slope) {
        accepted = true;
        break;
      }
      step *= .5;
    }
    if (!accepted) {
      result.status = "line_search_failed";
      break;
    }
    theta.swap(candidate);
    gradient.swap(new_gradient);
    curvature.swap(new_curvature);
    diagonal.swap(new_diagonal);
    loss = new_loss;
    ++result.work.parameter_updates;
    record(iteration);
  }
  result.converged = norm(gradient) <= options.tolerance;
  if (result.converged) result.status = "converged";
  result.weights.assign(theta.begin(), theta.end() - 1);
  result.intercept = theta.back();
  return result;
}

std::vector<double> predict_glm(const Dataset& data, const GLMResult& model,
                                absl::Span<const double> offset, bool linear) {
  validate_model(data, model, offset);
  const auto f = family(model.family);
  std::vector<double> prediction(data.rows);
  for (std::size_t i = 0; i < data.rows; ++i) {
    const double eta =
        data.dot(i, model.weights) + model.intercept + (offset.empty() ? 0 : offset[i]);
    const double mean = linear || f == Family::gaussian ? eta
                        : f == Family::binomial         ? sigmoid(eta)
                                                        : std::exp(eta);
    if (!std::isfinite(eta) || !std::isfinite(mean))
      throw std::runtime_error("GLM prediction overflow");
    prediction[i] = mean;
  }
  return prediction;
}

double glm_deviance(const Dataset& data, const GLMResult& model, absl::Span<const double> weights,
                    absl::Span<const double> offset) {
  validate_model(data, model, offset);
  GLMOptions options;
  options.family = model.family;
  options.link = model.link;
  Problem problem(data, options, weights, offset);
  double deviance = 0;
  std::vector<double> theta = model.weights;
  theta.push_back(model.intercept);
  for (std::size_t i = 0; i < data.rows; ++i) {
    if (problem.q[i] == 0) continue;
    const double eta = problem.eta(i, theta), y = data.labels[i];
    double unit;
    if (problem.distribution == Family::gaussian)
      unit = (y - eta) * (y - eta);
    else if (problem.distribution == Family::binomial) {
      unit = y * softplus(-eta) + (1 - y) * softplus(eta);
      if (y > 0) unit += y * std::log(y);
      if (y < 1) unit += (1 - y) * std::log1p(-y);
      unit *= 2;
    } else if (y == 0)
      unit = 2 * std::exp(eta);
    else {
      const double difference = eta - std::log(y);
      if (std::abs(difference) < 1e-4)
        unit = y * difference * difference * (1 + difference / 3 + difference * difference / 12);
      else
        unit = 2 * (std::exp(eta) - y - y * difference);
    }
    if (!std::isfinite(eta) || !std::isfinite(unit))
      throw std::runtime_error("GLM deviance overflow");
    deviance += problem.q[i] * std::max(0., unit);
  }
  if (!std::isfinite(deviance)) throw std::runtime_error("GLM deviance overflow");
  return deviance;
}
}  // namespace logitlab
