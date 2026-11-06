#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "detail/families.hpp"
#include "detail/glm_api.hpp"
#include "detail/kernels.hpp"
#include "detail/validation.hpp"
#include "detail/weights.hpp"
#include "logitlab/generalized_linear_model.hpp"

namespace logitlab {
namespace {
using detail::dot;
using detail::Family;
using detail::family;
using detail::finite;
using detail::link;
using detail::norm;
using detail::require;
using detail::softplus;
void validate_offset(absl::Span<const double> offset, std::size_t rows) {
  require(offset.empty() || (offset.size() == rows && finite(offset)),
          "offset must be finite with one value per row");
}
struct Problem {
  const Dataset& data;
  GLMOptions options;
  Family distribution;
  absl::Span<const double> offset;
  detail::ObservationWeights q;
  Problem(const Dataset& d, GLMOptions o, absl::Span<const double> weights,
          absl::Span<const double> offsets, bool known_storage = false)
      : data(detail::checked_data(d, false, known_storage)),
        options(std::move(o)),
        distribution(family(options.family)),
        offset(offsets),
        q(d.rows, weights) {
    require(options.link == "auto" || options.link == link(distribution),
            "unsupported family/link pair; canonical links only");
    require(options.max_iter > 0 && options.max_cg_iter > 0 && std::isfinite(options.l2) &&
                options.l2 >= 0 && std::isfinite(options.tolerance) && options.tolerance > 0,
            "invalid GLM optimization options");
    validate_offset(offset, d.rows);
    for (double response : d.labels) {
      if (distribution == Family::binomial)
        require(response >= 0 && response <= 1, "binomial responses must be in [0,1]");
      if (distribution == Family::poisson)
        require(response >= 0, "poisson responses must be nonnegative");
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
    const bool zero_features = detail::zero_coefficients(theta, data.cols);
    ++work.full_evaluations;
    detail::with_family(distribution, [&](auto response) {
      detail::with_rows(data, [&](const auto& rows) {
        for (std::size_t begin = 0; begin < data.rows; begin += detail::kResponseBatch) {
          const auto count = std::min(detail::kResponseBatch, data.rows - begin);
          detail::ResponseBatch batch;
          for (std::size_t lane = 0; lane < count; ++lane) {
            const auto i = begin + lane;
            ++work.row_visits;
            batch.eta[lane] = 0;
            batch.target[lane] = 0;
            if (q[i] == 0) continue;
            batch.eta[lane] = (zero_features ? 0. : rows[i].dot(theta)) + theta.back() +
                              (offset.empty() ? 0. : offset[i]);
            if (!std::isfinite(batch.eta[lane])) {
              objective = INFINITY;
              return;
            }
            batch.target[lane] = data.labels[i];
          }
          response.batch(batch, count);
          for (std::size_t lane = 0; lane < count; ++lane) {
            const auto i = begin + lane;
            if (q[i] == 0) continue;
            const auto& moment = batch.values[lane];
            objective += q[i] * moment.loss;
            const double r = q[i] * moment.residual, h = q[i] * moment.variance;
            curvature[i] = h;
            rows[i].add_moments(r, h, absl::MakeSpan(gradient), absl::MakeSpan(diagonal));
            if (options.fit_intercept) {
              gradient.back() += r;
              diagonal.back() += h;
            }
          }
        }
      });
    });
    for (std::size_t j = 0; j < data.cols; ++j) {
      objective += .5 * options.l2 * theta[j] * theta[j];
      gradient[j] += options.l2 * theta[j];
      diagonal[j] += options.l2;
    }
    return objective;
  }
  void hvp(absl::Span<const double> vector, absl::Span<const double> curvature,
           std::vector<double>& out, Work& work) const {
    out.resize(vector.size());
    detail::hessian_vector(data, curvature, vector, options.l2, options.fit_intercept,
                           absl::MakeSpan(out));
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
    } else
      theta.back() = poisson_intercept();
    return theta;
  }
  double poisson_intercept() const {
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
    return (log_mean - std::log(mass)) - maximum_offset;
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
  const double gradient_norm = norm(gradient);
  // Avoid oversolving the local linear model near the requested full-gradient
  // tolerance. Only a fresh complete gradient can certify outer convergence.
  const double floor = .5 * problem.options.tolerance;
  const double stop = problem.distribution == Family::gaussian
                          ? floor
                          : std::max(floor, std::min(.5, std::sqrt(gradient_norm)) * gradient_norm);
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
void validate_model(const Dataset& data, const GLMResult& model, absl::Span<const double> offset,
                    bool known_storage = false) {
  if (!known_storage) data.validate(false, false);
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

namespace {
struct Trial {
  std::vector<double> candidate, gradient, curvature, diagonal;
  double loss = INFINITY;
};
bool backtrack(const Problem& problem, absl::Span<const double> theta,
               absl::Span<const double> direction, double loss, double slope, Work& work,
               Trial& trial) {
  double step = 1.;
  for (std::size_t attempt = 0; attempt < 60; ++attempt) {
    for (std::size_t j = 0; j < theta.size(); ++j)
      trial.candidate[j] = theta[j] + step * direction[j];
    trial.loss =
        problem.evaluate(trial.candidate, trial.gradient, trial.curvature, trial.diagonal, work);
    if (finite(trial.candidate) && std::isfinite(trial.loss) && finite(trial.gradient) &&
        finite(trial.diagonal) && trial.loss <= loss + 1e-4 * step * slope)
      return true;
    step *= .5;
  }
  return false;
}
}  // namespace

GLMResult fit_impl(const Dataset& data, const GLMOptions& options, absl::Span<const double> weights,
                   absl::Span<const double> offset, bool known_storage) {
  Problem problem(data, options, weights, offset, known_storage);
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
  Trial trial;
  trial.candidate.resize(theta.size());
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
    if (!backtrack(problem, theta, step_direction, loss, slope, result.work, trial)) {
      result.status = "line_search_failed";
      break;
    }
    theta.swap(trial.candidate);
    gradient.swap(trial.gradient);
    curvature.swap(trial.curvature);
    diagonal.swap(trial.diagonal);
    loss = trial.loss;
    ++result.work.parameter_updates;
    record(iteration);
  }
  result.converged = norm(gradient) <= options.tolerance;
  if (result.converged) result.status = "converged";
  result.weights.assign(theta.begin(), theta.end() - 1);
  result.intercept = theta.back();
  return result;
}

std::vector<double> predict_impl(const Dataset& data, const GLMResult& model,
                                 absl::Span<const double> offset, bool linear, bool known_storage) {
  validate_model(data, model, offset, known_storage);
  const auto f = family(model.family);
  std::vector<double> prediction(data.rows);
  for (std::size_t i = 0; i < data.rows; ++i) {
    const double eta =
        data.dot(i, model.weights) + model.intercept + (offset.empty() ? 0 : offset[i]);
    const double mean = linear || f == Family::gaussian ? eta
                        : f == Family::binomial         ? detail::sigmoid(eta)
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
  Problem problem(data, options, weights, offset, true);
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
      unit = detail::Binomial{}(eta, y).loss;
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
GLMResult fit_glm(const Dataset& data, const GLMOptions& options, absl::Span<const double> weights,
                  absl::Span<const double> offset) {
  return fit_impl(data, options, weights, offset, false);
}
GLMResult detail::fit_glm_validated(const Dataset& data, const GLMOptions& options,
                                    absl::Span<const double> weights,
                                    absl::Span<const double> offset) {
  return fit_impl(data, options, weights, offset, true);
}
std::vector<double> predict_glm(const Dataset& data, const GLMResult& model,
                                absl::Span<const double> offset, bool linear) {
  return predict_impl(data, model, offset, linear, false);
}
std::vector<double> detail::predict_glm_validated(const Dataset& data, const GLMResult& model,
                                                  absl::Span<const double> offset, bool linear) {
  return predict_impl(data, model, offset, linear, true);
}
}  // namespace logitlab
