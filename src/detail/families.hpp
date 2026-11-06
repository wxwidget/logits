#pragma once
#include <string>

#include "detail/numeric.hpp"
#include "detail/vector_math.hpp"

namespace logitlab::detail {
struct Moments {
  double loss, residual, variance;
};
constexpr std::size_t kResponseBatch = 4;
struct ResponseBatch {
  double eta[kResponseBatch]{}, target[kResponseBatch]{};
  Moments values[kResponseBatch]{};
};
struct Gaussian {
  Moments operator()(double eta, double y) const {
    const double residual = eta - y;
    return {.5 * residual * residual, residual, 1.};
  }
  void batch(ResponseBatch& values, std::size_t count) const {
    for (std::size_t i = 0; i < count; ++i)
      values.values[i] = (*this)(values.eta[i], values.target[i]);
  }
};
struct Binomial {
  static Moments from_cached(double eta, double y, double t, double log_term) {
    const double inverse = 1 / (1 + t);
    const double small_probability = t * inverse;
    const double linear = eta >= 0 ? (y == 1 ? 0. : (1 - y) * eta) : (y == 0 ? 0. : -y * eta);
    const double residual = eta >= 0 ? (1 - y) - small_probability : small_probability - y;
    return {log_term + linear, residual, small_probability * inverse};
  }
  Moments operator()(double eta, double y) const {
    // One exponential serves loss, gradient and curvature. Signed residuals
    // remain informative when the conditional mean rounds to exactly 0 or 1.
    const double t = std::exp(-std::abs(eta));
    return from_cached(eta, y, t, std::log1p(t));
  }
  void batch(ResponseBatch& values, std::size_t count) const {
    if (count != kResponseBatch || !vector_math::available()) {
      for (std::size_t i = 0; i < count; ++i)
        values.values[i] = (*this)(values.eta[i], values.target[i]);
      return;
    }
    double signed_eta[4], exponential[4], log_term[4];
    for (std::size_t i = 0; i < 4; ++i) signed_eta[i] = -std::abs(values.eta[i]);
    vector_math::exp4(signed_eta, exponential);
    vector_math::log1p4(exponential, log_term);
    for (std::size_t i = 0; i < 4; ++i)
      values.values[i] = from_cached(values.eta[i], values.target[i], exponential[i], log_term[i]);
  }
};
struct Poisson {
  Moments operator()(double eta, double y) const {
    const double mean = std::exp(eta);
    return {mean - y * eta, mean - y, mean};
  }
  void batch(ResponseBatch& values, std::size_t count) const {
    if (count != kResponseBatch || !vector_math::available()) {
      for (std::size_t i = 0; i < count; ++i)
        values.values[i] = (*this)(values.eta[i], values.target[i]);
      return;
    }
    double mean[4];
    vector_math::exp4(values.eta, mean);
    for (std::size_t i = 0; i < 4; ++i)
      values.values[i] = {mean[i] - values.target[i] * values.eta[i], mean[i] - values.target[i],
                          mean[i]};
  }
};
enum class Family { gaussian, binomial, poisson };
inline Family family(const std::string& name) {
  if (name == "gaussian") return Family::gaussian;
  if (name == "binomial") return Family::binomial;
  if (name == "poisson") return Family::poisson;
  throw std::invalid_argument("family must be gaussian, binomial or poisson");
}
inline std::string link(Family value) {
  if (value == Family::gaussian) return "identity";
  return value == Family::binomial ? "logit" : "log";
}
template <class F>
decltype(auto) with_family(Family value, F&& operation) {
  switch (value) {
    case Family::gaussian:
      return operation(Gaussian{});
    case Family::binomial:
      return operation(Binomial{});
    case Family::poisson:
      return operation(Poisson{});
  }
  throw std::invalid_argument("invalid family enum");
}
}  // namespace logitlab::detail
