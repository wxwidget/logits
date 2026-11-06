#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "detail/families.hpp"
#include "detail/kernels.hpp"
#include "detail/numeric.hpp"
#include "detail/weights.hpp"
#include "logitlab/linear_regression.hpp"

using namespace logitlab;
void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
int main() {
  try {
    for (bool csr : {false, true}) {
      Dataset data;
      data.rows = 2;
      data.cols = 7;
      data.values = {1, 0, 2, -1, 0, 3, 4, 0, -2, 1, 0, 3, 0, 2};
      if (csr) {
        data.values = {1, 2, -1, 3, 4, -2, 1, 3, 2};
        data.indices = {0, 2, 3, 5, 6, 1, 2, 4, 6};
        data.offsets = {0, 5, 9};
      }
      std::vector<double> theta = {.2, -.3, .1, .4, .2, -.1, .3, .7};
      std::vector<double> gradient(8), diagonal(8), hv(8), curvature = {.2, .8};
      std::vector<double> moment_gradient(8), moment_diagonal(8), expected_gradient(8),
          expected_diagonal(8), basis(8);
      detail::with_rows(data, [&](const auto& rows) {
        for (std::size_t i = 0; i < data.rows; ++i) {
          auto row = rows[i];
          check(std::abs(row.dot(theta) - data.dot(i, theta)) < 1e-14, "row dot mismatch");
          row.add(curvature[i], absl::MakeSpan(gradient));
          row.add_diagonal(curvature[i], absl::MakeSpan(diagonal));
          row.add_moments(.3, curvature[i], absl::MakeSpan(moment_gradient),
                          absl::MakeSpan(moment_diagonal));
          data.add(i, .3, absl::MakeSpan(expected_gradient));
          for (std::size_t j = 0; j < data.cols; ++j) {
            basis[j] = 1.;
            const double feature = data.dot(i, basis);
            expected_diagonal[j] += curvature[i] * feature * feature;
            basis[j] = 0.;
          }
        }
      });
      detail::hessian_vector(data, curvature, theta, .1, true, absl::MakeSpan(hv));
      std::vector<double> expected(8);
      for (std::size_t i = 0; i < data.rows; ++i) {
        const double value = curvature[i] * (data.dot(i, theta) + theta.back());
        data.add(i, value, absl::MakeSpan(expected));
        expected.back() += value;
      }
      for (std::size_t j = 0; j < data.cols; ++j) expected[j] += .1 * theta[j];
      for (std::size_t j = 0; j < hv.size(); ++j)
        check(std::abs(hv[j] - expected[j]) < 1e-14, "Hessian-vector mismatch");
      check(gradient.back() == 0 && diagonal.back() == 0, "row touched intercept");
      for (std::size_t j = 0; j < moment_gradient.size(); ++j) {
        check(std::abs(moment_gradient[j] - expected_gradient[j]) < 1e-14,
              "joint gradient mismatch");
        check(std::abs(moment_diagonal[j] - expected_diagonal[j]) < 1e-14,
              "joint diagonal mismatch");
      }
    }
    check(detail::norm(std::vector<double>{1e-300, 0}) > 0, "norm underflow");
    check(std::isfinite(detail::norm(std::vector<double>{1e300, 1e300})), "norm overflow");
    detail::ObservationWeights a(3, std::vector<double>{1e300, 0, 1e300});
    check(a[0] == .5 && a[1] == 0 && a[2] == .5, "weight normalization");
    check(std::isnan(detail::norm(std::vector<double>{NAN})), "NaN norm hidden");
#if LOGITLAB_HAS_AVX2
    if (detail::simd::enabled()) {
      std::vector<std::int32_t> source(17);
      std::vector<std::size_t> converted(17);
      for (std::size_t j = 0; j < source.size(); ++j) source[j] = static_cast<std::int32_t>(j);
      check(detail::simd::copy_indices32(source.data(), converted.data(), source.size()),
            "valid int32 indices rejected");
      for (std::size_t j = 0; j < source.size(); ++j)
        check(converted[j] == j, "int32 widening changed an index");
      for (std::size_t j : {0, 7, 8, 16}) {
        source[j] = -1;
        check(!detail::simd::copy_indices32(source.data(), converted.data(), source.size()),
              "negative int32 index accepted");
        source[j] = static_cast<std::int32_t>(j);
      }
    }
#endif
    std::vector<double> finite_values(35, std::numeric_limits<double>::max());
    check(detail::finite(finite_values), "finite maximum rejected");
    for (std::size_t position : {0, 4, 16, 34}) {
      for (double invalid : {INFINITY, -INFINITY, NAN}) {
        finite_values[position] = invalid;
        check(!detail::finite(finite_values), "non-finite lane accepted");
      }
      finite_values[position] = 0.;
    }
    for (bool csr : {false, true}) {
      Dataset data;
      data.rows = 2;
      data.cols = 35;
      std::vector<double> theta(35, .1), expected(35), actual(35);
      if (csr) data.offsets.push_back(0);
      for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 35; ++j) {
          double value = static_cast<double>((i * 35 + j) % 7) - 3;
          if (!csr || value != 0) {
            data.values.push_back(value);
            if (csr) data.indices.push_back(j);
          }
        }
        if (csr) data.offsets.push_back(data.values.size());
      }
      detail::with_rows(data, [&](const auto& rows) {
        for (std::size_t i = 0; i < 2; ++i) {
          check(std::abs(rows[i].dot(theta) - data.dot(i, theta)) < 1e-13, "SIMD dot mismatch");
          rows[i].add(.3, absl::MakeSpan(actual));
          data.add(i, .3, absl::MakeSpan(expected));
        }
      });
      for (std::size_t j = 0; j < 35; ++j)
        check(std::abs(actual[j] - expected[j]) < 1e-13, "SIMD gradient mismatch");
    }
    Dataset malformed;
    malformed.rows = 1;
    malformed.cols = std::numeric_limits<std::size_t>::max() - 1;
    malformed.labels = {0};
    bool rejected = false;
    try {
      fit_linear(malformed);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check(rejected, "invalid storage allocated optimizer buffers before validation");
    detail::with_family(detail::Family::gaussian, [](auto response) {
      detail::ResponseBatch batch;
      for (std::size_t i = 0; i < 4; ++i) {
        batch.eta[i] = static_cast<double>(i) - 1.5;
        batch.target[i] = .2 * i;
      }
      response.batch(batch, 4);
      for (std::size_t i = 0; i < 4; ++i)
        check(std::abs(batch.values[i].loss - response(batch.eta[i], batch.target[i]).loss) < 1e-13,
              "Gaussian batch mismatch");
    });
    for (auto family : {detail::Family::binomial, detail::Family::poisson}) {
      detail::with_family(family, [](auto response) {
        detail::ResponseBatch batch;
        const double eta[4] = {-30., -.0001, .0001, 30.};
        for (std::size_t i = 0; i < 4; ++i) {
          batch.eta[i] = eta[i];
          batch.target[i] = .2 * i;
        }
        response.batch(batch, 4);
        for (std::size_t i = 0; i < 4; ++i) {
          const auto expected = response(eta[i], batch.target[i]);
          check(std::abs(batch.values[i].loss - expected.loss) <
                    1e-12 * (1 + std::abs(expected.loss)),
                "vector loss mismatch");
          check(std::abs(batch.values[i].residual - expected.residual) <
                    1e-12 * (1 + std::abs(expected.residual)),
                "vector gradient mismatch");
        }
      });
    }
    std::cout << "Shared row/numeric kernel checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
