#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>

#include "logitlab/linear_regression.hpp"
#include "logitlab/logistic_regression.hpp"
namespace py = pybind11;
using namespace logitlab;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
Dataset dense(Array x, py::object labels, bool binary_labels) {
  if (x.ndim() != 2) throw std::invalid_argument("X must have two dimensions");
  Dataset d;
  d.rows = x.shape(0);
  d.cols = x.shape(1);
  if (x.size()) d.values.assign(x.data(), x.data() + x.size());
  if (!labels.is_none()) {
    auto y = py::cast<Array>(labels);
    if (y.ndim() != 1) throw std::invalid_argument("y must have one dimension");
    if (y.size()) d.labels.assign(y.data(), y.data() + y.size());
  }
  d.validate(!labels.is_none(), binary_labels);
  return d;
}
Dataset csr(Array data,
            py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> indices,
            py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> offsets,
            std::size_t rows, std::size_t cols, py::object labels, bool binary_labels) {
  if (data.ndim() != 1 || indices.ndim() != 1 || offsets.ndim() != 1)
    throw std::invalid_argument("CSR arrays must be one dimensional");
  Dataset d;
  d.rows = rows;
  d.cols = cols;
  if (data.size()) d.values.assign(data.data(), data.data() + data.size());
  for (py::ssize_t k = 0; k < indices.size(); ++k) {
    auto i = indices.data()[k];
    if (i < 0) throw std::invalid_argument("negative CSR index");
    d.indices.push_back(i);
  }
  for (py::ssize_t k = 0; k < offsets.size(); ++k) {
    auto i = offsets.data()[k];
    if (i < 0) throw std::invalid_argument("negative CSR offset");
    d.offsets.push_back(i);
  }
  if (d.offsets.empty()) throw std::invalid_argument("CSR offsets must not be empty");
  if (!labels.is_none()) {
    auto y = py::cast<Array>(labels);
    if (y.ndim() != 1) throw std::invalid_argument("y must have one dimension");
    if (y.size()) d.labels.assign(y.data(), y.data() + y.size());
  }
  d.validate(!labels.is_none(), binary_labels);
  return d;
}
PYBIND11_MODULE(_core, m) {
  py::class_<Dataset>(m, "Dataset")
      .def_static("dense", &dense, py::arg("X"), py::arg("y") = py::none(),
                  py::arg("binary_labels") = true)
      .def_static("csr", &csr, py::arg("data"), py::arg("indices"), py::arg("indptr"),
                  py::arg("rows"), py::arg("cols"), py::arg("y") = py::none(),
                  py::arg("binary_labels") = true);
  py::class_<Options>(m, "Options")
      .def(py::init<>())
      .def_readwrite("solver", &Options::solver)
      .def_readwrite("max_epochs", &Options::max_epochs)
      .def_readwrite("batch_size", &Options::batch_size)
      .def_readwrite("history_size", &Options::history_size)
      .def_readwrite("learning_rate", &Options::learning_rate)
      .def_readwrite("l2", &Options::l2)
      .def_readwrite("tolerance", &Options::tolerance)
      .def_readwrite("seed", &Options::seed)
      .def_readwrite("fit_intercept", &Options::fit_intercept);
  py::class_<Work>(m, "Work")
      .def_readonly("full_evaluations", &Work::full_evaluations)
      .def_readonly("batch_gradients", &Work::batch_gradients)
      .def_readonly("hessian_vector_products", &Work::hessian_vector_products)
      .def_readonly("row_visits", &Work::row_visits)
      .def_readonly("parameter_updates", &Work::parameter_updates);
  py::class_<Iteration>(m, "Iteration")
      .def_readonly("epoch", &Iteration::epoch)
      .def_readonly("loss", &Iteration::loss)
      .def_readonly("gradient_norm", &Iteration::gradient_norm)
      .def_readonly("seconds", &Iteration::seconds)
      .def_readonly("work", &Iteration::work);
  py::class_<Result>(m, "Result")
      .def_readonly("weights", &Result::weights)
      .def_readonly("intercept", &Result::intercept)
      .def_readonly("converged", &Result::converged)
      .def_readonly("status", &Result::status)
      .def_readonly("history", &Result::history)
      .def_readonly("work", &Result::work);
  py::class_<LinearOptions>(m, "LinearOptions")
      .def(py::init<>())
      .def(py::init<const LinearOptions&>())
      .def_readwrite("solver", &LinearOptions::solver)
      .def_readwrite("max_iter", &LinearOptions::max_iter)
      .def_readwrite("l1", &LinearOptions::l1)
      .def_readwrite("l2", &LinearOptions::l2)
      .def_readwrite("tolerance", &LinearOptions::tolerance)
      .def_readwrite("learning_rate", &LinearOptions::learning_rate)
      .def_readwrite("fit_intercept", &LinearOptions::fit_intercept)
      .def_readwrite("positive", &LinearOptions::positive)
      .def_readwrite("lower_bounds", &LinearOptions::lower_bounds)
      .def_readwrite("upper_bounds", &LinearOptions::upper_bounds)
      .def_readwrite("intercept_lower", &LinearOptions::intercept_lower)
      .def_readwrite("intercept_upper", &LinearOptions::intercept_upper);
  py::class_<LinearIteration>(m, "LinearIteration")
      .def_readonly("iteration", &LinearIteration::iteration)
      .def_readonly("loss", &LinearIteration::loss)
      .def_readonly("stationarity_norm", &LinearIteration::stationarity_norm)
      .def_readonly("seconds", &LinearIteration::seconds)
      .def_readonly("work", &LinearIteration::work);
  py::class_<LinearResult>(m, "LinearResult")
      .def_readonly("weights", &LinearResult::weights)
      .def_readonly("intercept", &LinearResult::intercept)
      .def_readonly("converged", &LinearResult::converged)
      .def_readonly("status", &LinearResult::status)
      .def_readonly("solver", &LinearResult::solver)
      .def_readonly("history", &LinearResult::history)
      .def_readonly("work", &LinearResult::work);
  m.def(
      "fit_linear",
      [](const Dataset& data, LinearOptions options, std::vector<double> sample_weight) {
        py::gil_scoped_release release;
        return fit_linear(data, options, sample_weight);
      },
      py::arg("data"), py::arg("options"), py::arg("sample_weight") = std::vector<double>{});
  m.def("predict_linear", &predict_linear, py::call_guard<py::gil_scoped_release>());
  m.def(
      "evaluate_linear",
      [](const Dataset& data, const std::vector<double>& w, const LinearOptions& options,
         const std::vector<double>& sample_weight) {
        std::vector<double> gradient(w.size());
        double loss = linear_objective(data, w, options, absl::MakeSpan(gradient), sample_weight);
        return py::make_tuple(loss, gradient);
      },
      py::arg("data"), py::arg("parameters"), py::arg("options"),
      py::arg("sample_weight") = std::vector<double>{});
  m.def("fit", [](const Dataset& data, Options options) {
    // Snapshot options before releasing the GIL: Python may mutate its Options object.
    py::gil_scoped_release release;
    return fit(data, options);
  });
  m.def("predict", &predict, py::call_guard<py::gil_scoped_release>());
  m.def("evaluate", [](const Dataset& d, const std::vector<double>& w, const Options& o) {
    d.validate();
    std::vector<double> g(w.size());
    double f = objective(d, w, o, absl::MakeSpan(g));
    return py::make_tuple(f, g);
  });
}
