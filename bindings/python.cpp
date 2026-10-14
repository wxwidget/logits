#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>

#include "logitlab/logistic_regression.hpp"
namespace py = pybind11;
using namespace logitlab;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
Dataset dense(Array x, py::object labels) {
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
  d.validate(!labels.is_none());
  return d;
}
Dataset csr(Array data,
            py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> indices,
            py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> offsets,
            std::size_t rows, std::size_t cols, py::object labels) {
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
  d.validate(!labels.is_none());
  return d;
}
PYBIND11_MODULE(_core, m) {
  py::class_<Dataset>(m, "Dataset")
      .def_static("dense", &dense, py::arg("X"), py::arg("y") = py::none())
      .def_static("csr", &csr, py::arg("data"), py::arg("indices"), py::arg("indptr"),
                  py::arg("rows"), py::arg("cols"), py::arg("y") = py::none());
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
  py::class_<Iteration>(m, "Iteration")
      .def_readonly("epoch", &Iteration::epoch)
      .def_readonly("loss", &Iteration::loss)
      .def_readonly("gradient_norm", &Iteration::gradient_norm)
      .def_readonly("seconds", &Iteration::seconds);
  py::class_<Result>(m, "Result")
      .def_readonly("weights", &Result::weights)
      .def_readonly("intercept", &Result::intercept)
      .def_readonly("converged", &Result::converged)
      .def_readonly("status", &Result::status)
      .def_readonly("history", &Result::history);
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
