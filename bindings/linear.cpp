#include "common.hpp"
#include "detail/linear_api.hpp"
#include "logitlab/linear_regression.hpp"
namespace logitlab::bindings {
void register_linear(py::module_& m) {
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
        return detail::fit_linear_validated(data, options, sample_weight);
      },
      py::arg("data"), py::arg("options"), py::arg("sample_weight") = std::vector<double>{});
  m.def("predict_linear", &detail::predict_linear_validated,
        py::call_guard<py::gil_scoped_release>());
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
}
}  // namespace logitlab::bindings
