#include "common.hpp"
#include "detail/logistic_api.hpp"
#include "logitlab/logistic_regression.hpp"
namespace logitlab::bindings {
void register_logistic(py::module_& m) {
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
      .def_readonly("seconds", &Iteration::seconds)
      .def_readonly("work", &Iteration::work);
  py::class_<Result>(m, "Result")
      .def_readonly("weights", &Result::weights)
      .def_readonly("intercept", &Result::intercept)
      .def_readonly("converged", &Result::converged)
      .def_readonly("status", &Result::status)
      .def_readonly("history", &Result::history)
      .def_readonly("work", &Result::work);
  m.def("fit", [](const Dataset& data, Options options) {
    // Snapshot options before releasing the GIL: Python may mutate its Options object.
    py::gil_scoped_release release;
    return detail::fit_logistic_validated(data, options);
  });
  m.def("predict", &detail::predict_logistic_validated, py::call_guard<py::gil_scoped_release>());
  m.def("evaluate", [](const Dataset& d, const std::vector<double>& w, const Options& o) {
    d.validate();
    std::vector<double> g(w.size());
    double f = objective(d, w, o, absl::MakeSpan(g));
    return py::make_tuple(f, g);
  });
}
}  // namespace logitlab::bindings
