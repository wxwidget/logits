#include "common.hpp"
#include "detail/glm_api.hpp"
#include "logitlab/generalized_linear_model.hpp"
namespace logitlab::bindings {
void register_glm(py::module_& m) {
  py::class_<GLMOptions>(m, "GLMOptions")
      .def(py::init<>())
      .def(py::init<const GLMOptions&>())
      .def_readwrite("family", &GLMOptions::family)
      .def_readwrite("link", &GLMOptions::link)
      .def_readwrite("max_iter", &GLMOptions::max_iter)
      .def_readwrite("max_cg_iter", &GLMOptions::max_cg_iter)
      .def_readwrite("l2", &GLMOptions::l2)
      .def_readwrite("tolerance", &GLMOptions::tolerance)
      .def_readwrite("fit_intercept", &GLMOptions::fit_intercept);
  py::class_<GLMIteration>(m, "GLMIteration")
      .def_readonly("iteration", &GLMIteration::iteration)
      .def_readonly("loss", &GLMIteration::loss)
      .def_readonly("gradient_norm", &GLMIteration::gradient_norm)
      .def_readonly("seconds", &GLMIteration::seconds)
      .def_readonly("work", &GLMIteration::work);
  py::class_<GLMResult>(m, "GLMResult")
      .def_readonly("weights", &GLMResult::weights)
      .def_readonly("intercept", &GLMResult::intercept)
      .def_readonly("family", &GLMResult::family)
      .def_readonly("link", &GLMResult::link)
      .def_readonly("converged", &GLMResult::converged)
      .def_readonly("status", &GLMResult::status)
      .def_readonly("history", &GLMResult::history)
      .def_readonly("work", &GLMResult::work);
  m.def(
      "fit_glm",
      [](const Dataset& data, GLMOptions options, std::vector<double> weights,
         std::vector<double> offset) {
        py::gil_scoped_release release;
        return detail::fit_glm_validated(data, options, weights, offset);
      },
      py::arg("data"), py::arg("options"), py::arg("sample_weight") = std::vector<double>{},
      py::arg("offset") = std::vector<double>{});
  m.def(
      "predict_glm",
      [](const Dataset& data, const GLMResult& model, std::vector<double> offset, bool linear) {
        py::gil_scoped_release release;
        return detail::predict_glm_validated(data, model, offset, linear);
      },
      py::arg("data"), py::arg("model"), py::arg("offset") = std::vector<double>{},
      py::arg("linear") = false);
  m.def(
      "glm_deviance",
      [](const Dataset& data, const GLMResult& model, std::vector<double> weights,
         std::vector<double> offset) {
        py::gil_scoped_release release;
        return glm_deviance(data, model, weights, offset);
      },
      py::arg("data"), py::arg("model"), py::arg("sample_weight") = std::vector<double>{},
      py::arg("offset") = std::vector<double>{});
  m.def(
      "evaluate_glm",
      [](const Dataset& data, std::vector<double> theta, GLMOptions options,
         std::vector<double> weights, std::vector<double> offset) {
        std::vector<double> gradient(theta.size());
        double loss;
        {
          py::gil_scoped_release release;
          loss = glm_objective(data, theta, options, absl::MakeSpan(gradient), weights, offset);
        }
        return py::make_tuple(loss, gradient);
      },
      py::arg("data"), py::arg("parameters"), py::arg("options"),
      py::arg("sample_weight") = std::vector<double>{}, py::arg("offset") = std::vector<double>{});
}
}  // namespace logitlab::bindings
