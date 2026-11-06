#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace logitlab::bindings {
namespace py = pybind11;
void register_data(py::module_&);
void register_logistic(py::module_&);
void register_linear(py::module_&);
void register_glm(py::module_&);
}  // namespace logitlab::bindings
