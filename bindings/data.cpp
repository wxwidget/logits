#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "common.hpp"
#include "detail/simd.hpp"
#include "logitlab/dataset.hpp"
namespace logitlab::bindings {
namespace {
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
template <class Integer>
std::vector<std::size_t> copy_integers(const py::array& array, const char* negative_message) {
  py::array_t<Integer, py::array::c_style | py::array::forcecast> values(array);
  if (values.ndim() != 1) throw std::invalid_argument("CSR arrays must be one dimensional");
  std::vector<std::size_t> result(values.size());
#if LOGITLAB_HAS_AVX2
  if constexpr (std::is_same_v<Integer, std::int32_t>) {
    if (detail::simd::enabled()) {
      if (!detail::simd::copy_indices32(values.data(), result.data(), result.size()))
        throw std::invalid_argument(negative_message);
      return result;
    }
  }
#endif
  for (py::ssize_t k = 0; k < values.size(); ++k) {
    const auto value = values.data()[k];
    if (value < 0) throw std::invalid_argument(negative_message);
    result[static_cast<std::size_t>(k)] = static_cast<std::size_t>(value);
  }
  return result;
}
std::vector<std::size_t> copy_indices(const py::object& object, const char* negative_message) {
  auto array = py::array::ensure(object);
  if (!array) throw std::invalid_argument("CSR arrays must be numeric arrays");
  // SciPy commonly uses int32. Convert directly into owned size_t storage,
  // rather than allocating a temporary int64 NumPy array first.
  if (array.dtype().is(py::dtype::of<std::int32_t>()))
    return copy_integers<std::int32_t>(array, negative_message);
  return copy_integers<std::int64_t>(array, negative_message);
}
void copy_labels(Dataset& data, const py::object& labels) {
  if (labels.is_none()) return;
  auto y = py::cast<Array>(labels);
  if (y.ndim() != 1) throw std::invalid_argument("y must have one dimension");
  if (y.size()) data.labels.assign(y.data(), y.data() + y.size());
}
Dataset dense(Array x, py::object labels, bool binary_labels) {
  if (x.ndim() != 2) throw std::invalid_argument("X must have two dimensions");
  Dataset d;
  d.rows = x.shape(0);
  d.cols = x.shape(1);
  if (x.size()) d.values.assign(x.data(), x.data() + x.size());
  copy_labels(d, labels);
  d.validate(!labels.is_none(), binary_labels);
  return d;
}
Dataset csr(Array data, py::object indices, py::object offsets, std::size_t rows, std::size_t cols,
            py::object labels, bool binary_labels) {
  if (data.ndim() != 1) throw std::invalid_argument("CSR arrays must be one dimensional");
  Dataset d;
  d.rows = rows;
  d.cols = cols;
  if (data.size()) d.values.assign(data.data(), data.data() + data.size());
  d.indices = copy_indices(indices, "negative CSR index");
  d.offsets = copy_indices(offsets, "negative CSR offset");
  if (d.offsets.empty()) throw std::invalid_argument("CSR offsets must not be empty");
  copy_labels(d, labels);
  d.validate(!labels.is_none(), binary_labels);
  return d;
}
}  // namespace
void register_data(py::module_& m) {
  py::class_<Dataset>(m, "Dataset")
      .def_static("dense", &dense, py::arg("X"), py::arg("y") = py::none(),
                  py::arg("binary_labels") = true)
      .def_static("csr", &csr, py::arg("data"), py::arg("indices"), py::arg("indptr"),
                  py::arg("rows"), py::arg("cols"), py::arg("y") = py::none(),
                  py::arg("binary_labels") = true);
  py::class_<Work>(m, "Work")
      .def_readonly("full_evaluations", &Work::full_evaluations)
      .def_readonly("batch_gradients", &Work::batch_gradients)
      .def_readonly("hessian_vector_products", &Work::hessian_vector_products)
      .def_readonly("row_visits", &Work::row_visits)
      .def_readonly("parameter_updates", &Work::parameter_updates);
}
}  // namespace logitlab::bindings
