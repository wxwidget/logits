#include "common.hpp"
#include "detail/simd.hpp"
#include "detail/vector_math.hpp"
PYBIND11_MODULE(_core, module) {
  // Shared types must be registered before model-specific signatures.
  logitlab::bindings::register_data(module);
  logitlab::bindings::register_logistic(module);
  logitlab::bindings::register_linear(module);
  logitlab::bindings::register_glm(module);
  module.def("_runtime_features", []() {
    pybind11::dict features;
    features["simd"] = logitlab::detail::simd::enabled() ? "avx2_fma" : "scalar";
    features["vector_math"] =
        logitlab::detail::vector_math::available() ? "glibc_libmvec" : "scalar_stdlib";
    return features;
  });
}
