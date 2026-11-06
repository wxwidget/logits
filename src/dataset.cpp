#include "logitlab/dataset.hpp"

#include <cmath>
#include <limits>

#include "detail/numeric.hpp"

namespace logitlab {
using detail::require;
void Dataset::validate(bool labels_required, bool binary_labels) const {
  require(rows > 0 && cols > 0, "data must have positive dimensions");
  require(rows < std::numeric_limits<std::size_t>::max(), "row count overflow");
  require(cols < std::numeric_limits<std::size_t>::max(), "column count overflow");
  if (sparse()) {
    require(offsets.size() == rows + 1 && offsets.front() == 0 && offsets.back() == values.size(),
            "invalid CSR offsets");
    require(indices.size() == values.size(), "invalid CSR indices");
    for (std::size_t i = 0; i < rows; ++i) {
      require(offsets[i] <= offsets[i + 1] && offsets[i + 1] <= values.size(), "invalid CSR row");
      for (auto k = offsets[i]; k < offsets[i + 1]; ++k) {
        require(indices[k] < cols, "CSR index out of range");
        require(k == offsets[i] || indices[k - 1] < indices[k],
                "CSR indices must be sorted and unique");
      }
    }
  } else {
    require(indices.empty(), "dense data must not contain CSR indices");
    require(rows <= std::numeric_limits<std::size_t>::max() / cols && values.size() == rows * cols,
            "invalid dense shape");
  }
  require(detail::finite(values), "features must be finite");
  if (labels_required) {
    require(labels.size() == rows, "label count does not match rows");
    for (double y : labels) {
      require(std::isfinite(y), "targets must be finite");
      if (binary_labels) require(y == 0 || y == 1, "labels must be 0 or 1");
    }
  }
}
double Dataset::dot(std::size_t row, absl::Span<const double> w) const {
  double s = 0;
  if (sparse())
    for (auto k = offsets[row]; k < offsets[row + 1]; ++k) s += values[k] * w[indices[k]];
  else
    for (std::size_t j = 0; j < cols; ++j) s += values[row * cols + j] * w[j];
  return s;
}
void Dataset::add(std::size_t row, double scale, absl::Span<double> g) const {
  if (sparse())
    for (auto k = offsets[row]; k < offsets[row + 1]; ++k) g[indices[k]] += scale * values[k];
  else
    for (std::size_t j = 0; j < cols; ++j) g[j] += scale * values[row * cols + j];
}
}  // namespace logitlab
