#pragma once
#include "detail/numeric.hpp"
#include "logitlab/dataset.hpp"

namespace logitlab::detail {
inline void validate_targets(const Dataset& data, bool binary) {
  require(data.labels.size() == data.rows, "label count does not match rows");
  // The immutable Python factory has already checked finiteness. The model
  // still needs to check its target domain when a Dataset is reused.
  for (double value : data.labels) {
    if (binary) require(value == 0 || value == 1, "labels must be 0 or 1");
  }
}
// Only immutable Python Dataset objects may use known_storage=true. Mutable
// public C++ Dataset entry points always perform complete storage validation.
inline const Dataset& checked_data(const Dataset& data, bool binary, bool known_storage = false) {
  if (known_storage)
    validate_targets(data, binary);
  else
    data.validate(true, binary);
  return data;
}
}  // namespace logitlab::detail
