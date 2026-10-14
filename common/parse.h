#pragma once
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
namespace logitlab::legacy {
inline void split(std::vector<std::string>& out, const std::string& input,
                  const std::string& delimiters) {
  out = absl::StrSplit(input, absl::ByAnyChar(delimiters), absl::SkipEmpty());
}
inline std::string is_any_of(const char* chars) { return chars; }
template <class T>
T lexical_cast(const std::string& text);
template <>
inline double lexical_cast<double>(const std::string& text) {
  double value;
  if (!absl::SimpleAtod(text, &value)) throw std::invalid_argument("invalid number: " + text);
  return value;
}
template <>
inline int lexical_cast<int>(const std::string& text) {
  int value;
  if (!absl::SimpleAtoi(text, &value)) throw std::invalid_argument("invalid integer: " + text);
  return value;
}
inline std::string feature_name(unsigned int i) { return absl::StrCat(i); }
}  // namespace logitlab::legacy
