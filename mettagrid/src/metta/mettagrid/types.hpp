#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <pybind11/pybind11.h>

namespace py = pybind11;

// PufferLib expects particular datatypes

inline py::object dtype_observations() {
  auto np = py::module_::import("numpy");
  return np.attr("dtype")(np.attr("uint8"));
}

inline py::object dtype_terminals() {
  auto np = py::module_::import("numpy");
  return np.attr("dtype")(np.attr("bool_"));
}

inline py::object dtype_truncations() {
  auto np = py::module_::import("numpy");
  return np.attr("dtype")(np.attr("bool_"));
}

inline py::object dtype_rewards() {
  auto np = py::module_::import("numpy");
  return np.attr("dtype")(np.attr("float32"));
}

inline py::object dtype_actions() {
  auto np = py::module_::import("numpy");
  return np.attr("dtype")(np.attr("int32"));
}

inline py::object dtype_masks() {
  auto np = py::module_::import("numpy");
  return np.attr("dtype")(np.attr("bool_"));
}

inline py::object dtype_success() {
  auto np = py::module_::import("numpy");
  return np.attr("dtype")(np.attr("bool_"));
}

typedef uint8_t ObservationType;
typedef bool TerminalType;
typedef bool TruncationType;
typedef float RewardType;
typedef int32_t ActionType;
typedef ActionType ActionArg;
typedef bool MaskType;
typedef bool SuccessType;

#endif  // TYPES_HPP_
