#ifndef METTAGRID_METTAGRID_TYPES_HPP_
#define METTAGRID_METTAGRID_TYPES_HPP_

#include <pybind11/pybind11.h>

namespace py = pybind11;

// PufferLib expects particular datatypes

inline py::object np_observations_dtype() {
  auto np = py::module_::import("numpy");
  return np.attr("dtype")(np.attr("uint8"));
}

inline py::object np_terminals_dtype() {
  auto np = py::module_::import("numpy");
  return np.attr("dtype")(np.attr("bool_"));
}

inline py::object np_truncations_dtype() {
  auto np = py::module_::import("numpy");
  return np.attr("dtype")(np.attr("bool_"));
}

inline py::object np_rewards_dtype() {
  auto np = py::module_::import("numpy");
  return np.attr("dtype")(np.attr("float32"));
}

inline py::object np_actions_dtype() {
  auto np = py::module_::import("numpy");
  return np.attr("dtype")(np.attr("int32"));
}

inline py::object np_masks_dtype() {
  auto np = py::module_::import("numpy");
  return np.attr("dtype")(np.attr("bool_"));
}

inline py::object np_success_dtype() {
  auto np = py::module_::import("numpy");
  return np.attr("dtype")(np.attr("bool_"));
}

typedef uint8_t c_observations_type;
typedef bool c_terminals_type;
typedef bool c_truncations_type;
typedef float c_rewards_type;
typedef int32_t c_actions_type;
typedef bool c_masks_type;
typedef bool c_success_type;

#endif  // METTAGRID_METTAGRID_TYPES_HPP_
