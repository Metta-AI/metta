#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_TYPES_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_TYPES_HPP_

#include <pybind11/pybind11.h>

namespace py = pybind11;

// PufferLib expects particular datatypes
// These data types must match PufferLib -- see pufferlib/vector.py

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

using ObservationType = uint8_t;
using TerminalType = bool;
using TruncationType = bool;
using RewardType = float;
using ActionType = int32_t;
using ActionArg = ActionType;
using MaskType = bool;
using SuccessType = bool;

using InventoryItem = uint8_t;
using InventoryQuantity = uint8_t;
using InventoryProbability = float;  // probability for fractional consumption
using InventoryDelta = int16_t;      // cover full range of allowed changes (+/-255)

using GridCoord = uint16_t;     // this sets the maximum possible map width or height
using GridObjectId = uint32_t;  // this sets the maximum tracked objects

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_CORE_TYPES_HPP_
