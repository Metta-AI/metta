// modify_target.hpp - Simple action to modify resources of a target at a position
#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MODIFY_TARGET_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MODIFY_TARGET_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <map>
#include <string>
#include <stdexcept>

#include "actions/action_handler.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "core/types.hpp"

struct ModifyTargetConfig : public ActionConfig {
  std::map<InventoryItem, InventoryProbability> modifies;

  ModifyTargetConfig(const std::map<InventoryItem, InventoryQuantity>& required_resources = {},
                     const std::map<InventoryItem, InventoryProbability>& consumed_resources = {},
                     const std::map<InventoryItem, InventoryProbability>& modifies = {})
      : ActionConfig(required_resources, consumed_resources), modifies(modifies) {
    // Validate modifies values for finiteness and magnitude
    for (const auto& [item, value] : modifies) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("ModifyTargetConfig: modifies values must be finite");
      }
      // Check that absolute value when rounded up doesn't exceed 255
      // (since inventories are uint8_t and deltas are int16_t)
      if (std::ceil(std::abs(value)) > 255) {
        throw std::invalid_argument("ModifyTargetConfig: modifies values must have |ceil(value)| <= 255");
      }
    }
  }
};

class ModifyTarget : public ActionHandler {
public:
  explicit ModifyTarget(const ModifyTargetConfig& cfg, const std::string& name = "modify_target")
      : ActionHandler(cfg, name), _modifies(cfg.modifies) {}

  unsigned char max_arg() const override {
    // Arg encodes target position: high 4 bits = row offset, low 4 bits = col offset
    return 255;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // Decode target position from arg
    int row_offset = static_cast<int>((arg >> 4) & 0x0F) - 7;  // -7 to 8 range
    int col_offset = static_cast<int>(arg & 0x0F) - 7;          // -7 to 8 range
    
    // Compute target position in int to handle negative offsets properly
    int target_row_int = static_cast<int>(actor->location.r) + row_offset;
    int target_col_int = static_cast<int>(actor->location.c) + col_offset;
    
    // Check bounds including negative values
    if (target_row_int < 0 || target_row_int >= static_cast<int>(_grid->height) || 
        target_col_int < 0 || target_col_int >= static_cast<int>(_grid->width)) {
      return false;
    }
    
    // Safe to cast to GridCoord now
    GridCoord target_row = static_cast<GridCoord>(target_row_int);
    GridCoord target_col = static_cast<GridCoord>(target_col_int);
    
    // Find target at position (check agents first, then objects)
    GridLocation agent_loc(target_row, target_col, GridLayer::AgentLayer);
    GridObject* target = _grid->object_at(agent_loc);
    
    Agent* agent_target = nullptr;
    Converter* converter_target = nullptr;
    
    if (target != nullptr) {
      agent_target = static_cast<Agent*>(target);
    } else {
      GridLocation obj_loc(target_row, target_col, GridLayer::ObjectLayer);
      target = _grid->object_at(obj_loc);
      if (target != nullptr) {
        converter_target = dynamic_cast<Converter*>(target);
      }
    }
    
    // Only succeed if we have a valid modifiable target
    if (agent_target == nullptr && converter_target == nullptr) {
      return false;  // Target is either missing or not modifiable (e.g., Wall)
    }
    
    // Apply modifications to target
    for (const auto& [item, amount] : _modifies) {
      InventoryDelta delta = compute_probabilistic_delta(amount);
      if (delta != 0) {
        if (agent_target != nullptr) {
          agent_target->update_inventory(item, delta);
        } else if (converter_target != nullptr) {
          converter_target->update_inventory(item, delta);
        }
      }
    }
    
    return true;
  }

private:
  std::map<InventoryItem, InventoryProbability> _modifies;
};

namespace py = pybind11;

inline void bind_modify_target_config(py::module& m) {
  py::class_<ModifyTargetConfig, ActionConfig, std::shared_ptr<ModifyTargetConfig>>(m, "ModifyTargetConfig")
      .def(py::init<const std::map<InventoryItem, InventoryQuantity>&,
                    const std::map<InventoryItem, InventoryProbability>&,
                    const std::map<InventoryItem, InventoryProbability>&>(),
           py::arg("required_resources") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::map<InventoryItem, InventoryProbability>(),
           py::arg("modifies") = std::map<InventoryItem, InventoryProbability>())
      .def_readwrite("modifies", &ModifyTargetConfig::modifies);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MODIFY_TARGET_HPP_