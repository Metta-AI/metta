#include "bindings/mettagrid_c.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "handler/handler_bindings.hpp"
#include "actions/attack.hpp"
#include "actions/change_vibe.hpp"
#include "actions/move_config.hpp"
#include "actions/transfer.hpp"
#include "core/aoe_bindings.hpp"
#include "core/grid.hpp"
#include "objects/agent.hpp"
#include "objects/assembler.hpp"
#include "objects/chest.hpp"
#include "objects/collective.hpp"
#include "objects/protocol.hpp"
#include "objects/wall.hpp"
#include "systems/clipper_config.hpp"
#include "systems/packed_coordinate.hpp"
#include "systems/stats_tracker.hpp"

namespace py = pybind11;

py::dict MettaGrid::grid_objects(int min_row, int max_row, int min_col, int max_col, const py::list& ignore_types) {
  py::dict objects;

  // Determine if bounding box filtering is enabled
  bool use_bounds = (min_row >= 0 && max_row >= 0 && min_col >= 0 && max_col >= 0);

  // Convert ignore_types list (type names) to type IDs for O(1) integer comparison
  std::unordered_set<TypeId> ignore_type_ids;
  for (const auto& item : ignore_types) {
    std::string type_name = item.cast<std::string>();
    // Find the type_id for this type_name
    for (size_t type_id = 0; type_id < object_type_names.size(); ++type_id) {
      if (object_type_names[type_id] == type_name) {
        ignore_type_ids.insert(static_cast<TypeId>(type_id));
        break;
      }
    }
  }
  bool use_type_filter = !ignore_type_ids.empty();

  for (unsigned int obj_id = 1; obj_id < _grid->objects.size(); obj_id++) {
    auto obj = _grid->object(obj_id);
    if (!obj) continue;

    // Filter by type_id if specified (fast integer comparison)
    if (use_type_filter) {
      if (ignore_type_ids.find(obj->type_id) != ignore_type_ids.end()) {
        continue;
      }
    }

    // Filter by bounding box if specified
    if (use_bounds) {
      if (obj->location.r < min_row || obj->location.r >= max_row || obj->location.c < min_col ||
          obj->location.c >= max_col) {
        continue;
      }
    }

    py::dict obj_dict;
    obj_dict["id"] = obj_id;
    obj_dict["type_name"] = object_type_names[obj->type_id];
    // Location here is defined as XYZ coordinates specifically to be used by MettaScope.
    // We define that for location: x is column, y is row. Currently, no z for grid objects.
    // Note: it might be different for matrix computations.
    obj_dict["location"] = py::make_tuple(obj->location.c, obj->location.r);

    obj_dict["r"] = obj->location.r;          // To remove
    obj_dict["c"] = obj->location.c;          // To remove

    // Inject observation features
    auto features = obj->obs_features();
    for (const auto& feature : features) {
      auto feature_name_it = feature_id_to_name.find(feature.feature_id);
      if (feature_name_it != feature_id_to_name.end()) {
        obj_dict[py::str(feature_name_it->second)] = feature.value;
      }
    }

    if (auto* has_inventory = dynamic_cast<HasInventory*>(obj)) {
      py::dict inventory_dict;
      for (const auto& [resource, quantity] : has_inventory->inventory.get()) {
        inventory_dict[py::int_(resource)] = quantity;
      }
      obj_dict["inventory"] = inventory_dict;
    }

    // Add collective_id for alignable objects
    Collective* collective = obj->getCollective();
    if (collective != nullptr) {
      // Find the index of this collective in _collectives
      for (size_t i = 0; i < _collectives.size(); i++) {
        if (_collectives[i].get() == collective) {
          obj_dict["collective_id"] = static_cast<int>(i);
          break;
        }
      }
    }

    // Inject agent-specific info
    if (auto* agent = dynamic_cast<Agent*>(obj)) {
      obj_dict["group_id"] = agent->group;
      obj_dict["group_name"] = agent->group_name;
      obj_dict["is_frozen"] = !!agent->frozen;
      obj_dict["freeze_remaining"] = agent->frozen;
      obj_dict["freeze_duration"] = agent->freeze_duration;
      obj_dict["vibe"] = agent->vibe;
      obj_dict["agent_id"] = agent->agent_id;
      obj_dict["current_stat_reward"] = agent->current_stat_reward;
      obj_dict["steps_without_motion"] = agent->steps_without_motion;

      // We made resource limits more complicated than this, and need to review how to expose them.
      // py::dict resource_limits_dict;
      // for (const auto& [resource, quantity] : agent->inventory.limits) {
      //   resource_limits_dict[py::int_(resource)] = quantity;
      // }
      // obj_dict["resource_limits"] = resource_limits_dict;
    }

    // Add assembler-specific info
    if (auto* assembler = dynamic_cast<Assembler*>(obj)) {
      obj_dict["cooldown_remaining"] = assembler->cooldown_remaining();
      obj_dict["cooldown_duration"] = assembler->cooldown_duration;
      obj_dict["is_clipped"] = assembler->is_clipped;
      obj_dict["is_clip_immune"] = assembler->clip_immune;
      obj_dict["uses_count"] = assembler->uses_count;
      obj_dict["max_uses"] = assembler->max_uses;
      obj_dict["allow_partial_usage"] = assembler->allow_partial_usage;

      // Add current protocol ID (pattern byte)
      obj_dict["current_protocol_id"] = static_cast<int>(assembler->get_local_vibe());

      // Add current protocol information
      const Protocol* current_protocol = assembler->get_current_protocol();
      if (current_protocol) {
        py::dict input_resources_dict;
        for (const auto& [resource, quantity] : current_protocol->input_resources) {
          input_resources_dict[py::int_(resource)] = quantity;
        }
        obj_dict["current_protocol_inputs"] = input_resources_dict;

        py::dict output_resources_dict;
        for (const auto& [resource, quantity] : current_protocol->output_resources) {
          output_resources_dict[py::int_(resource)] = quantity;
        }
        obj_dict["current_protocol_outputs"] = output_resources_dict;
        obj_dict["current_protocol_cooldown"] = current_protocol->cooldown;
      }

      // Add all protocols information
      const std::unordered_map<GroupVibe, vector<std::shared_ptr<Protocol>>>& active_protocols =
          assembler->is_clipped ? assembler->unclip_protocols : assembler->protocols;
      py::list protocols_list;

      for (const auto& [vibe, protocols] : active_protocols) {
        for (const auto& protocol : protocols) {
          py::dict protocol_dict;

          py::dict input_resources_dict;
          for (const auto& [resource, quantity] : protocol->input_resources) {
            input_resources_dict[py::int_(resource)] = quantity;
          }
          protocol_dict["inputs"] = input_resources_dict;

          py::dict output_resources_dict;
          for (const auto& [resource, quantity] : protocol->output_resources) {
            output_resources_dict[py::int_(resource)] = quantity;
          }
          protocol_dict["outputs"] = output_resources_dict;
          protocol_dict["cooldown"] = protocol->cooldown;
          protocol_dict["min_agents"] = protocol->min_agents;
          protocol_dict["vibes"] = protocol->vibes;
          protocols_list.append(protocol_dict);
        }
      }
      obj_dict["protocols"] = protocols_list;
    }

    // Add chest-specific info
    if (auto* chest = dynamic_cast<Chest*>(obj)) {
      // Convert vibe_transfers map to dict
      py::dict vibe_transfers_dict;
      for (const auto& [vibe, resource_deltas] : chest->vibe_transfers) {
        py::dict resource_dict;
        for (const auto& [resource, delta] : resource_deltas) {
          resource_dict[py::int_(resource)] = delta;
        }
        vibe_transfers_dict[py::int_(vibe)] = resource_dict;
      }
      obj_dict["vibe_transfers"] = vibe_transfers_dict;
    }

    objects[py::int_(obj_id)] = obj_dict;
  }

  return objects;
}

GridCoord MettaGrid::map_width() {
  return _grid->width;
}

GridCoord MettaGrid::map_height() {
  return _grid->height;
}

py::array_t<float> MettaGrid::get_episode_rewards() {
  return _episode_rewards;
}

py::array_t<ActionType> MettaGrid::actions() {
  return _actions;
}

py::dict MettaGrid::get_episode_stats() {
  // Returns a dictionary with the following structure:
  // {
  //   "game": dict[str, float],  // Global game statistics
  //   "agent": list[dict[str, float]],  // Per-agent statistics
  // }

  py::dict stats;
  stats["game"] = py::cast(_stats->to_dict());

  py::list agent_stats;
  for (const auto& agent : _agents) {
    agent_stats.append(py::cast(agent->stats.to_dict()));
  }
  stats["agent"] = agent_stats;

  return stats;
}

py::list MettaGrid::action_success_py() {
  return py::cast(_action_success);
}

py::dict MettaGrid::get_collective_inventories() {
  // Returns a dictionary mapping collective names to their inventories
  // { "collective_name": { "resource_name": quantity, ... }, ... }
  py::dict result;
  for (const auto& collective : _collectives) {
    py::dict inventory_dict;
    for (const auto& [resource_id, quantity] : collective->inventory.get()) {
      // Use resource name instead of ID for mettascope compatibility
      if (resource_id < resource_names.size()) {
        inventory_dict[py::str(resource_names[resource_id])] = quantity;
      }
    }
    result[py::str(collective->name)] = inventory_dict;
  }
  return result;
}

py::none MettaGrid::set_inventory(GridObjectId agent_id,
                                  const std::unordered_map<InventoryItem, InventoryQuantity>& inventory) {
  if (agent_id < _agents.size()) {
    this->_agents[agent_id]->set_inventory(inventory);
  }
  return py::none();
}

py::array_t<ObservationType> MettaGrid::observations() {
  return _observations;
}

py::array_t<TerminalType> MettaGrid::terminals() {
  return _terminals;
}

py::array_t<TruncationType> MettaGrid::truncations() {
  return _truncations;
}

py::array_t<RewardType> MettaGrid::rewards() {
  return _rewards;
}

py::array_t<MaskType> MettaGrid::masks() {
  // Return action masks - currently not computed, return empty array
  // TODO: Implement proper action masking if needed
  auto result = py::array_t<MaskType>(
      {static_cast<py::ssize_t>(_agents.size()), static_cast<py::ssize_t>(_action_handlers.size())});
  auto r = result.template mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < r.shape(0); i++) {
    for (py::ssize_t j = 0; j < r.shape(1); j++) {
      r(i, j) = 1;  // All actions available by default
    }
  }
  return result;
}

// Pybind11 module definition
PYBIND11_MODULE(mettagrid_c, m) {
  m.doc() = "MettaGrid environment";  // optional module docstring

  PackedCoordinate::bind_packed_coordinate(m);

  // Bind Protocol near its definition
  bind_protocol(m);

  // MettaGrid class bindings
  py::class_<MettaGrid>(m, "MettaGrid")
      .def(py::init<const GameConfig&, const py::list&, unsigned int>())
      .def("step", &MettaGrid::step)
      .def("set_buffers",
           &MettaGrid::set_buffers,
           py::arg("observations").noconvert(),
           py::arg("terminals").noconvert(),
           py::arg("truncations").noconvert(),
           py::arg("rewards").noconvert(),
           py::arg("actions").noconvert())
      .def("grid_objects",
           &MettaGrid::grid_objects,
           py::arg("min_row") = -1,
           py::arg("max_row") = -1,
           py::arg("min_col") = -1,
           py::arg("max_col") = -1,
           py::arg("ignore_types") = py::list())
      .def("observations", &MettaGrid::observations)
      .def("terminals", &MettaGrid::terminals)
      .def("truncations", &MettaGrid::truncations)
      .def("rewards", &MettaGrid::rewards)
      .def("masks", &MettaGrid::masks)
      .def("actions", &MettaGrid::actions)
      .def_property_readonly("map_width", &MettaGrid::map_width)
      .def_property_readonly("map_height", &MettaGrid::map_height)
      .def("get_episode_rewards", &MettaGrid::get_episode_rewards)
      .def("get_episode_stats", &MettaGrid::get_episode_stats)
      .def("action_success", &MettaGrid::action_success_py)
      .def_readonly("obs_width", &MettaGrid::obs_width)
      .def_readonly("obs_height", &MettaGrid::obs_height)
      .def_readonly("max_steps", &MettaGrid::max_steps)
      .def_readonly("current_step", &MettaGrid::current_step)
      .def_readonly("object_type_names", &MettaGrid::object_type_names)
      .def_readonly("resource_names", &MettaGrid::resource_names)
      .def("set_inventory", &MettaGrid::set_inventory, py::arg("agent_id"), py::arg("inventory"))
      .def("get_collective_inventories", &MettaGrid::get_collective_inventories);

  // Expose this so we can cast python WallConfig / AgentConfig to a common GridConfig cpp object.
  py::class_<GridObjectConfig, std::shared_ptr<GridObjectConfig>>(m, "GridObjectConfig");

  bind_wall_config(m);

  // ##MettaGridConfig
  // We expose these as much as we can to Python. Defining the initializer (and the object's constructor) means
  // we can create these in Python as AgentConfig(**agent_config_dict). And then we expose the fields individually.
  // This is verbose! But it seems like it's the best way to do it.
  //
  // We use shared_ptr because we expect to effectively have multiple python objects wrapping the same C++ object.
  // This comes from us creating (e.g.) various config objects, and then storing them in GameConfig's maps.
  // We're, like 80% sure on this reasoning.

  bind_inventory_config(m);
  bind_collective_config(m);
  bind_agent_config(m);
  bind_assembler_config(m);
  bind_chest_config(m);
  bind_action_config(m);
  bind_attack_action_config(m);
  bind_vibe_transfer_effect(m);
  bind_transfer_action_config(m);
  bind_change_vibe_action_config(m);
  bind_move_action_config(m);
  bind_global_obs_config(m);
  bind_clipper_config(m);
  bind_game_config(m);

  // AOE system bindings
  bind_aoe_config(m);

  // Activation handler bindings
  bind_handler_config(m);

  // Export data types from types.hpp
  m.attr("dtype_observations") = dtype_observations();
  m.attr("dtype_terminals") = dtype_terminals();
  m.attr("dtype_truncations") = dtype_truncations();
  m.attr("dtype_rewards") = dtype_rewards();
  m.attr("dtype_actions") = dtype_actions();
  m.attr("dtype_masks") = dtype_masks();
  m.attr("dtype_success") = dtype_success();

#ifdef METTA_WITH_RAYLIB
  py::class_<HermesPy>(m, "Hermes")
      .def(py::init<>())
      .def("update", &HermesPy::update, py::arg("env"))
      .def("render", &HermesPy::render);
#endif
}
