#include "bindings/mettagrid_c.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "actions/action_handler.hpp"
#include "actions/attack.hpp"
#include "actions/change_glyph.hpp"
#include "actions/resource_mod.hpp"
#include "config/mettagrid_config.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "env/buffer_views.hpp"
#include "env/mettagrid_engine.hpp"
#include "objects/agent.hpp"
#include "objects/assembler.hpp"
#include "objects/chest.hpp"
#include "objects/constants.hpp"
#include "objects/converter.hpp"
#include "objects/has_inventory.hpp"
#include "objects/recipe.hpp"
#include "objects/wall.hpp"
#include "renderer/hermes.hpp"
#include "systems/clipper_config.hpp"
#include "systems/observation_encoder.hpp"
#include "systems/packed_coordinate.hpp"
#include "systems/stats_tracker.hpp"

namespace py = pybind11;
using mettagrid::env::ActionMatrixView;
using mettagrid::env::BufferSet;
using mettagrid::env::ObservationBufferView;

namespace {

std::vector<std::vector<std::string>> ConvertMap(const py::list& map) {
  std::vector<std::vector<std::string>> result;
  result.reserve(static_cast<size_t>(py::len(map)));
  for (const auto& row_obj : map) {
    py::list row_list = row_obj.cast<py::list>();
    std::vector<std::string> row;
    row.reserve(static_cast<size_t>(py::len(row_list)));
    for (const auto& cell_obj : row_list) {
      row.push_back(cell_obj.cast<std::string>());
    }
    result.push_back(std::move(row));
  }
  return result;
}

BufferSet MakeBufferSet(py::array_t<uint8_t>& observations,
                        py::array_t<bool>& terminals,
                        py::array_t<bool>& truncations,
                        py::array_t<float>& rewards,
                        py::array_t<float>& episode_rewards) {
  auto obs_info = observations.request();
  if (obs_info.ndim != 3) {
    throw std::runtime_error("observations must be a 3D array");
  }

  auto term_info = terminals.request();
  auto trunc_info = truncations.request();
  auto rewards_info = rewards.request();
  auto episode_info = episode_rewards.request();

  BufferSet buffers;
  buffers.observations = ObservationBufferView{static_cast<ObservationType*>(obs_info.ptr),
                                               static_cast<size_t>(obs_info.shape[0]),
                                               static_cast<size_t>(obs_info.shape[1]),
                                               static_cast<size_t>(obs_info.shape[2])};
  buffers.terminals = mettagrid::env::ArrayView<TerminalType>{
      static_cast<TerminalType*>(term_info.ptr), static_cast<size_t>(term_info.shape[0])};
  buffers.truncations = mettagrid::env::ArrayView<TruncationType>{
      static_cast<TruncationType*>(trunc_info.ptr), static_cast<size_t>(trunc_info.shape[0])};
  buffers.rewards = mettagrid::env::ArrayView<RewardType>{
      static_cast<RewardType*>(rewards_info.ptr), static_cast<size_t>(rewards_info.shape[0])};
  buffers.episode_rewards = mettagrid::env::ArrayView<RewardType>{
      static_cast<RewardType*>(episode_info.ptr), static_cast<size_t>(episode_info.shape[0])};
  return buffers;
}

ActionMatrixView MakeActionMatrixView(py::array_t<ActionType, py::array::c_style>& actions) {
  auto info = actions.request();
  if (info.ndim != 2 || info.shape[1] != 2) {
    throw std::runtime_error("actions array must have shape [num_agents, 2]");
  }
  return ActionMatrixView{static_cast<ActionType*>(info.ptr), static_cast<size_t>(info.shape[0])};
}
}  // namespace

MettaGrid::MettaGrid(const GameConfig& game_config, const py::list map, unsigned int seed)
    : obs_width(game_config.obs_width),
      obs_height(game_config.obs_height),
      current_step(0),
      max_steps(game_config.max_steps),
      episode_truncates(game_config.episode_truncates),
      resource_names(game_config.resource_names),
      object_type_names(),
      initial_grid_hash(0),
      _actions(),
      _observations(),
      _terminals(),
      _truncations(),
      _rewards(),
      _episode_rewards(),
      _buffer_views(),
      _engine(std::make_unique<mettagrid::env::MettaGridEngine>(game_config, ConvertMap(map), seed)),
      _tag_id_map(game_config.tag_id_map) {
  object_type_names = _engine->object_type_names;
  resource_names = _engine->resource_names;
  initial_grid_hash = _engine->initial_grid_hash;

  const size_t num_agents = _engine->num_agents();
  const ssize_t tokens = static_cast<ssize_t>(game_config.num_observation_tokens);

  _observations =
      py::array_t<uint8_t, py::array::c_style>({static_cast<ssize_t>(num_agents), tokens, static_cast<ssize_t>(3)});
  _terminals = py::array_t<bool, py::array::c_style>({static_cast<ssize_t>(num_agents)}, {sizeof(TerminalType)});
  _truncations = py::array_t<bool, py::array::c_style>({static_cast<ssize_t>(num_agents)}, {sizeof(TruncationType)});
  _rewards = py::array_t<float, py::array::c_style>({static_cast<ssize_t>(num_agents)}, {sizeof(RewardType)});
  _episode_rewards =
      py::array_t<float, py::array::c_style>({static_cast<ssize_t>(num_agents)}, {sizeof(RewardType)});

  _actions =
      py::array_t<ActionType, py::array::c_style>({static_cast<ssize_t>(num_agents), static_cast<ssize_t>(2)});
  std::fill_n(static_cast<ActionType*>(_actions.request().ptr), static_cast<ssize_t>(num_agents) * 2, 0);

  refresh_buffer_views();
  _engine->set_buffers(_buffer_views);
  _engine->reset();
  current_step = _engine->current_step;
}

MettaGrid::~MettaGrid() = default;

void MettaGrid::refresh_buffer_views() {
  _buffer_views = MakeBufferSet(_observations, _terminals, _truncations, _rewards, _episode_rewards);
}

py::tuple MettaGrid::reset() {
  _engine->reset();
  current_step = _engine->current_step;
  return py::make_tuple(_observations, py::dict());
}

void MettaGrid::validate_buffers() {
  _engine->validate_buffers();
}

void MettaGrid::set_buffers(const py::array_t<uint8_t, py::array::c_style>& observations,
                            const py::array_t<bool, py::array::c_style>& terminals,
                            const py::array_t<bool, py::array::c_style>& truncations,
                            const py::array_t<float, py::array::c_style>& rewards) {
  _observations = observations;
  _terminals = terminals;
  _truncations = truncations;
  _rewards = rewards;
  _episode_rewards = py::array_t<float, py::array::c_style>(
      {static_cast<ssize_t>(_rewards.shape(0))}, {sizeof(float)});
  refresh_buffer_views();
  _engine->set_buffers(_buffer_views);
}

py::tuple MettaGrid::step(const py::array_t<ActionType, py::array::c_style> actions) {
  const size_t num_agents = _engine->num_agents();
  auto info = actions.request();
  py::array_t<ActionType, py::array::c_style> action_matrix;

  const auto& flat_map = _engine->flat_action_map();

  auto assign_flat = [&](auto& view, size_t agent_idx, ActionType flat_index) {
    if (flat_index < 0 || static_cast<size_t>(flat_index) >= flat_map.size()) {
      view(agent_idx, 0) = -1;
      view(agent_idx, 1) = 0;
      return;
    }
    const auto& mapping = flat_map[static_cast<size_t>(flat_index)];
    view(agent_idx, 0) = mapping.first;
    view(agent_idx, 1) = mapping.second;
  };

  if (info.ndim == 1) {
    if (info.shape[0] != static_cast<ssize_t>(num_agents)) {
      throw std::runtime_error("actions has the wrong shape");
    }
    auto view = actions.unchecked<1>();
    action_matrix = py::array_t<ActionType, py::array::c_style>(
        {static_cast<ssize_t>(num_agents), static_cast<ssize_t>(2)});
    auto converted_view = action_matrix.mutable_unchecked<2>();
    for (size_t agent_idx = 0; agent_idx < num_agents; ++agent_idx) {
      assign_flat(converted_view, agent_idx, view(agent_idx));
    }
  } else if (info.ndim == 2) {
    if (info.shape[0] != static_cast<ssize_t>(num_agents)) {
      throw std::runtime_error("actions has the wrong shape");
    }
    if (info.shape[1] == 1) {
      auto view = actions.unchecked<2>();
      action_matrix = py::array_t<ActionType, py::array::c_style>(
          {static_cast<ssize_t>(num_agents), static_cast<ssize_t>(2)});
      auto converted_view = action_matrix.mutable_unchecked<2>();
      for (size_t agent_idx = 0; agent_idx < num_agents; ++agent_idx) {
        assign_flat(converted_view, agent_idx, view(agent_idx, 0));
      }
    } else if (info.shape[1] == 2) {
      action_matrix = actions;
    } else {
      throw std::runtime_error("actions has the wrong shape");
    }
  } else {
    throw std::runtime_error("actions has the wrong shape");
  }

  _actions = action_matrix;
  auto matrix_view = MakeActionMatrixView(_actions);
  _engine->step(matrix_view);
  current_step = _engine->current_step;

  return py::make_tuple(_observations, _rewards, _terminals, _truncations, py::dict());
}

py::dict MettaGrid::grid_objects(int min_row,
                                 int max_row,
                                 int min_col,
                                 int max_col,
                                 const py::list& ignore_types) {
  py::dict objects;

  bool use_bounds = (min_row >= 0 && max_row >= 0 && min_col >= 0 && max_col >= 0);

  std::unordered_set<TypeId> ignore_type_ids;
  for (const auto& item : ignore_types) {
    std::string type_name = item.cast<std::string>();
    for (size_t type_id = 0; type_id < object_type_names.size(); ++type_id) {
      if (object_type_names[type_id] == type_name) {
        ignore_type_ids.insert(static_cast<TypeId>(type_id));
        break;
      }
    }
  }
  bool use_type_filter = !ignore_type_ids.empty();

  const auto& grid = _engine->grid();
  const auto& encoder = _engine->observation_encoder();

  for (unsigned int obj_id = 1; obj_id < grid.objects.size(); obj_id++) {
    auto obj = grid.object(obj_id);
    if (!obj) {
      continue;
    }

    if (use_type_filter && ignore_type_ids.contains(obj->type_id)) {
      continue;
    }

    if (use_bounds) {
      if (obj->location.r < min_row || obj->location.r >= max_row || obj->location.c < min_col ||
          obj->location.c >= max_col) {
        continue;
      }
    }

    py::dict obj_dict;
    obj_dict["id"] = obj_id;
    obj_dict["type"] = obj->type_id;
    obj_dict["type_id"] = obj->type_id;
    obj_dict["type_name"] = object_type_names[obj->type_id];
    obj_dict["location"] = py::make_tuple(obj->location.c, obj->location.r, obj->location.layer);
    obj_dict["is_swappable"] = obj->swappable();
    obj_dict["r"] = obj->location.r;
    obj_dict["c"] = obj->location.c;
    obj_dict["layer"] = obj->location.layer;

    auto features = obj->obs_features();
    for (const auto& feature : features) {
      const auto& names = encoder.feature_names();
      auto it = names.find(feature.feature_id);
      if (it != names.end()) {
        obj_dict[py::str(it->second)] = feature.value;
      }
    }

    if (auto* has_inventory = dynamic_cast<HasInventory*>(obj)) {
      py::dict inventory_dict;
      for (const auto& [resource, quantity] : has_inventory->inventory.get()) {
        inventory_dict[py::int_(resource)] = quantity;
      }
      obj_dict["inventory"] = inventory_dict;
    }

    if (auto* agent_ptr = dynamic_cast<Agent*>(obj)) {
      obj_dict["orientation"] = static_cast<int>(agent_ptr->orientation);
      obj_dict["group_id"] = agent_ptr->group;
      obj_dict["group_name"] = agent_ptr->group_name;
      obj_dict["is_frozen"] = !!agent_ptr->frozen;
      obj_dict["freeze_remaining"] = agent_ptr->frozen;
      obj_dict["freeze_duration"] = agent_ptr->freeze_duration;
      obj_dict["glyph"] = agent_ptr->glyph;
      obj_dict["agent_id"] = agent_ptr->agent_id;
      obj_dict["action_failure_penalty"] = agent_ptr->action_failure_penalty;
      obj_dict["current_stat_reward"] = agent_ptr->current_stat_reward;
      obj_dict["prev_action_name"] = agent_ptr->prev_action_name;
      obj_dict["steps_without_motion"] = agent_ptr->steps_without_motion;
    }

    if (auto* converter = dynamic_cast<Converter*>(obj)) {
      obj_dict["is_converting"] = converter->converting;
      obj_dict["is_cooling_down"] = converter->cooling_down;
      obj_dict["conversion_duration"] = converter->conversion_ticks;
      obj_dict["cooldown_duration"] = converter->next_cooldown_time();
      obj_dict["output_limit"] = converter->max_output;
      py::dict input_resources_dict;
      for (const auto& [resource, quantity] : converter->input_resources) {
        input_resources_dict[py::int_(resource)] = quantity;
      }
      obj_dict["input_resources"] = input_resources_dict;
      py::dict output_resources_dict;
      for (const auto& [resource, quantity] : converter->output_resources) {
        output_resources_dict[py::int_(resource)] = quantity;
      }
      obj_dict["output_resources"] = output_resources_dict;
    }

    if (auto* assembler = dynamic_cast<Assembler*>(obj)) {
      obj_dict["cooldown_remaining"] = assembler->cooldown_remaining();
      obj_dict["cooldown_duration"] = assembler->cooldown_duration;
      obj_dict["cooldown_progress"] = assembler->cooldown_progress();
      obj_dict["is_clipped"] = assembler->is_clipped;
      obj_dict["is_clip_immune"] = assembler->clip_immune;
      obj_dict["uses_count"] = assembler->uses_count;
      obj_dict["max_uses"] = assembler->max_uses;
      obj_dict["allow_partial_usage"] = assembler->allow_partial_usage;
      obj_dict["exhaustion"] = assembler->exhaustion;
      obj_dict["cooldown_multiplier"] = assembler->cooldown_multiplier;
      obj_dict["current_recipe_id"] = static_cast<int>(assembler->get_agent_pattern_byte());

      const Recipe* current_recipe = assembler->get_current_recipe();
      if (current_recipe) {
        py::dict input_resources_dict;
        for (const auto& [resource, quantity] : current_recipe->input_resources) {
          input_resources_dict[py::int_(resource)] = quantity;
        }
        obj_dict["current_recipe_inputs"] = input_resources_dict;

        py::dict output_resources_dict;
        for (const auto& [resource, quantity] : current_recipe->output_resources) {
          output_resources_dict[py::int_(resource)] = quantity;
        }
        obj_dict["current_recipe_outputs"] = output_resources_dict;
        obj_dict["current_recipe_cooldown"] = current_recipe->cooldown;
      }

      const auto& active_recipes = assembler->is_clipped ? assembler->unclip_recipes : assembler->recipes;
      py::list recipes_list;
      for (const auto& recipe_ptr : active_recipes) {
        if (!recipe_ptr) {
          continue;
        }
        py::dict recipe_dict;
        py::dict input_resources_dict;
        for (const auto& [resource, quantity] : recipe_ptr->input_resources) {
          input_resources_dict[py::int_(resource)] = quantity;
        }
        recipe_dict["inputs"] = input_resources_dict;

        py::dict output_resources_dict;
        for (const auto& [resource, quantity] : recipe_ptr->output_resources) {
          output_resources_dict[py::int_(resource)] = quantity;
        }
        recipe_dict["outputs"] = output_resources_dict;
        recipe_dict["cooldown"] = recipe_ptr->cooldown;
        recipes_list.append(recipe_dict);
      }
      obj_dict["recipes"] = recipes_list;
    }

    if (auto* chest = dynamic_cast<Chest*>(obj)) {
      obj_dict["resource_type"] = static_cast<int>(chest->resource_type);
      obj_dict["max_inventory"] = chest->max_inventory;
      py::dict position_deltas_dict;
      for (const auto& [pos, delta] : chest->position_deltas) {
        position_deltas_dict[py::int_(pos)] = delta;
      }
      obj_dict["position_deltas"] = position_deltas_dict;
    }

    objects[py::int_(obj_id)] = obj_dict;
  }

  return objects;
}

py::list MettaGrid::action_names() {
  py::list names;
  for (const auto& name : _engine->flat_action_names()) {
    names.append(py::str(name));
  }
  return names;
}

GridCoord MettaGrid::map_width() {
  return _engine->map_width();
}

GridCoord MettaGrid::map_height() {
  return _engine->map_height();
}

py::dict MettaGrid::feature_normalizations() {
  py::dict normalizations;
  const auto& encoder = _engine->observation_encoder();
  for (const auto& [feature_id, value] : encoder.feature_normalizations()) {
    normalizations[py::int_(feature_id)] = py::float_(value);
  }
  return normalizations;
}

py::dict MettaGrid::feature_spec() {
  py::dict feature_spec;
  const auto& encoder = _engine->observation_encoder();
  const auto& names = encoder.feature_names();
  const auto& normalizations = encoder.feature_normalizations();

  for (const auto& [feature_id, feature_name] : names) {
    py::dict spec;
    auto norm_it = normalizations.find(feature_id);
    if (norm_it != normalizations.end()) {
      spec["normalization"] = py::float_(norm_it->second);
    }
    spec["id"] = py::int_(feature_id);

    if (feature_name == "tag") {
      py::dict tag_map;
      for (const auto& [tag_id, tag_name] : _tag_id_map) {
        tag_map[py::int_(tag_id)] = py::str(tag_name);
      }
      spec["values"] = tag_map;
    }

    feature_spec[py::str(feature_name)] = spec;
  }
  return feature_spec;
}

size_t MettaGrid::num_agents() const {
  return _engine->num_agents();
}

py::array_t<float> MettaGrid::get_episode_rewards() {
  return _episode_rewards;
}

py::dict MettaGrid::get_episode_stats() {
  py::dict stats;
  stats["game"] = py::cast(_engine->stats().to_dict());

  py::list agent_stats;
  for (auto* agent_ptr : _engine->agents()) {
    agent_stats.append(py::cast(agent_ptr->stats.to_dict()));
  }
  stats["agent"] = agent_stats;
  return stats;
}

py::object MettaGrid::action_space() {
  auto gym = py::module_::import("gymnasium");
  auto spaces = gym.attr("spaces");
  return spaces.attr("Discrete")(py::int_(_engine->flat_action_map().size()));
}

py::object MettaGrid::observation_space() {
  auto gym = py::module_::import("gymnasium");
  auto spaces = gym.attr("spaces");

  auto observation_info = _observations.request();
  auto shape = observation_info.shape;
  auto space_shape = py::tuple(observation_info.ndim - 1);
  for (ssize_t i = 0; i < observation_info.ndim - 1; i++) {
    space_shape[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)];
  }

  ObservationType min_value = std::numeric_limits<ObservationType>::min();
  ObservationType max_value = std::numeric_limits<ObservationType>::max();
  return spaces.attr("Box")(min_value, max_value, space_shape, py::arg("dtype") = dtype_observations());
}

py::list MettaGrid::action_success_py() {
  return py::cast(_engine->action_success());
}

py::list MettaGrid::max_action_args() {
  return py::cast(_engine->max_action_args());
}

py::list MettaGrid::action_catalog() {
  py::list catalog;
  const auto& flat_map = _engine->flat_action_map();
  const auto& flat_names = _engine->flat_action_names();
  const auto& handlers = _engine->action_handlers();
  for (size_t idx = 0; idx < flat_map.size(); ++idx) {
    const auto& mapping = flat_map[idx];
    py::dict entry;
    entry["flat_index"] = py::int_(idx);
    entry["action_id"] = py::int_(mapping.first);
    entry["param"] = py::int_(mapping.second);
    entry["base_name"] = py::str(handlers[static_cast<size_t>(mapping.first)]->action_name());
    entry["variant_name"] = py::str(flat_names[idx]);
    catalog.append(std::move(entry));
  }
  return catalog;
}

py::list MettaGrid::object_type_names_py() {
  return py::cast(object_type_names);
}

py::list MettaGrid::resource_names_py() {
  return py::cast(resource_names);
}

py::none MettaGrid::set_inventory(GridObjectId agent_id,
                                  const std::unordered_map<InventoryItem, InventoryQuantity>& inventory) {
  if (agent_id < num_agents()) {
    _engine->agent(agent_id)->set_inventory(inventory);
  }
  return py::none();
}

// Pybind11 module definition
PYBIND11_MODULE(mettagrid_c, m) {
  m.doc() = "MettaGrid environment";

  py::class_<GridObjectConfig, std::shared_ptr<GridObjectConfig>>(m, "GridObjectConfig");
  bind_wall_config(m);
  bind_inventory_config(m);
  bind_agent_config(m);
  bind_converter_config(m);
  bind_assembler_config(m);
  bind_chest_config(m);
  bind_action_config(m);
  bind_attack_action_config(m);
  bind_change_glyph_action_config(m);
  bind_resource_mod_config(m);
  bind_global_obs_config(m);
  bind_clipper_config(m);
  bind_game_config(m);
  bind_recipe(m);
  PackedCoordinate::bind_packed_coordinate(m);

  py::class_<MettaGrid>(m, "MettaGrid")
      .def(py::init<const GameConfig&, const py::list&, unsigned int>())
      .def("reset", &MettaGrid::reset)
      .def("step", &MettaGrid::step, py::arg("actions").noconvert())
      .def("set_buffers",
           &MettaGrid::set_buffers,
           py::arg("observations").noconvert(),
           py::arg("terminals").noconvert(),
           py::arg("truncations").noconvert(),
           py::arg("rewards").noconvert())
      .def("grid_objects",
           &MettaGrid::grid_objects,
           py::arg("min_row") = -1,
           py::arg("max_row") = -1,
           py::arg("min_col") = -1,
           py::arg("max_col") = -1,
           py::arg("ignore_types") = py::list())
      .def("action_names", &MettaGrid::action_names)
      .def_property_readonly("map_width", &MettaGrid::map_width)
      .def_property_readonly("map_height", &MettaGrid::map_height)
      .def_property_readonly("num_agents", &MettaGrid::num_agents)
      .def("get_episode_rewards", &MettaGrid::get_episode_rewards)
      .def("get_episode_stats", &MettaGrid::get_episode_stats)
      .def_property_readonly("action_space", &MettaGrid::action_space)
      .def_property_readonly("observation_space", &MettaGrid::observation_space)
      .def("action_success", &MettaGrid::action_success_py)
      .def("max_action_args", &MettaGrid::max_action_args)
      .def("action_catalog", &MettaGrid::action_catalog)
      .def("object_type_names", &MettaGrid::object_type_names_py)
      .def("feature_normalizations", &MettaGrid::feature_normalizations)
      .def("feature_spec", &MettaGrid::feature_spec)
      .def_readonly("obs_width", &MettaGrid::obs_width)
      .def_readonly("obs_height", &MettaGrid::obs_height)
      .def_readonly("max_steps", &MettaGrid::max_steps)
      .def_readonly("current_step", &MettaGrid::current_step)
      .def("resource_names", &MettaGrid::resource_names_py)
      .def_readonly("initial_grid_hash", &MettaGrid::initial_grid_hash)
      .def("set_inventory", &MettaGrid::set_inventory, py::arg("agent_id"), py::arg("inventory"));

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
