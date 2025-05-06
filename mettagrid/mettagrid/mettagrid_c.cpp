#include "mettagrid_c.hpp"
#include "grid.hpp"
#include "event.hpp"
#include "stats_tracker.hpp"
#include "action_handler.hpp"
#include "agent.hpp"
#include "observation_encoder.hpp"
#include "objects/constants.hpp"
#include "objects/wall.hpp"
#include "objects/converter.hpp"
#include "actions/move.hpp"
#include "actions/rotate.hpp"
#include "actions/get_output.hpp"
#include "actions/put_recipe_items.hpp"
#include "actions/attack.hpp"
#include "actions/attack_nearest.hpp"
#include "actions/noop.hpp"
#include "actions/swap.hpp"
#include "actions/change_color.hpp"

#include <gymnasium/gymnasium.hpp>
#include <pybind11/gil.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/operators.h>

namespace py = pybind11;

// Constructor implementation
MettaGrid::MettaGrid(py::dict env_cfg, py::array_t<char> map) {
    auto cfg = env_cfg["game"].cast<py::dict>();
    _cfg = cfg;
    
    int num_agents = cfg["num_agents"].cast<int>();
    _max_timestep = cfg["max_steps"].cast<unsigned int>();
    _obs_width = cfg["obs_width"].cast<unsigned short>();
    _obs_height = cfg["obs_height"].cast<unsigned short>();
    
    _current_timestep = 0;
    
    // Initialize grid
    auto map_info = map.request();
    _grid = std::make_unique<Grid>(map_info.shape[1], map_info.shape[0]);
    
    // Initialize event manager and stats tracker
    _event_manager = std::make_unique<EventManager>();
    _stats = std::make_unique<StatsTracker>();
    
    // Initialize observation encoder
    _obs_encoder = std::make_unique<ObservationEncoder>();
    _grid_features = _obs_encoder->feature_names();
    
    // Initialize buffers
    auto observations = py::array_t<unsigned char>({
        num_agents,
        _grid_features.size(),
        _obs_height,
        _obs_width
    });
    auto terminals = py::array_t<char>(num_agents);
    auto truncations = py::array_t<char>(num_agents);
    auto rewards = py::array_t<float>(num_agents);
    
    set_buffers(observations, terminals, truncations, rewards);
    
    // Initialize action handlers
    std::vector<std::unique_ptr<ActionHandler>> actions;
    
    if (cfg["actions"]["put_items"]["enabled"].cast<bool>()) {
        actions.push_back(std::make_unique<PutRecipeItems>(cfg["actions"]["put_items"]));
    }
    if (cfg["actions"]["get_items"]["enabled"].cast<bool>()) {
        actions.push_back(std::make_unique<GetOutput>(cfg["actions"]["get_items"]));
    }
    if (cfg["actions"]["noop"]["enabled"].cast<bool>()) {
        actions.push_back(std::make_unique<Noop>(cfg["actions"]["noop"]));
    }
    if (cfg["actions"]["move"]["enabled"].cast<bool>()) {
        actions.push_back(std::make_unique<Move>(cfg["actions"]["move"]));
    }
    if (cfg["actions"]["rotate"]["enabled"].cast<bool>()) {
        actions.push_back(std::make_unique<Rotate>(cfg["actions"]["rotate"]));
    }
    if (cfg["actions"]["attack"]["enabled"].cast<bool>()) {
        actions.push_back(std::make_unique<Attack>(cfg["actions"]["attack"]));
        actions.push_back(std::make_unique<AttackNearest>(cfg["actions"]["attack"]));
    }
    if (cfg["actions"]["swap"]["enabled"].cast<bool>()) {
        actions.push_back(std::make_unique<Swap>(cfg["actions"]["swap"]));
    }
    if (cfg["actions"]["change_color"]["enabled"].cast<bool>()) {
        actions.push_back(std::make_unique<ChangeColorAction>(cfg["actions"]["change_color"]));
    }
    
    init_action_handlers(actions);
    
    // Initialize group rewards
    auto groups = cfg["groups"].cast<py::dict>();
    _group_rewards_np = py::array_t<double>(groups.size());
    _group_rewards = _group_rewards_np;
    
    for (const auto& [key, value] : groups) {
        auto group = value.cast<py::dict>();
        unsigned int id = group["id"].cast<unsigned int>();
        _group_sizes[id] = 0;
        _group_reward_pct[id] = group.get("group_reward_pct", 0.0).cast<float>();
    }
    
    // Initialize objects from map
    auto map_data = map.unchecked<2>();
    for (int r = 0; r < map_info.shape[0]; r++) {
        for (int c = 0; c < map_info.shape[1]; c++) {
            std::string cell = map_data(r, c).cast<std::string>();
            
            if (cell == "wall") {
                auto wall = std::make_unique<Wall>(r, c, cfg["objects"]["wall"]);
                _grid->add_object(wall.get());
                _stats->incr("objects.wall");
            }
            else if (cell == "block") {
                auto block = std::make_unique<Wall>(r, c, cfg["objects"]["block"]);
                _grid->add_object(block.get());
                _stats->incr("objects.block");
            }
            else if (cell.starts_with("mine")) {
                std::string m = cell;
                if (m.find('.') == std::string::npos) {
                    m = "mine.red";
                }
                auto converter = std::make_unique<Converter>(r, c, cfg["objects"][m], ObjectType::MineT);
                _grid->add_object(converter.get());
                converter->set_event_manager(_event_manager.get());
                _stats->incr("objects." + cell);
            }
            // TODO: Add other object types
        }
    }
}

MettaGrid::~MettaGrid() = default;

// Example implementation of some key methods
py::tuple MettaGrid::reset() {
    if (_current_timestep > 0) {
        throw std::runtime_error("Cannot reset after stepping");
    }

    // Reset all buffers
    auto terminals_view = _terminals.mutable_unchecked<1>();
    auto truncations_view = _truncations.mutable_unchecked<1>();
    auto episode_rewards_view = _episode_rewards.mutable_unchecked<1>();
    auto observations_view = _observations.mutable_unchecked<4>();
    auto rewards_view = _rewards.mutable_unchecked<1>();

    for (py::ssize_t i = 0; i < terminals_view.shape(0); i++) {
        terminals_view(i) = 0;
        truncations_view(i) = 0;
        episode_rewards_view(i) = 0;
        rewards_view(i) = 0;
    }

    // Clear observations
    for (py::ssize_t i = 0; i < observations_view.shape(0); i++) {
        for (py::ssize_t j = 0; j < observations_view.shape(1); j++) {
            for (py::ssize_t k = 0; k < observations_view.shape(2); k++) {
                for (py::ssize_t l = 0; l < observations_view.shape(3); l++) {
                    observations_view(i, j, k, l) = 0;
                }
            }
        }
    }

    // Compute initial observations
    auto zero_actions = py::array_t<int>({_agents.size(), 2});
    _compute_observations(zero_actions);

    return py::make_tuple(_observations_np, py::dict());
}

void MettaGrid::set_buffers(
    py::array_t<unsigned char> observations,
    py::array_t<char> terminals,
    py::array_t<char> truncations,
    py::array_t<float> rewards
) {
    _observations_np = observations;
    _observations = observations;
    _terminals_np = terminals;
    _terminals = terminals;
    _truncations_np = truncations;
    _truncations = truncations;
    _rewards_np = rewards;
    _rewards = rewards;
    _episode_rewards_np = py::array_t<float>(rewards.shape(0));
    _episode_rewards = _episode_rewards_np;

    for (size_t i = 0; i < _agents.size(); i++) {
        _agents[i]->init(&_rewards.mutable_unchecked<1>()(i));
    }
}

// TODO: Implement remaining methods
// - step()
// - grid()
// - grid_objects()
// - action_names()
// - current_timestep()
// - map_width()
// - map_height()
// - grid_features()
// - num_agents()
// - observe()
// - observe_at()
// - get_episode_rewards()
// - get_episode_stats()
// - render_ascii()
// - action_space()
// - observation_space()
// - action_success()
// - max_action_args()
// - object_type_names()
// - inventory_item_names()
// - render()
// - init_action_handlers()
// - add_agent()
// - _compute_observation()
// - _compute_observations()
// - _step()

// Pybind11 module definition
PYBIND11_MODULE(mettagrid_c, m) {
    m.doc() = "MettaGrid environment"; // optional module docstring
    
    py::class_<MettaGrid>(m, "MettaGrid")
        .def(py::init<py::dict, py::array_t<char>>())
        .def("reset", &MettaGrid::reset)
        .def("step", &MettaGrid::step)
        .def("set_buffers", &MettaGrid::set_buffers)
        .def("grid", &MettaGrid::grid)
        .def("grid_objects", &MettaGrid::grid_objects)
        .def("action_names", &MettaGrid::action_names)
        .def("current_timestep", &MettaGrid::current_timestep)
        .def("map_width", &MettaGrid::map_width)
        .def("map_height", &MettaGrid::map_height)
        .def("grid_features", &MettaGrid::grid_features)
        .def("num_agents", &MettaGrid::num_agents)
        .def("observe", &MettaGrid::observe)
        .def("observe_at", &MettaGrid::observe_at)
        .def("get_episode_rewards", &MettaGrid::get_episode_rewards)
        .def("get_episode_stats", &MettaGrid::get_episode_stats)
        .def("render_ascii", &MettaGrid::render_ascii)
        .def("action_space", &MettaGrid::action_space)
        .def("observation_space", &MettaGrid::observation_space)
        .def("action_success", &MettaGrid::action_success)
        .def("max_action_args", &MettaGrid::max_action_args)
        .def("object_type_names", &MettaGrid::object_type_names)
        .def("inventory_item_names", &MettaGrid::inventory_item_names)
        .def("render", &MettaGrid::render);
} 
