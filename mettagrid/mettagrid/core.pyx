# distutils: language = c++
# cython: language_level=3

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

# Python imports
import numpy as np
cimport numpy as cnp
import gymnasium as gym
from omegaconf import DictConfig, ListConfig, OmegaConf

# Import in the correct order to avoid forward declaration issues
from mettagrid.grid_object cimport GridCoord, GridObject, GridObjectId, GridLocation, Layer
from mettagrid.grid cimport Grid
from mettagrid.event cimport EventManager, EventHandler
from mettagrid.stats_tracker cimport StatsTracker
from mettagrid.action_handler cimport ActionArg, ActionHandler, ActionConfig
from mettagrid.observation_encoder cimport ObsType, ObservationEncoder

# Now import core.pxd which depends on all the above
from mettagrid.core cimport MettaGrid

# Now import the object module definitions which depend on the core types
from mettagrid.objects.constants cimport ObjectLayers, ObjectTypeNames, ObjectTypeAscii, InventoryItemNames, ObjectType
from mettagrid.objects.metta_object cimport ObjectConfig
from mettagrid.objects.agent cimport Agent
from mettagrid.objects.wall cimport Wall
from mettagrid.objects.converter cimport Converter
from mettagrid.objects.production_handler cimport ProductionHandler, CoolDownHandler

# Finally import action handlers
from mettagrid.actions.move cimport Move
from mettagrid.actions.rotate cimport Rotate
from mettagrid.actions.get_output cimport GetOutput
from mettagrid.actions.put_recipe_items cimport PutRecipeItems
from mettagrid.actions.attack cimport Attack
from mettagrid.actions.attack_nearest cimport AttackNearest
from mettagrid.actions.noop cimport Noop
from mettagrid.actions.swap cimport Swap
from mettagrid.actions.change_color cimport ChangeColorAction

# Constants
obs_np_type = np.uint8

# Utility functions
cdef ObjectConfig convert_to_object_config(config_dict):
    """Convert a Python dictionary to an ObjectConfig."""
    cdef ObjectConfig obj_config
    cdef string key_str
    
    for key, value in config_dict.items():
        key_str = key.encode('utf8')
        obj_config[key_str] = int(value)
    
    return obj_config

cdef ActionConfig convert_to_action_config(config_dict):
    """Convert a Python dictionary to an ActionConfig."""
    cdef ActionConfig action_config
    cdef string key_str
    
    for key, value in config_dict.items():
        key_str = key.encode('utf8')
        action_config[key_str] = int(value)
    
    return action_config

# Wrapper class for the C++ implementation
cdef class PyMettaGrid:
    cdef:
        object _cfg
        Grid* _grid
        EventManager* _event_manager
        StatsTracker* _stats
        
        # The C++ implementation of MettaGrid
        MettaGrid* _cpp_mettagrid
        
        # Python-accessible buffers/data
        cnp.ndarray _observations_np
        ObsType[:,:,:,:] _observations
        cnp.ndarray _terminals_np
        char[:] _terminals
        cnp.ndarray _truncations_np
        char[:] _truncations
        cnp.ndarray _rewards_np
        float[:] _rewards
        cnp.ndarray _episode_rewards_np
        float[:] _episode_rewards
        cnp.ndarray _group_rewards_np
        double[:] _group_rewards
        
        # Track agents for Python API
        vector[Agent*] _agents

    def __init__(self, env_cfg: DictConfig | ListConfig, np_map: cnp.ndarray):
        # Initialize configuration
        cfg = OmegaConf.create(env_cfg.game)
        self._cfg = cfg
        num_agents = cfg.num_agents
        max_timestep = cfg.max_steps

        # Create a vector of Layer type instead of int
        cdef vector[Layer] layer_for_type_id_vec
        for type_id, layer in dict(ObjectLayers).items():
            # Cast to Layer type if needed
            layer_for_type_id_vec.push_back(<Layer>layer)
        
        obs_width = cfg.obs_width
        obs_height = cfg.obs_height
        
        # Initialize the grid
        self._grid = new Grid(np_map.shape[1], np_map.shape[0], layer_for_type_id_vec)
        
        # Initialize event manager and stats
        self._event_manager = new EventManager()
        self._stats = new StatsTracker()
        
        # Initialize event manager
        self._event_manager.init(self._grid, self._stats)

        # The order of this needs to match the order in the Events enum
        self._event_manager.event_handlers.push_back(new ProductionHandler(self._event_manager))
        self._event_manager.event_handlers.push_back(new CoolDownHandler(self._event_manager))
        
        # Create the C++ MettaGrid instance
        self._cpp_mettagrid = new MettaGrid(self._grid, num_agents, max_timestep, 
                                           obs_width, obs_height)
        
        # Connect the event manager and stats tracker
        self._cpp_mettagrid.set_stats(self._stats)
        self._cpp_mettagrid.init_event_manager(self._event_manager)
        
        # Create observation and reward buffers
        # For now, hardcode feature size to avoid calling C++ before buffers are set
        cdef int feature_size = 32  # Estimate, will be updated after fully initialized
        self._observations_np = np.zeros(
            (num_agents, feature_size, obs_height, obs_width),
            dtype=obs_np_type)
        self._terminals_np = np.zeros(num_agents, dtype=np.int8)
        self._truncations_np = np.zeros(num_agents, dtype=np.int8)
        self._rewards_np = np.zeros(num_agents, dtype=np.float32)
        self._episode_rewards_np = np.zeros(num_agents, dtype=np.float32)
        
        # Set buffer views for Cython access
        self._observations = self._observations_np
        self._terminals = self._terminals_np
        self._truncations = self._truncations_np
        self._rewards = self._rewards_np
        self._episode_rewards = self._episode_rewards_np
        
        # Initialize group rewards
        self._group_rewards_np = np.zeros(len(cfg.groups))
        self._group_rewards = self._group_rewards_np
        
        # Set up buffers in C++ implementation
        self._cpp_mettagrid.set_buffers(
            <ObsType*><void*>self._observations_np.data,
            <char*><void*>self._terminals_np.data,
            <char*><void*>self._truncations_np.data,
            <float*><void*>self._rewards_np.data,
            <float*><void*>self._episode_rewards_np.data,
            num_agents
        )
        
        # Initialize group rewards in C++ implementation
        self._cpp_mettagrid.init_group_rewards(<double*><void*>self._group_rewards_np.data, self._group_rewards_np.shape[0])
        
        # Initialize objects from the map
        self._initialize_objects_from_map(np_map, cfg)
        
        # Set up action handlers
        self._setup_action_handlers(cfg)
        
        # Initialize group reward percentages and sizes
        cdef unsigned int group_id
        cdef float pct
        group_sizes = {g.id: 0 for g in cfg.groups.values()}
        group_reward_pct = {g.id: g.get("group_reward_pct", 0) for g in cfg.groups.values()}
        
        for group_id, pct in group_reward_pct.items():
            self._cpp_mettagrid.set_group_reward_pct(group_id, pct)
        
        for group_id, size in group_sizes.items():
            self._cpp_mettagrid.set_group_size(group_id, size)

    def __dealloc__(self):
        # Clean up the C++ objects
        if self._cpp_mettagrid != NULL:
            del self._cpp_mettagrid
        if self._grid != NULL:
            del self._grid
        if self._event_manager != NULL:
            del self._event_manager
        if self._stats != NULL:
            del self._stats

    # Helper method to get grid features as Python list
    cdef list _get_grid_features(self):
        cdef vector[string] features = self._cpp_mettagrid.grid_features()
        cdef list result = []
        cdef int i
        for i in range(features.size()):
            result.append(features[i].decode('utf8'))
        return result

    cdef void _initialize_objects_from_map(self, np_map, cfg):
        """Initialize grid objects from the provided map configuration."""
        cdef Agent* agent
        cdef Converter* converter = NULL
        cdef string group_name_bytes
        cdef unsigned int group_id
        cdef map[string, float] cpp_rewards
        cdef string item_bytes
        cdef string max_key
        cdef ObjectConfig obj_config

        group_sizes = {g.id: 0 for g in cfg.groups.values()}

        for r in range(np_map.shape[0]):
            for c in range(np_map.shape[1]):
                if np_map[r,c] == "wall":
                    wall_config = OmegaConf.to_container(cfg.objects.wall)
                    obj_config = convert_to_object_config(wall_config)
                    wall = new Wall(<GridCoord>r, <GridCoord>c, obj_config)
                    self._grid.add_object(wall)
                    self._stats.incr(b"objects.wall")
                elif np_map[r,c] == "block":
                    block_config = OmegaConf.to_container(cfg.objects.block)
                    obj_config = convert_to_object_config(block_config)
                    block = new Wall(<GridCoord>r, <GridCoord>c, obj_config)
                    self._grid.add_object(block)
                    self._stats.incr(b"objects.block")
                elif np_map[r,c].startswith("mine"):
                    m = np_map[r,c]
                    if "." not in m:
                        m = "mine.red"
                    mine_config = OmegaConf.to_container(cfg.objects[m])
                    obj_config = convert_to_object_config(mine_config)
                    converter = new Converter(<GridCoord>r, <GridCoord>c, obj_config, ObjectType.MineT)
                elif np_map[r,c].startswith("generator"):
                    m = np_map[r,c]
                    if "." not in m:
                        m = "generator.red"
                    gen_config = OmegaConf.to_container(cfg.objects[m])
                    obj_config = convert_to_object_config(gen_config)
                    converter = new Converter(<GridCoord>r, <GridCoord>c, obj_config, ObjectType.GeneratorT)
                elif np_map[r,c] == "altar":
                    altar_config = OmegaConf.to_container(cfg.objects.altar)
                    obj_config = convert_to_object_config(altar_config)
                    converter = new Converter(<GridCoord>r, <GridCoord>c, obj_config, ObjectType.AltarT)
                elif np_map[r,c] == "armory":
                    armory_config = OmegaConf.to_container(cfg.objects.armory)
                    obj_config = convert_to_object_config(armory_config)
                    converter = new Converter(<GridCoord>r, <GridCoord>c, obj_config, ObjectType.ArmoryT)
                elif np_map[r,c] == "lasery":
                    lasery_config = OmegaConf.to_container(cfg.objects.lasery)
                    obj_config = convert_to_object_config(lasery_config)
                    converter = new Converter(<GridCoord>r, <GridCoord>c, obj_config, ObjectType.LaseryT)
                elif np_map[r,c] == "lab":
                    lab_config = OmegaConf.to_container(cfg.objects.lab)
                    obj_config = convert_to_object_config(lab_config)
                    converter = new Converter(<GridCoord>r, <GridCoord>c, obj_config, ObjectType.LabT)
                elif np_map[r,c] == "factory":
                    factory_config = OmegaConf.to_container(cfg.objects.factory)
                    obj_config = convert_to_object_config(factory_config)
                    converter = new Converter(<GridCoord>r, <GridCoord>c, obj_config, ObjectType.FactoryT)
                elif np_map[r,c] == "temple":
                    temple_config = OmegaConf.to_container(cfg.objects.temple)
                    obj_config = convert_to_object_config(temple_config)
                    converter = new Converter(<GridCoord>r, <GridCoord>c, obj_config, ObjectType.TempleT)
                elif np_map[r,c].startswith("agent."):
                    parts = np_map[r,c].split(".")
                    group_name = parts[1]
                    # Convert Python string to C++ string
                    group_name_bytes = group_name.encode('utf8')
                    
                    agent_cfg = OmegaConf.to_container(OmegaConf.merge(
                        cfg.agent, cfg.groups[group_name].props))
                    rewards = agent_cfg.get("rewards", {})
                    del agent_cfg["rewards"]
                    
                    # Convert the agent config to ObjectConfig
                    agent_obj_config = convert_to_object_config(agent_cfg)
                    
                    # Clear and refill the C++ map
                    cpp_rewards.clear()
                    for item in InventoryItemNames:
                        item_bytes = item.encode('utf8')
                        cpp_rewards[item_bytes] = rewards.get(item, 0)
                        max_key = (item + "_max").encode('utf8')
                        cpp_rewards[max_key] = rewards.get(item + "_max", 1000)
                    
                    group_id = cfg.groups[group_name].id
                    agent = new Agent(
                        <GridCoord>r, <GridCoord>c, group_name_bytes, group_id, agent_obj_config, cpp_rewards)
                    self._grid.add_object(<GridObject*>agent)
                    agent.agent_id = self._agents.size()
                    self._cpp_mettagrid.add_agent(agent)
                    self._agents.push_back(agent)  # Keep track for Python API
                    
                    # Increment group size
                    group_sizes[group_id] += 1
                    self._cpp_mettagrid.set_group_size(group_id, group_sizes[group_id])

                if converter != NULL:
                    stat_bytes = ("objects." + np_map[r,c]).encode('utf8')
                    self._stats.incr(stat_bytes)
                    self._grid.add_object(<GridObject*>converter)
                    converter.set_event_manager(self._event_manager)
                    converter = NULL
            
    cdef void _setup_action_handlers(self, cfg):
        """Set up the action handlers based on configuration."""
        cdef vector[ActionHandler*] actions
        cdef ActionConfig action_config
        
        # Create action handlers based on config
        if cfg.actions.put_items.enabled:
            put_items_config = OmegaConf.to_container(cfg.actions.put_items)
            action_config = convert_to_action_config(put_items_config)
            actions.push_back(new PutRecipeItems(action_config))
            
        if cfg.actions.get_items.enabled:
            get_items_config = OmegaConf.to_container(cfg.actions.get_items)
            action_config = convert_to_action_config(get_items_config)
            actions.push_back(new GetOutput(action_config))
            
        if cfg.actions.noop.enabled:
            noop_config = OmegaConf.to_container(cfg.actions.noop)
            action_config = convert_to_action_config(noop_config)
            actions.push_back(new Noop(action_config))
            
        if cfg.actions.move.enabled:
            move_config = OmegaConf.to_container(cfg.actions.move)
            action_config = convert_to_action_config(move_config)
            actions.push_back(new Move(action_config))
            
        if cfg.actions.rotate.enabled:
            rotate_config = OmegaConf.to_container(cfg.actions.rotate)
            action_config = convert_to_action_config(rotate_config)
            actions.push_back(new Rotate(action_config))
            
        if cfg.actions.attack.enabled:
            attack_config = OmegaConf.to_container(cfg.actions.attack)
            action_config = convert_to_action_config(attack_config)
            actions.push_back(new Attack(action_config))
            
            # For AttackNearest, reuse the same config
            actions.push_back(new AttackNearest(action_config))
            
        if cfg.actions.swap.enabled:
            swap_config = OmegaConf.to_container(cfg.actions.swap)
            action_config = convert_to_action_config(swap_config)
            actions.push_back(new Swap(action_config))
            
        if cfg.actions.change_color.enabled:
            change_color_config = OmegaConf.to_container(cfg.actions.change_color)
            action_config = convert_to_action_config(change_color_config)
            actions.push_back(new ChangeColorAction(action_config))
        
        # Initialize the action handlers in the C++ implementation
        self._cpp_mettagrid.init_action_handlers(actions)
    
    # Python API methods
    
    def reset(self):
        """Reset the environment and return initial observation."""
        # Can't reset after stepping
        if self._cpp_mettagrid.current_timestep() > 0:
            raise NotImplementedError("Cannot reset after stepping")

        # Reset buffers
        self._terminals[:] = 0
        self._truncations[:] = 0
        self._episode_rewards[:] = 0
        self._observations[:, :, :, :] = 0
        self._rewards[:] = 0

        # Initialize observations
        cdef int** actions = <int**>malloc(self._agents.size() * sizeof(int*))
        cdef int i
        for i in range(self._agents.size()):
            actions[i] = <int*>malloc(2 * sizeof(int))
            actions[i][0] = 0
            actions[i][1] = 0
        
        self._cpp_mettagrid.compute_observations(actions)
        
        # Clean up temporary actions array
        for i in range(self._agents.size()):
            free(actions[i])
        free(actions)
        
        return (self._observations_np, {})
    
    def step(self, actions):
        """Take a step in the environment with the given actions."""
        # Convert numpy array to the right shape if needed
        actions_array = np.asarray(actions, dtype=np.int32)
        
        # Create a C-compatible array for actions
        cdef int** c_actions = <int**>malloc(actions_array.shape[0] * sizeof(int*))
        cdef int i
        for i in range(actions_array.shape[0]):
            c_actions[i] = <int*>malloc(2 * sizeof(int))
            c_actions[i][0] = actions_array[i, 0]
            c_actions[i][1] = actions_array[i, 1]
        
        # Take a step in the C++ implementation
        self._cpp_mettagrid.step(c_actions)
        
        # Process group rewards - use a raw pointer cast that Cython can understand
        self._cpp_mettagrid.compute_group_rewards(<float*><void*>self._rewards_np.data)
        
        # Clean up the C-array for actions
        for i in range(actions_array.shape[0]):
            free(c_actions[i])
        free(c_actions)
        
        return (self._observations_np, self._rewards_np, self._terminals_np, self._truncations_np, {})
    
    # Rest of the methods remain unchanged
    def action_names(self):
        # Get the action success from the C++ implementation and convert to Python list
        cdef vector[bool] success = self._cpp_mettagrid.action_success()
        cdef list result = []
        cdef int i
        for i in range(success.size()):
            result.append(success[i])
        return result
    
    def current_timestep(self):
        return self._cpp_mettagrid.current_timestep()
    
    def map_width(self):
        return self._cpp_mettagrid.map_width()
    
    def map_height(self):
        return self._cpp_mettagrid.map_height()
    
    def grid_features(self):
        return self._get_grid_features()
    
    def num_agents(self):
        return self._cpp_mettagrid.num_agents()
    
    def enable_reward_decay(self, decay_time_steps = None):
        """Enable reward decay mechanism."""
        cdef int decay_time = -1 if decay_time_steps is None else decay_time_steps
        self._cpp_mettagrid.enable_reward_decay(decay_time)
    
    def disable_reward_decay(self):
        """Disable reward decay mechanism."""
        self._cpp_mettagrid.disable_reward_decay()
    
    def observe(
        self,
        GridObjectId observer_id,
        unsigned short obs_width,
        unsigned short obs_height,
        observation):
        
        # Convert from numpy array if needed
        cdef ObsType[:,:,:] obs_view
        if isinstance(observation, np.ndarray):
            obs_view = observation
        else:
            obs_view = observation
        
        # Create a flat array for C++ to write into
        features_size = len(self.grid_features())
        cdef cnp.ndarray flat_obs = np.zeros((obs_height, obs_width, features_size), dtype=np.uint8)
        
        # Call the C++ implementation
        self._cpp_mettagrid.observe(observer_id, obs_width, obs_height, <ObsType*><void*>flat_obs.data)
        
        # Copy the results to the original observation (can be optimized)
        for r in range(obs_height):
            for c in range(obs_width):
                for f in range(features_size):
                    obs_view[r, c, f] = flat_obs[r, c, f]
    
    def observe_at(
        self,
        unsigned short row,
        unsigned short col,
        unsigned short obs_width,
        unsigned short obs_height,
        observation):
        
        # Convert from numpy array if needed
        cdef ObsType[:,:,:] obs_view
        if isinstance(observation, np.ndarray):
            obs_view = observation
        else:
            obs_view = observation
        
        # Create a flat array for C++ to write into
        features_size = len(self.grid_features())
        cdef cnp.ndarray flat_obs = np.zeros((obs_height, obs_width, features_size), dtype=np.uint8)
        
        # Call the C++ implementation
        self._cpp_mettagrid.observe_at(row, col, obs_width, obs_height, <ObsType*><void*>flat_obs.data)
        
        # Copy the results back to the original observation
        for r in range(obs_height):
            for c in range(obs_width):
                for f in range(features_size):
                    obs_view[r, c, f] = flat_obs[r, c, f]
    
    def get_episode_rewards(self):
        return self._episode_rewards_np
        
    def get_episode_stats(self):
        """Get statistics from the game and agents."""
        # Start with empty dictionaries
        game_stats = {}
        agent_stats = []
        
        # Convert C++ game stats to Python dict - using a direct approach
        # that doesn't rely on complex iterator types
        cdef map[string, int] cpp_stats = self._stats.stats()
        
        # Use a simpler approach - create a Python dictionary from the C++ map
        # This avoids the iterator issues
        for key_bytes, value in dict(cpp_stats).items():
            key = key_bytes.decode('utf8')
            game_stats[key] = value
        
        # Convert agent stats - also using a simplified approach
        for i in range(self._agents.size()):
            agent = self._agents[i]
            agent_stat_map = agent.stats.stats()
            agent_stat_dict = {}
            
            # Convert to Python dict 
            for key_bytes, value in dict(agent_stat_map).items():
                key = key_bytes.decode('utf8')
                agent_stat_dict[key] = value
                
            agent_stats.append(agent_stat_dict)
            
        return {
            "game": game_stats,
            "agent": agent_stats
        }
    
    def render_ascii(self):
        """Render the grid as an ASCII representation."""
        grid = np.full((self._grid.height, self._grid.width), " ", dtype=np.str_)
        
        # Iterate through objects and update grid
        cdef GridObject* obj
        cdef int obj_id
        
        for obj_id in range(1, self._grid.objects.size()):
            obj = self._grid.object(obj_id)
            grid[obj.location.r, obj.location.c] = ObjectTypeAscii[obj._type_id].decode('utf8')
            
        return grid
    
    # Gym compatibility properties
    
    @property
    def action_space(self):
        # Get max action arguments
        cdef vector[unsigned char] max_args = self._cpp_mettagrid.max_action_args()
        cdef unsigned char max_arg = 0
        cdef int i
        
        # Find the maximum value
        for i in range(max_args.size()):
            if max_args[i] > max_arg:
                max_arg = max_args[i]
        
        return gym.spaces.MultiDiscrete(
            (self._cpp_mettagrid.num_agents(), max_arg + 1), 
            dtype=np.int64
        )
    
    @property
    def observation_space(self):
        features_size = len(self.grid_features())
        return gym.spaces.Box(
            0,
            255,
            shape=(self._cpp_mettagrid.map_height(), self._cpp_mettagrid.map_width(), features_size),
            dtype=obs_np_type
        )
    
    def action_success(self):
        cdef vector[bool] success = self._cpp_mettagrid.action_success()
        cdef list result = []
        cdef int i
        for i in range(success.size()):
            result.append(success[i])
        return result
    
    def max_action_args(self):
        cdef vector[unsigned char] max_args = self._cpp_mettagrid.max_action_args()
        cdef list result = []
        cdef int i
        for i in range(max_args.size()):
            result.append(max_args[i])
        return result
    
    def object_type_names(self):
        return ObjectTypeNames
    
    def inventory_item_names(self):
        return InventoryItemNames
    
    def render(self):
        grid = self.render_ascii()
        for r in grid:
            print("".join(r))