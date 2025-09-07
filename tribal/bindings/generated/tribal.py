from ctypes import *
import os, sys

dir = os.path.dirname(sys.modules["tribal"].__file__)
if sys.platform == "win32":
  libName = "tribal.dll"
elif sys.platform == "darwin":
  libName = "libtribal.dylib"
else:
  libName = "libtribal.so"
dll = cdll.LoadLibrary(os.path.join(dir, libName))

class TribalError(Exception):
    pass

class SeqIterator(object):
    def __init__(self, seq):
        self.idx = 0
        self.seq = seq
    def __iter__(self):
        return self
    def __next__(self):
        if self.idx < len(self.seq):
            self.idx += 1
            return self.seq[self.idx - 1]
        else:
            self.idx = 0
            raise StopIteration

MAP_AGENTS = 15

OBSERVATION_WIDTH = 11

OBSERVATION_HEIGHT = 11

OBSERVATION_LAYERS = 19

MAP_WIDTH = 100

MAP_HEIGHT = 50

class SeqInt(Structure):
    _fields_ = [("ref", c_ulonglong)]

    def __bool__(self):
        return self.ref != None

    def __eq__(self, obj):
        return self.ref == obj.ref

    def __del__(self):
        dll.tribal_seq_int_unref(self)

    def __init__(self):
        self.ref = dll.tribal_new_seq_int()

    def __len__(self):
        return dll.tribal_seq_int_len(self)

    def __getitem__(self, index):
        return dll.tribal_seq_int_get(self, index)

    def __setitem__(self, index, value):
        dll.tribal_seq_int_set(self, index, value)

    def __delitem__(self, index):
        dll.tribal_seq_int_delete(self, index)

    def append(self, value):
        dll.tribal_seq_int_add(self, value)

    def clear(self):
        dll.tribal_seq_int_clear(self)

    def __iter__(self):
        return SeqIterator(self)

class SeqFloat(Structure):
    _fields_ = [("ref", c_ulonglong)]

    def __bool__(self):
        return self.ref != None

    def __eq__(self, obj):
        return self.ref == obj.ref

    def __del__(self):
        dll.tribal_seq_float_unref(self)

    def __init__(self):
        self.ref = dll.tribal_new_seq_float()

    def __len__(self):
        return dll.tribal_seq_float_len(self)

    def __getitem__(self, index):
        return dll.tribal_seq_float_get(self, index)

    def __setitem__(self, index, value):
        dll.tribal_seq_float_set(self, index, value)

    def __delitem__(self, index):
        dll.tribal_seq_float_delete(self, index)

    def append(self, value):
        dll.tribal_seq_float_add(self, value)

    def clear(self):
        dll.tribal_seq_float_clear(self)

    def __iter__(self):
        return SeqIterator(self)

class SeqBool(Structure):
    _fields_ = [("ref", c_ulonglong)]

    def __bool__(self):
        return self.ref != None

    def __eq__(self, obj):
        return self.ref == obj.ref

    def __del__(self):
        dll.tribal_seq_bool_unref(self)

    def __init__(self):
        self.ref = dll.tribal_new_seq_bool()

    def __len__(self):
        return dll.tribal_seq_bool_len(self)

    def __getitem__(self, index):
        return dll.tribal_seq_bool_get(self, index)

    def __setitem__(self, index, value):
        dll.tribal_seq_bool_set(self, index, value)

    def __delitem__(self, index):
        dll.tribal_seq_bool_delete(self, index)

    def append(self, value):
        dll.tribal_seq_bool_add(self, value)

    def clear(self):
        dll.tribal_seq_bool_clear(self)

    def __iter__(self):
        return SeqIterator(self)

def check_error():
    result = dll.tribal_check_error()
    return result

def take_error():
    result = dll.tribal_take_error().decode("utf8")
    return result

class TribalGameConfig(Structure):
    _fields_ = [
        ("max_steps", c_longlong),
        ("ore_per_battery", c_longlong),
        ("batteries_per_heart", c_longlong),
        ("enable_combat", c_bool),
        ("clippy_spawn_rate", c_double),
        ("clippy_damage", c_longlong),
        ("heart_reward", c_double),
        ("ore_reward", c_double),
        ("battery_reward", c_double),
        ("survival_penalty", c_double),
        ("death_penalty", c_double)
    ]

    def __init__(self, max_steps, ore_per_battery, batteries_per_heart, enable_combat, clippy_spawn_rate, clippy_damage, heart_reward, ore_reward, battery_reward, survival_penalty, death_penalty):
        self.max_steps = max_steps
        self.ore_per_battery = ore_per_battery
        self.batteries_per_heart = batteries_per_heart
        self.enable_combat = enable_combat
        self.clippy_spawn_rate = clippy_spawn_rate
        self.clippy_damage = clippy_damage
        self.heart_reward = heart_reward
        self.ore_reward = ore_reward
        self.battery_reward = battery_reward
        self.survival_penalty = survival_penalty
        self.death_penalty = death_penalty

    def __eq__(self, obj):
        return self.max_steps == obj.max_steps and self.ore_per_battery == obj.ore_per_battery and self.batteries_per_heart == obj.batteries_per_heart and self.enable_combat == obj.enable_combat and self.clippy_spawn_rate == obj.clippy_spawn_rate and self.clippy_damage == obj.clippy_damage and self.heart_reward == obj.heart_reward and self.ore_reward == obj.ore_reward and self.battery_reward == obj.battery_reward and self.survival_penalty == obj.survival_penalty and self.death_penalty == obj.death_penalty

class TribalConfig(Structure):
    _fields_ = [
        ("game", TribalGameConfig),
        ("desync_episodes", c_bool)
    ]

    def __init__(self, game, desync_episodes):
        self.game = game
        self.desync_episodes = desync_episodes

    def __eq__(self, obj):
        return self.game == obj.game and self.desync_episodes == obj.desync_episodes

class TribalEnv(Structure):
    _fields_ = [("ref", c_ulonglong)]

    def __bool__(self):
        return self.ref != None

    def __eq__(self, obj):
        return self.ref == obj.ref

    def __del__(self):
        dll.tribal_tribal_env_unref(self)

    def __init__(self, config):
        result = dll.tribal_new_tribal_env(config)
        self.ref = result

    def reset_env(self):
        """
        Reset the environment to initial state
        """
        dll.tribal_tribal_env_reset_env(self)

    def step(self, actions):
        """
        Step environment with actions
        actions: flat sequence of [action_type, argument] pairs
        Length should be MapAgents * 2
        """
        result = dll.tribal_tribal_env_step(self, actions)
        return result

    def get_observations(self):
        """
        Get current observations as flat sequence
        """
        result = dll.tribal_tribal_env_get_observations(self)
        return result

    def get_rewards(self):
        """
        Get current step rewards for each agent
        """
        result = dll.tribal_tribal_env_get_rewards(self)
        return result

    def get_terminated(self):
        """
        Get terminated status for each agent
        """
        result = dll.tribal_tribal_env_get_terminated(self)
        return result

    def get_truncated(self):
        """
        Get truncated status for each agent  
        """
        result = dll.tribal_tribal_env_get_truncated(self)
        return result

    def get_current_step(self):
        """
        Get current step number
        """
        result = dll.tribal_tribal_env_get_current_step(self)
        return result

    def is_episode_done(self):
        """
        Check if episode should end
        """
        result = dll.tribal_tribal_env_is_episode_done(self)
        return result

    def render_text(self):
        """
        Get text rendering of current state
        """
        result = dll.tribal_tribal_env_render_text(self).decode("utf8")
        return result

def default_max_steps():
    """
    Get default max steps value
    """
    result = dll.tribal_default_max_steps()
    return result

def default_tribal_config():
    """
    Create default tribal configuration
    """
    result = dll.tribal_default_tribal_config()
    return result

dll.tribal_seq_int_unref.argtypes = [SeqInt]
dll.tribal_seq_int_unref.restype = None

dll.tribal_new_seq_int.argtypes = []
dll.tribal_new_seq_int.restype = c_ulonglong

dll.tribal_seq_int_len.argtypes = [SeqInt]
dll.tribal_seq_int_len.restype = c_longlong

dll.tribal_seq_int_get.argtypes = [SeqInt, c_longlong]
dll.tribal_seq_int_get.restype = c_longlong

dll.tribal_seq_int_set.argtypes = [SeqInt, c_longlong, c_longlong]
dll.tribal_seq_int_set.restype = None

dll.tribal_seq_int_delete.argtypes = [SeqInt, c_longlong]
dll.tribal_seq_int_delete.restype = None

dll.tribal_seq_int_add.argtypes = [SeqInt, c_longlong]
dll.tribal_seq_int_add.restype = None

dll.tribal_seq_int_clear.argtypes = [SeqInt]
dll.tribal_seq_int_clear.restype = None

dll.tribal_seq_float_unref.argtypes = [SeqFloat]
dll.tribal_seq_float_unref.restype = None

dll.tribal_new_seq_float.argtypes = []
dll.tribal_new_seq_float.restype = c_ulonglong

dll.tribal_seq_float_len.argtypes = [SeqFloat]
dll.tribal_seq_float_len.restype = c_longlong

dll.tribal_seq_float_get.argtypes = [SeqFloat, c_longlong]
dll.tribal_seq_float_get.restype = c_double

dll.tribal_seq_float_set.argtypes = [SeqFloat, c_longlong, c_double]
dll.tribal_seq_float_set.restype = None

dll.tribal_seq_float_delete.argtypes = [SeqFloat, c_longlong]
dll.tribal_seq_float_delete.restype = None

dll.tribal_seq_float_add.argtypes = [SeqFloat, c_double]
dll.tribal_seq_float_add.restype = None

dll.tribal_seq_float_clear.argtypes = [SeqFloat]
dll.tribal_seq_float_clear.restype = None

dll.tribal_seq_bool_unref.argtypes = [SeqBool]
dll.tribal_seq_bool_unref.restype = None

dll.tribal_new_seq_bool.argtypes = []
dll.tribal_new_seq_bool.restype = c_ulonglong

dll.tribal_seq_bool_len.argtypes = [SeqBool]
dll.tribal_seq_bool_len.restype = c_longlong

dll.tribal_seq_bool_get.argtypes = [SeqBool, c_longlong]
dll.tribal_seq_bool_get.restype = c_bool

dll.tribal_seq_bool_set.argtypes = [SeqBool, c_longlong, c_bool]
dll.tribal_seq_bool_set.restype = None

dll.tribal_seq_bool_delete.argtypes = [SeqBool, c_longlong]
dll.tribal_seq_bool_delete.restype = None

dll.tribal_seq_bool_add.argtypes = [SeqBool, c_bool]
dll.tribal_seq_bool_add.restype = None

dll.tribal_seq_bool_clear.argtypes = [SeqBool]
dll.tribal_seq_bool_clear.restype = None

dll.tribal_check_error.argtypes = []
dll.tribal_check_error.restype = c_bool

dll.tribal_take_error.argtypes = []
dll.tribal_take_error.restype = c_char_p

dll.tribal_tribal_env_unref.argtypes = [TribalEnv]
dll.tribal_tribal_env_unref.restype = None

dll.tribal_new_tribal_env.argtypes = [TribalConfig]
dll.tribal_new_tribal_env.restype = c_ulonglong

dll.tribal_tribal_env_reset_env.argtypes = [TribalEnv]
dll.tribal_tribal_env_reset_env.restype = None

dll.tribal_tribal_env_step.argtypes = [TribalEnv, SeqInt]
dll.tribal_tribal_env_step.restype = c_bool

dll.tribal_tribal_env_get_observations.argtypes = [TribalEnv]
dll.tribal_tribal_env_get_observations.restype = SeqInt

dll.tribal_tribal_env_get_rewards.argtypes = [TribalEnv]
dll.tribal_tribal_env_get_rewards.restype = SeqFloat

dll.tribal_tribal_env_get_terminated.argtypes = [TribalEnv]
dll.tribal_tribal_env_get_terminated.restype = SeqBool

dll.tribal_tribal_env_get_truncated.argtypes = [TribalEnv]
dll.tribal_tribal_env_get_truncated.restype = SeqBool

dll.tribal_tribal_env_get_current_step.argtypes = [TribalEnv]
dll.tribal_tribal_env_get_current_step.restype = c_longlong

dll.tribal_tribal_env_is_episode_done.argtypes = [TribalEnv]
dll.tribal_tribal_env_is_episode_done.restype = c_bool

dll.tribal_tribal_env_render_text.argtypes = [TribalEnv]
dll.tribal_tribal_env_render_text.restype = c_char_p

dll.tribal_default_max_steps.argtypes = []
dll.tribal_default_max_steps.restype = c_longlong

dll.tribal_default_tribal_config.argtypes = []
dll.tribal_default_tribal_config.restype = TribalConfig

