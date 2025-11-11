from ctypes import *
import os, sys

dir = os.path.dirname(sys.modules["fast_agents"].__file__)
if sys.platform == "win32":
  libName = "fast_agents.dll"
elif sys.platform == "darwin":
  libName = "libfast_agents.dylib"
else:
  libName = "libfast_agents.so"
dll = cdll.LoadLibrary(os.path.join(dir, libName))

class FastAgentsError(Exception):
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

def init_chook():
    dll.fast_agents_init_chook()

class RandomAgent(Structure):
    _fields_ = [("ref", c_ulonglong)]

    def __bool__(self):
        return self.ref != None

    def __eq__(self, obj):
        return self.ref == obj.ref

    def __del__(self):
        dll.fast_agents_random_agent_unref(self)

    def __init__(self, agent_id, environment_config):
        result = dll.fast_agents_new_random_agent(agent_id, environment_config.encode("utf8"))
        self.ref = result

    @property
    def agent_id(self):
        return dll.fast_agents_random_agent_get_agent_id(self)

    @agent_id.setter
    def agent_id(self, agent_id):
        dll.fast_agents_random_agent_set_agent_id(self, agent_id)

    def reset(self):
        dll.fast_agents_random_agent_reset(self)

    def step(self, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions):
        dll.fast_agents_random_agent_step(self, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

class ThinkyAgent(Structure):
    _fields_ = [("ref", c_ulonglong)]

    def __bool__(self):
        return self.ref != None

    def __eq__(self, obj):
        return self.ref == obj.ref

    def __del__(self):
        dll.fast_agents_thinky_agent_unref(self)

    def __init__(self, agent_id, environment_config):
        result = dll.fast_agents_new_thinky_agent(agent_id, environment_config.encode("utf8"))
        self.ref = result

    @property
    def agent_id(self):
        return dll.fast_agents_thinky_agent_get_agent_id(self)

    @agent_id.setter
    def agent_id(self, agent_id):
        dll.fast_agents_thinky_agent_set_agent_id(self, agent_id)

    def reset(self):
        dll.fast_agents_thinky_agent_reset(self)

    def step(self, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions):
        dll.fast_agents_thinky_agent_step(self, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

class RaceCarAgent(Structure):
    _fields_ = [("ref", c_ulonglong)]

    def __bool__(self):
        return self.ref != None

    def __eq__(self, obj):
        return self.ref == obj.ref

    def __del__(self):
        dll.fast_agents_race_car_agent_unref(self)

    def __init__(self, agent_id, environment_config):
        result = dll.fast_agents_new_race_car_agent(agent_id, environment_config.encode("utf8"))
        self.ref = result

    @property
    def agent_id(self):
        return dll.fast_agents_race_car_agent_get_agent_id(self)

    @agent_id.setter
    def agent_id(self, agent_id):
        dll.fast_agents_race_car_agent_set_agent_id(self, agent_id)

    def reset(self):
        dll.fast_agents_race_car_agent_reset(self)

    def step(self, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions):
        dll.fast_agents_race_car_agent_step(self, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

dll.fast_agents_init_chook.argtypes = []
dll.fast_agents_init_chook.restype = None

dll.fast_agents_random_agent_unref.argtypes = [RandomAgent]
dll.fast_agents_random_agent_unref.restype = None

dll.fast_agents_new_random_agent.argtypes = [c_longlong, c_char_p]
dll.fast_agents_new_random_agent.restype = c_ulonglong

dll.fast_agents_random_agent_get_agent_id.argtypes = [RandomAgent]
dll.fast_agents_random_agent_get_agent_id.restype = c_longlong

dll.fast_agents_random_agent_set_agent_id.argtypes = [RandomAgent, c_longlong]
dll.fast_agents_random_agent_set_agent_id.restype = None

dll.fast_agents_random_agent_reset.argtypes = [RandomAgent]
dll.fast_agents_random_agent_reset.restype = None

dll.fast_agents_random_agent_step.argtypes = [RandomAgent, c_longlong, c_longlong, c_longlong, pointer, c_longlong, pointer]
dll.fast_agents_random_agent_step.restype = None

dll.fast_agents_thinky_agent_unref.argtypes = [ThinkyAgent]
dll.fast_agents_thinky_agent_unref.restype = None

dll.fast_agents_new_thinky_agent.argtypes = [c_longlong, c_char_p]
dll.fast_agents_new_thinky_agent.restype = c_ulonglong

dll.fast_agents_thinky_agent_get_agent_id.argtypes = [ThinkyAgent]
dll.fast_agents_thinky_agent_get_agent_id.restype = c_longlong

dll.fast_agents_thinky_agent_set_agent_id.argtypes = [ThinkyAgent, c_longlong]
dll.fast_agents_thinky_agent_set_agent_id.restype = None

dll.fast_agents_thinky_agent_reset.argtypes = [ThinkyAgent]
dll.fast_agents_thinky_agent_reset.restype = None

dll.fast_agents_thinky_agent_step.argtypes = [ThinkyAgent, c_longlong, c_longlong, c_longlong, pointer, c_longlong, pointer]
dll.fast_agents_thinky_agent_step.restype = None

dll.fast_agents_race_car_agent_unref.argtypes = [RaceCarAgent]
dll.fast_agents_race_car_agent_unref.restype = None

dll.fast_agents_new_race_car_agent.argtypes = [c_longlong, c_char_p]
dll.fast_agents_new_race_car_agent.restype = c_ulonglong

dll.fast_agents_race_car_agent_get_agent_id.argtypes = [RaceCarAgent]
dll.fast_agents_race_car_agent_get_agent_id.restype = c_longlong

dll.fast_agents_race_car_agent_set_agent_id.argtypes = [RaceCarAgent, c_longlong]
dll.fast_agents_race_car_agent_set_agent_id.restype = None

dll.fast_agents_race_car_agent_reset.argtypes = [RaceCarAgent]
dll.fast_agents_race_car_agent_reset.restype = None

dll.fast_agents_race_car_agent_step.argtypes = [RaceCarAgent, c_longlong, c_longlong, c_longlong, pointer, c_longlong, pointer]
dll.fast_agents_race_car_agent_step.restype = None

