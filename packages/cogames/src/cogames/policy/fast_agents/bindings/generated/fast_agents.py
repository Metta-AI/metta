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

class FastAgents(Structure):
    _fields_ = [("ref", c_ulonglong)]

    def __bool__(self):
        return self.ref != None

    def __eq__(self, obj):
        return self.ref == obj.ref

    def __del__(self):
        dll.fast_agents_fast_agents_unref(self)

    def __init__(self, agent_id, environment_config):
        result = dll.fast_agents_new_fast_agents(agent_id, environment_config.encode("utf8"))
        self.ref = result

    @property
    def agent_id(self):
        return dll.fast_agents_fast_agents_get_agent_id(self)

    @agent_id.setter
    def agent_id(self, agent_id):
        dll.fast_agents_fast_agents_set_agent_id(self, agent_id)

    def reset(self):
        dll.fast_agents_fast_agents_reset(self)

    def step(self, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions):
        dll.fast_agents_fast_agents_step(self, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

def init_chook():
    dll.fast_agents_init_chook()

dll.fast_agents_fast_agents_unref.argtypes = [FastAgents]
dll.fast_agents_fast_agents_unref.restype = None

dll.fast_agents_new_fast_agents.argtypes = [c_longlong, c_char_p]
dll.fast_agents_new_fast_agents.restype = c_ulonglong

dll.fast_agents_fast_agents_get_agent_id.argtypes = [FastAgents]
dll.fast_agents_fast_agents_get_agent_id.restype = c_longlong

dll.fast_agents_fast_agents_set_agent_id.argtypes = [FastAgents, c_longlong]
dll.fast_agents_fast_agents_set_agent_id.restype = None

dll.fast_agents_fast_agents_reset.argtypes = [FastAgents]
dll.fast_agents_fast_agents_reset.restype = None

dll.fast_agents_fast_agents_step.argtypes = [FastAgents, c_longlong, c_longlong, c_longlong, c_void_p, c_longlong, c_void_p]
dll.fast_agents_fast_agents_step.restype = None

dll.fast_agents_init_chook.argtypes = []
dll.fast_agents_init_chook.restype = None

