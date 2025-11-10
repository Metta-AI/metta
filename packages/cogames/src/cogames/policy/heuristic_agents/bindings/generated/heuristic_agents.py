from ctypes import *
import os, sys

dir = os.path.dirname(sys.modules["heuristic_agents"].__file__)
if sys.platform == "win32":
  libName = "heuristic_agents.dll"
elif sys.platform == "darwin":
  libName = "libheuristic_agents.dylib"
else:
  libName = "libheuristic_agents.so"
dll = cdll.LoadLibrary(os.path.join(dir, libName))

class HeuristicAgentsError(Exception):
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

class HeuristicAgent(Structure):
    _fields_ = [("ref", c_ulonglong)]

    def __bool__(self):
        return self.ref != None

    def __eq__(self, obj):
        return self.ref == obj.ref

    def __del__(self):
        dll.heuristic_agents_heuristic_agent_unref(self)

    def __init__(self, agent_id, environment_config):
        result = dll.heuristic_agents_new_heuristic_agent(agent_id, environment_config.encode("utf8"))
        self.ref = result

    @property
    def agent_id(self):
        return dll.heuristic_agents_heuristic_agent_get_agent_id(self)

    @agent_id.setter
    def agent_id(self, agent_id):
        dll.heuristic_agents_heuristic_agent_set_agent_id(self, agent_id)

    def reset(self):
        dll.heuristic_agents_heuristic_agent_reset(self)

    def step(self, num_agents, num_tokens, size_token, row_observations, num_actions, raw_actions):
        dll.heuristic_agents_heuristic_agent_step(self, num_agents, num_tokens, size_token, row_observations, num_actions, raw_actions)

def init_chook():
    dll.heuristic_agents_init_chook()

dll.heuristic_agents_heuristic_agent_unref.argtypes = [HeuristicAgent]
dll.heuristic_agents_heuristic_agent_unref.restype = None

dll.heuristic_agents_new_heuristic_agent.argtypes = [c_longlong, c_char_p]
dll.heuristic_agents_new_heuristic_agent.restype = c_ulonglong

dll.heuristic_agents_heuristic_agent_get_agent_id.argtypes = [HeuristicAgent]
dll.heuristic_agents_heuristic_agent_get_agent_id.restype = c_longlong

dll.heuristic_agents_heuristic_agent_set_agent_id.argtypes = [HeuristicAgent, c_longlong]
dll.heuristic_agents_heuristic_agent_set_agent_id.restype = None

dll.heuristic_agents_heuristic_agent_reset.argtypes = [HeuristicAgent]
dll.heuristic_agents_heuristic_agent_reset.restype = None

dll.heuristic_agents_heuristic_agent_step.argtypes = [HeuristicAgent, c_longlong, c_longlong, c_longlong, pointer, c_longlong, pointer]
dll.heuristic_agents_heuristic_agent_step.restype = None

dll.heuristic_agents_init_chook.argtypes = []
dll.heuristic_agents_init_chook.restype = None

