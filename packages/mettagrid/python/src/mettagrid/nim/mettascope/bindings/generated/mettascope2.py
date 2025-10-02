from ctypes import *
import os, sys

dir = os.path.dirname(sys.modules["mettascope2"].__file__)
if sys.platform == "win32":
  libName = "mettascope2.dll"
elif sys.platform == "darwin":
  libName = "libmettascope2.dylib"
else:
  libName = "libmettascope2.so"
dll = cdll.LoadLibrary(os.path.join(dir, libName))

class Mettascope2Error(Exception):
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

class ActionRequest(Structure):
    _fields_ = [
        ("agent_id", c_longlong),
        ("action_id", c_longlong),
        ("argument", c_longlong)
    ]

    def __init__(self, agent_id, action_id, argument):
        self.agent_id = agent_id
        self.action_id = action_id
        self.argument = argument

    def __eq__(self, obj):
        return self.agent_id == obj.agent_id and self.action_id == obj.action_id and self.argument == obj.argument

class RenderResponse(Structure):
    _fields_ = [("ref", c_ulonglong)]

    def __bool__(self):
        return self.ref != None

    def __eq__(self, obj):
        return self.ref == obj.ref

    def __del__(self):
        dll.mettascope2_render_response_unref(self)

    @property
    def should_close(self):
        return dll.mettascope2_render_response_get_should_close(self)

    @should_close.setter
    def should_close(self, should_close):
        dll.mettascope2_render_response_set_should_close(self, should_close)

    class RenderResponseActions:

        def __init__(self, render_response):
            self.render_response = render_response

        def __len__(self):
            return dll.mettascope2_render_response_actions_len(self.render_response)

        def __getitem__(self, index):
            return dll.mettascope2_render_response_actions_get(self.render_response, index)

        def __setitem__(self, index, value):
            dll.mettascope2_render_response_actions_set(self.render_response, index, value)

        def __delitem__(self, index):
            dll.mettascope2_render_response_actions_delete(self.render_response, index)

        def append(self, value):
            dll.mettascope2_render_response_actions_add(self.render_response, value)

        def clear(self):
            dll.mettascope2_render_response_actions_clear(self.render_response)

        def __iter__(self):
            return SeqIterator(self)

    @property
    def actions(self):
        return self.RenderResponseActions(self)

def init(data_dir, replay):
    result = dll.mettascope2_init(data_dir.encode("utf8"), replay.encode("utf8"))
    return result

def render(current_step, replay_step):
    result = dll.mettascope2_render(current_step, replay_step.encode("utf8"))
    return result

dll.mettascope2_render_response_unref.argtypes = [RenderResponse]
dll.mettascope2_render_response_unref.restype = None

dll.mettascope2_render_response_get_should_close.argtypes = [RenderResponse]
dll.mettascope2_render_response_get_should_close.restype = c_bool

dll.mettascope2_render_response_set_should_close.argtypes = [RenderResponse, c_bool]
dll.mettascope2_render_response_set_should_close.restype = None

dll.mettascope2_render_response_actions_len.argtypes = [RenderResponse]
dll.mettascope2_render_response_actions_len.restype = c_longlong

dll.mettascope2_render_response_actions_get.argtypes = [RenderResponse, c_longlong]
dll.mettascope2_render_response_actions_get.restype = ActionRequest

dll.mettascope2_render_response_actions_set.argtypes = [RenderResponse, c_longlong, ActionRequest]
dll.mettascope2_render_response_actions_set.restype = None

dll.mettascope2_render_response_actions_delete.argtypes = [RenderResponse, c_longlong]
dll.mettascope2_render_response_actions_delete.restype = None

dll.mettascope2_render_response_actions_add.argtypes = [RenderResponse, ActionRequest]
dll.mettascope2_render_response_actions_add.restype = None

dll.mettascope2_render_response_actions_clear.argtypes = [RenderResponse]
dll.mettascope2_render_response_actions_clear.restype = None

dll.mettascope2_init.argtypes = [c_char_p, c_char_p]
dll.mettascope2_init.restype = RenderResponse

dll.mettascope2_render.argtypes = [c_longlong, c_char_p]
dll.mettascope2_render.restype = RenderResponse

