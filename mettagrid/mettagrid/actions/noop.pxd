from mettagrid.action_handler cimport ActionHandler, ActionConfig

cdef extern from "noop.hpp":
    cdef cppclass Noop(ActionHandler):
        Noop(const ActionConfig& cfg)
