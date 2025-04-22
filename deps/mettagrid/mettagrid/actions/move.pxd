from mettagrid.action_handler cimport ActionHandler, ActionConfig

cdef extern from "move.hpp":
    cdef cppclass Move(ActionHandler):
        Move(const ActionConfig& cfg)
