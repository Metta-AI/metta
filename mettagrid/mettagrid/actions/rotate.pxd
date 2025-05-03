from mettagrid.action_handler cimport ActionHandler, ActionConfig

cdef extern from "rotate.hpp":
    cdef cppclass Rotate(ActionHandler):
        Rotate(const ActionConfig& cfg)
