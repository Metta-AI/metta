from mettagrid.action_handler cimport ActionHandler, ActionConfig

cdef extern from "swap.hpp":
    cdef cppclass Swap(ActionHandler):
        Swap(const ActionConfig& cfg)
