from mettagrid.action_handler cimport ActionHandler, ActionConfig

cdef extern from "attack.hpp":
    cdef cppclass Attack(ActionHandler):
        Attack(const ActionConfig& cfg)
