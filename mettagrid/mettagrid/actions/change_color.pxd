from mettagrid.action_handler cimport ActionHandler, ActionConfig

cdef extern from "change_color.hpp":
    cdef cppclass ChangeColorAction(ActionHandler):
        ChangeColorAction(const ActionConfig& cfg)
