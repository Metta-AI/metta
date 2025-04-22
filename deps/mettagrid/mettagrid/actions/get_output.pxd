from mettagrid.action_handler cimport ActionHandler, ActionConfig

cdef extern from "get_output.hpp":
    cdef cppclass GetOutput(ActionHandler):
        GetOutput(const ActionConfig& cfg)
