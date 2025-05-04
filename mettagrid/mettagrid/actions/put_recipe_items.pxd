from mettagrid.action_handler cimport ActionHandler, ActionConfig

cdef extern from "put_recipe_items.hpp":
    cdef cppclass PutRecipeItems(ActionHandler):
        PutRecipeItems(const ActionConfig& cfg)
