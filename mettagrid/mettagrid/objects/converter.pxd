from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.event cimport EventManager
from mettagrid.grid_object cimport GridCoord, TypeId
from .metta_object cimport ObjectConfig
from .has_inventory cimport HasInventory

cdef extern from "converter.hpp":
    cdef cppclass Converter(HasInventory):
        vector[unsigned char] recipe_input
        vector[unsigned char] recipe_output
        unsigned short max_output
        unsigned char type
        unsigned char conversion_ticks
        bint converting

        Converter(GridCoord r, GridCoord c, ObjectConfig cfg, TypeId type_id)
        void finish_converting()
        void set_event_manager(EventManager *event_manager)

        @staticmethod
        inline vector[string] feature_names(TypeId type_id)
