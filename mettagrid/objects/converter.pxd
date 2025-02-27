from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord
from .metta_object cimport ObjectConfig
from .has_inventory cimport HasInventory

cdef extern from "converter.hpp":
    cdef cppclass Converter(HasInventory):
        vector[unsigned char] recipe_input
        vector[unsigned char] recipe_output
        unsigned short max_output
        unsigned char type
        unsigned char recipe_duration
        bint converting

        Converter(GridCoord r, GridCoord c, ObjectConfig cfg)
        bint maybe_start_converting()
        void finish_converting()
        # @staticmethod
        # vector[string] feature_names()
