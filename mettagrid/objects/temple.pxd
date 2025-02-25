from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord
from mettagrid.objects.usable cimport Usable
from mettagrid.objects.metta_object cimport ObjectConfig

cdef extern from "temple.hpp":
    cdef cppclass Temple(Usable):
        Temple(GridCoord r, GridCoord c, ObjectConfig cfg) except +

        @staticmethod
        vector[string] feature_names()

