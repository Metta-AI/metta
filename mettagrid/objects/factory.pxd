from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord
from mettagrid.objects.metta_object cimport ObjectConfig
from mettagrid.objects.converter cimport Converter

cdef extern from "factory.hpp":
    cdef cppclass Factory(Converter):
        Factory(GridCoord r, GridCoord c, ObjectConfig cfg) except +

        @staticmethod
        vector[string] feature_names()

