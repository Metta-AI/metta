from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord
from mettagrid.objects.usable cimport Usable
from mettagrid.objects.metta_object cimport ObjectConfig
from mettagrid.observation_encoder cimport ObsType

cdef extern from "factory.hpp":
    cdef cppclass Factory(Usable):
        Factory(GridCoord r, GridCoord c, ObjectConfig cfg) except +

        @staticmethod
        vector[string] feature_names()
