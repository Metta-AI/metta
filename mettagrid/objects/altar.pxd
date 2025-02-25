from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord
from mettagrid.observation_encoder cimport ObsType
from .metta_object cimport ObjectConfig
from .usable cimport Usable

cdef extern from "altar.hpp":
    cdef cppclass Altar(Usable):
        Altar(GridCoord r, GridCoord c, ObjectConfig cfg) except +

        void obs(ObsType *obs)

        @staticmethod
        vector[string] feature_names()
