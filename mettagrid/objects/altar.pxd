from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord, GridLocation, GridObject
from mettagrid.observation_encoder cimport ObsType
from .metta_object cimport MettaObject, ObjectConfig
from .usable cimport Usable
from .constants cimport GridLayer, ObjectType

cdef extern from "altar.hpp":
    cdef cppclass Altar(Usable):
        Altar(GridCoord r, GridCoord c, ObjectConfig cfg)

        void obs(ObsType *obs)

        @staticmethod
        vector[string] feature_names()
