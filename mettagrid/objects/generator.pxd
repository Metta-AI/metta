from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord, GridLocation, GridObject
from mettagrid.observation_encoder cimport ObsType
from .constants cimport GridLayer, ObjectType
from .metta_object cimport MettaObject, ObjectConfig
from .usable cimport Usable


cdef extern from "generator.hpp":
    cdef cppclass Generator(Usable):
        Generator(GridCoord r, GridCoord c, ObjectConfig cfg)

        void obs(ObsType *obs)

        @staticmethod
        vector[string] feature_names()
