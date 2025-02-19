from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord, GridLocation, GridObject
from mettagrid.observation_encoder cimport ObsType
from mettagrid.objects.constants cimport GridLayer, ObjectType
from mettagrid.objects.metta_object cimport MettaObject, ObjectConfig

cdef extern from "wall.hpp":
    cdef cppclass Wall(MettaObject):
        Wall(GridCoord r, GridCoord c, ObjectConfig cfg)

        void obs(ObsType *obs)

        @staticmethod
        vector[string] feature_names()
