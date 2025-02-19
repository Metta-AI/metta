# distutils: language=c++
# cython: warn.undeclared=False

from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord, GridLocation
from mettagrid.objects.usable cimport Usable
from mettagrid.objects.agent cimport Agent
from mettagrid.objects.metta_object cimport ObjectConfig
from mettagrid.observation_encoder cimport ObsType

cdef extern from "factory.hpp":
    cdef cppclass Factory(Usable):
        Factory(GridCoord r, GridCoord c, ObjectConfig cfg) except +
        bint usable(const Agent *actor)
        void use(Agent *actor, float *rewards)

        void obs(ObsType *obs)

        @staticmethod
        vector[string] feature_names()
