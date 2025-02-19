# distutils: language=c++
# cython: warn.undeclared=False

from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord, GridLocation
from mettagrid.objects.usable cimport Usable
from mettagrid.objects.agent cimport Agent
from mettagrid.objects.metta_object cimport ObjectConfig
from mettagrid.observation_encoder cimport ObsType
cdef extern from "armory.hpp":
    cdef cppclass Armory(Usable):
        Armory(GridCoord r, GridCoord c, ObjectConfig cfg) except +
        bint usable(const Agent *actor)
        void use(Agent *actor, float *rewards)

        void obs(ObsType *obs)

        @staticmethod
        inline vector[string] feature_names()
