from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridCoord
from .metta_object cimport ObjectConfig
from .usable cimport Usable
from .agent cimport Agent

cdef extern from "converter.hpp":
    cdef cppclass Converter(Usable):
        short prey_r1_output_energy;
        short predator_r1_output_energy;
        short predator_r2_output_energy;

        Converter(GridCoord r, GridCoord c, ObjectConfig cfg)
        bint usable(const Agent *actor)
        void use(Agent *actor, unsigned int actor_id, float *rewards)
        @staticmethod
        vector[string] feature_names()
