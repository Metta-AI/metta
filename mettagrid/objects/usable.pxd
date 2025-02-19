from mettagrid.objects.metta_object cimport MettaObject, ObjectConfig
from mettagrid.objects.agent cimport Agent

cdef extern from "usable.hpp":
    cdef cppclass Usable(MettaObject):
        unsigned int use_cost
        unsigned int cooldown
        unsigned char ready

        void init_usable(ObjectConfig cfg)
        bint usable(const Agent *actor)
        void use(Agent *actor, float *rewards)
        bint is_usable_type()
