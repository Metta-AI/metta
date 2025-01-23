# distutils: language=c++

from libcpp.vector cimport vector
from libcpp.string cimport string

from mettagrid.grid_object cimport GridObject

ctypedef unsigned char ObsType

cdef class ObservationEncoder:
    cdef:
        unsigned int _obs_width
        unsigned int _obs_height

    cdef init(self, unsigned int obs_width, unsigned int obs_height)
    cdef encode(self, const GridObject *obj, ObsType[:] obs)
    cdef vector[string] feature_names(self)

    cpdef observation_space(self)
    cpdef obs_np_type(self)
