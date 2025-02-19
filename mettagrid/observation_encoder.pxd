from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdio cimport printf
from mettagrid.base_encoder cimport ObservationEncoder, ObsType
from mettagrid.grid_object cimport GridObject

cdef class MettaObservationEncoder(ObservationEncoder):
    cdef _encode(self, GridObject *obj, ObsType[:] obs, unsigned int offset)
    cdef vector[short] _offsets
    cdef vector[string] _feature_names
    cdef vector[vector[string]] _type_feature_names

cdef class MettaCompactObservationEncoder(MettaObservationEncoder):
    cdef unsigned int _num_features
