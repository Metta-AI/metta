from libcpp.vector cimport vector
from libcpp.string cimport string
from mettagrid.grid_object cimport GridObject, ObsType

cdef class ObservationEncoder:
    cdef:
        unsigned int _obs_width
        unsigned int _obs_height
        vector[short] _offsets
        vector[string] _feature_names
        vector[vector[string]] _type_feature_names
    
    cdef init(self, unsigned int obs_width, unsigned int obs_height)
    cdef encode(self, GridObject *obj, ObsType[:] obs)
    cdef _encode(self, GridObject *obj, ObsType[:] obs, unsigned int offset)
    cdef vector[string] feature_names(self)
    cpdef observation_space(self)
    cpdef obs_np_type(self)

cdef class CompactObservationEncoder(ObservationEncoder):
    cdef unsigned int _num_features
