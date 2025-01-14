# distutils: language=c++
# cython: warn.undeclared=False
# cython: c_api_binop_methods=True

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from mettagrid.grid_env import StatsTracker
from libc.stdio cimport printf
from mettagrid.base_encoder cimport ObservationEncoder, ObsType
from mettagrid.grid_object cimport GridObject, TypeId, GridCoord, GridLocation, GridObjectId
from mettagrid.event cimport EventHandler, EventArg

cdef class MettaObservationEncoder(ObservationEncoder):
    cdef _encode(self, GridObject *obj, ObsType[:] obs, unsigned int offset)
    cdef vector[short] _offsets
    cdef vector[string] _feature_names
    cdef vector[vector[string]] _type_feature_names

cdef class MettaCompactObservationEncoder(MettaObservationEncoder):
    cdef unsigned int _num_features
