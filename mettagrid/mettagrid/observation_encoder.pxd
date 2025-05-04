from libcpp.string cimport string
from libcpp.vector cimport vector

from mettagrid.grid_object cimport GridObject, ObsType

cdef extern from "observation_encoder.hpp":
    cdef cppclass ObservationEncoder:
        ObservationEncoder() except +
        void encode(const GridObject* obj, ObsType* obs)
        void encode(const GridObject* obj, ObsType* obs, const vector[unsigned int]& offsets)
        const vector[string]& feature_names()
        const vector[vector[string]]& type_feature_names()

