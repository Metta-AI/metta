# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport uint8_t, uint32_t

cdef extern from "cpp_processor.hpp":
    cdef cppclass ArrayProcessor:
        ArrayProcessor(uint32_t size) except +
        void process_array(uint8_t* data, uint32_t size) except +
        uint8_t* get_results() const
        uint32_t get_size() const