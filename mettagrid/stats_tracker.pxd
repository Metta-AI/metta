# distutils: language=c++

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string

cdef class StatsTracker:
    cdef:
        map[string, int] _game_stats
        vector[map[string, int]] _agent_stats

    cdef void agent_incr(
        self, unsigned int agent_idx, const char * key_str)
    cdef void game_incr(self, const char * key_str)

    cdef void agent_add(
        self, unsigned int agent_idx, const char * key_str, int value)
    cdef void game_add(self, const char * key_str, int value)

    cpdef to_pydict(self)
