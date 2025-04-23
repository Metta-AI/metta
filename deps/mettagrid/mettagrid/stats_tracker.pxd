from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string

cdef extern from "stats_tracker.hpp":
    cdef cppclass StatsTracker:
        map[string, int] _stats

        StatsTracker()

        void incr(const string& key)
        void incr(const string& key1, const string& key2)
        void incr(const string& key1, const string& key2, const string& key3)
        void incr(const string& key1, const string& key2, const string& key3, const string& key4)

        void add(const string& key, int value)
        void add(const string& key1, const string& key2, int value)
        void add(const string& key1, const string& key2, const string& key3, int value)
        void add(const string& key1, const string& key2, const string& key3, const string& key4, int value)

        void set_once(const string& key, int value)
        void set_once(const string& key1, const string& key2, int value)
        void set_once(const string& key1, const string& key2, const string& key3, int value)
        void set_once(const string& key1, const string& key2, const string& key3, const string& key4, int value)

        map[string, int] stats()
