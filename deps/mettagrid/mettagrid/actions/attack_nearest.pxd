from mettagrid.actions.attack cimport Attack, ActionConfig

cdef extern from "attack_nearest.hpp":
    cdef cppclass AttackNearest(Attack):
        AttackNearest(const ActionConfig& cfg)
