from libcpp.vector cimport vector
from .metta_object cimport MettaObject, ObjectConfig

cdef extern from "has_inventory.hpp":
    cdef cppclass HasInventory(MettaObject):
        vector[unsigned char] inventory
        void init_has_inventory(ObjectConfig cfg)
        bint has_inventory()
        bint inventory_is_accessible()

