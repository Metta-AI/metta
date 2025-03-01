from libcpp.vector cimport vector
from .metta_object cimport MettaObject, ObjectConfig
from mettagrid.objects.constants cimport InventoryItem

cdef extern from "has_inventory.hpp":
    cdef cppclass HasInventory(MettaObject):
        vector[unsigned char] inventory
        void init_has_inventory(ObjectConfig cfg)
        bint has_inventory()
        bint inventory_is_accessible()
        void update_inventory(InventoryItem item, unsigned char amount, float *reward)

