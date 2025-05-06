# Re-export the PyMettaGrid as MettaGrid to maintain compatibility with existing code
from mettagrid.env import PyMettaGrid as MettaGrid

# Export other useful components directly
from mettagrid.objects.constants import InventoryItemNames, ObjectLayers, ObjectTypeNames
from mettagrid.resolvers import register_resolvers

__all__ = ["MettaGrid", "ObjectTypeNames", "InventoryItemNames", "ObjectLayers", "register_resolvers"]
