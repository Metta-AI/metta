
from omegaconf import OmegaConf

from mettagrid.grid_object cimport GridLocation, Orientation
from mettagrid.action cimport ActionArg
from mettagrid.objects.agent cimport Agent
from mettagrid.objects.metta_object cimport MettaObject
from mettagrid.objects.constants cimport Events, GridLayer, InventoryItem, InventoryItemNames
from mettagrid.objects.converter cimport Converter
from mettagrid.actions.actions cimport MettaActionHandler

# Puts one recipe worth of resources into a Converter. Noop if not enough resources.
cdef class PutRecipeItems(MettaActionHandler):
    def __init__(self, cfg: OmegaConf):
        MettaActionHandler.__init__(self, cfg, "put_recipe_items")

    cdef unsigned char max_arg(self):
        return 0

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        cdef GridLocation target_loc = self.env._grid.relative_location(
            actor.location,
            <Orientation>actor.orientation
        )
        target_loc.layer = GridLayer.Object_Layer
        cdef MettaObject *target = <MettaObject*>self.env._grid.object_at(target_loc)
        if target == NULL or not target.has_inventory():
            return False

        # #Converter_and_HasInventory_are_the_same_thing
        cdef Converter *converter = <Converter*> target

        for i in range(converter.recipe_input.size()):
            if converter.recipe_input[i] > actor.inventory[i]:
                return False

        for i in range(converter.recipe_input.size()):
            actor.update_inventory(<InventoryItem>i, -converter.recipe_input[i], &self.env._rewards[actor_id])
            converter.update_inventory(<InventoryItem>i, converter.recipe_input[i], NULL)
            actor.stats.add(InventoryItemNames[i], b"put", converter.recipe_input[i]);

        return True
