
from omegaconf import OmegaConf

from mettagrid.grid_object cimport GridLocation, Orientation
from mettagrid.action cimport ActionArg
from mettagrid.objects.agent cimport Agent
from mettagrid.objects.metta_object cimport MettaObject
from mettagrid.objects.constants cimport Events, GridLayer, InventoryItem, InventoryItemNames
from mettagrid.objects.converter cimport Converter
from mettagrid.actions.actions cimport MettaActionHandler

cdef class GetOutput(MettaActionHandler):
    def __init__(self, cfg: OmegaConf):
        MettaActionHandler.__init__(self, cfg, "get_output")

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

        # ##Converter_and_HasInventory_are_the_same_thing
        # It's more correct to cast this as a HasInventory, but right now Converters are
        # the only implementors of HasInventory, and we also need to call maybe_start_converting
        # on them. We should later refactor this to we call .update_inventory on the target, and
        # have this automatically call maybe_start_converting. That's hard because we need to
        # let it maybe schedule events.
        cdef Converter *converter = <Converter*> target
        if not converter.inventory_is_accessible():
            return False

        for i in range(InventoryItem.InventoryCount):
            if converter.recipe_output[i] == 0:
                # We only want to take things the converter can produce. Otherwise it's a pain to
                # collect resources from a converter that's in the middle of processing a queue.
                continue
            # The actor will destroy anything it can't hold. That's not intentional, so feel free
            # to fix it.
            actor.stats.add(InventoryItemNames[i], b"get", converter.inventory[i])
            actor.update_inventory(<InventoryItem>i, converter.inventory[i], &self.env._rewards[actor_id])
            converter.update_inventory(<InventoryItem>i, -converter.inventory[i], NULL)

        return True
