
from libc.stdio cimport printf
from omegaconf import OmegaConf

from mettagrid.grid_object cimport GridLocation, GridObjectId, Orientation, GridObject
from mettagrid.action cimport ActionHandler, ActionArg
from mettagrid.objects cimport MettaObject, ObjectType, Usable, Altar, Agent, Events, GridLayer
from mettagrid.objects cimport Generator, Converter, InventoryItem, ObjectTypeNames, InventoryItemNames
from mettagrid.actions.actions cimport MettaActionHandler

cdef class Attack(MettaActionHandler):
    def __init__(self, cfg: OmegaConf):
        MettaActionHandler.__init__(self, cfg, "attack")

    cdef unsigned char max_arg(self):
        return 9

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        if arg > 9 or arg < 1:
            return False

        cdef short distance = 0
        cdef short offset = 0
        distance = 1 + (arg - 1) // 3
        offset = -((arg - 1) % 3 - 1)

        cdef GridLocation target_loc = self.env._grid.relative_location(
            actor.location,
            <Orientation>actor.orientation,
            distance, offset)

        target_loc.layer = GridLayer.Agent_Layer
        cdef Agent * agent_target = <Agent *>self.env._grid.object_at(target_loc)

        cdef unsigned short shield_damage = 0
        if agent_target:
            self.env._stats.agent_incr(actor_id, self._stats.target[agent_target._type_id].c_str())
            if agent_target.shield:
                shield_damage = -agent_target.update_energy(-actor.attack_damage, NULL)
                self.env._stats.agent_add(actor_id, "shield_damage", shield_damage)
                self.env._stats.agent_add(actor_id, "." + actor.species_name + ".shield_damage", shield_damage)
            if shield_damage < actor.attack_damage:
                agent_target.shield = False
                agent_target.frozen = agent_target.freeze_duration
                agent_target.update_energy(-agent_target.energy, NULL)
                self.env._stats.agent_incr(actor_id, "attack.frozen")
                self.env._stats.agent_incr(actor_id, "." + actor.species_name + ".attack.frozen")
                for item in range(InventoryItem.InventoryCount):
                    actor.update_inventory(item, agent_target.inventory[item])
                    self.env._stats.agent_add(actor_id, InventoryItemNames[item] + ".stolen", agent_target.inventory[item])
                    self.env._stats.agent_add(actor_id, "." + actor.species_name + "." + InventoryItemNames[item] + ".stolen", agent_target.inventory[item])
                    self.env._stats.agent_add(actor_id, InventoryItemNames[item] + ".gained", agent_target.inventory[item])
                    self.env._stats.agent_add(actor_id, "." + actor.species_name + "." + InventoryItemNames[item] + ".gained", agent_target.inventory[item])
                    agent_target.inventory[item] = 0

            return True

        target_loc.layer = GridLayer.Object_Layer
        cdef MettaObject * object_target = <MettaObject *>self.env._grid.object_at(target_loc)
        if object_target:
            self.env._stats.agent_incr(actor_id, self._stats.target[object_target._type_id].c_str())
            self.env._stats.agent_incr(actor_id, "." + actor.species_name + "." + self._stats.target[object_target._type_id].c_str())
            object_target.hp -= 1
            self.env._stats.agent_incr(actor_id, "damage." + ObjectTypeNames[object_target._type_id])
            self.env._stats.agent_incr(actor_id, "." + actor.species_name + ".damage." + ObjectTypeNames[object_target._type_id])
            if object_target.hp <= 0:
                self.env._grid.remove_object(object_target)
                self.env._stats.agent_incr(actor_id, "destroyed." + ObjectTypeNames[object_target._type_id])
                self.env._stats.agent_incr(actor_id, "." + actor.species_name + ".destroyed." + ObjectTypeNames[object_target._type_id])
            return True

        return False
