
from libc.stdio cimport printf
from libc.string cimport strcat, strcpy

from omegaconf import OmegaConf

from mettagrid.grid_object cimport GridLocation, GridObjectId, Orientation, GridObject
from mettagrid.action cimport ActionHandler, ActionArg
from mettagrid.objects cimport MettaObject, ObjectType, Usable, Altar, Agent, Events, GridLayer
from mettagrid.objects cimport Generator, Converter, InventoryItem, ObjectTypeNames, InventoryItemNames
from mettagrid.actions.actions cimport MettaActionHandler

cdef class Attack(MettaActionHandler):
    def __init__(self, cfg: OmegaConf, action_name: str="attack"):
        MettaActionHandler.__init__(self, cfg, action_name)
        self._priority = 1

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
        
        return self._handle_target(actor_id, actor, target_loc)

    cdef bint _handle_target(
        self,
        unsigned int actor_id,
        Agent * actor,
        GridLocation target_loc):

        target_loc.layer = GridLayer.Agent_Layer
        # Because we're looking on the agent layer, we can cast to Agent.
        cdef Agent * agent_target = <Agent *>self.env._grid.object_at(target_loc)
        cdef char stat_name[256]

        cdef unsigned short shield_damage = 0
        if agent_target:
            self.env._stats.agent_incr(actor_id, self._stats.target[agent_target._type_id].c_str())
            strcpy(stat_name, actor.group_name.c_str())
            strcat(stat_name, ".")
            strcat(stat_name, self._stats.target[agent_target._type_id].c_str())
            self.env._stats.agent_incr(actor_id, stat_name)
            if agent_target.shield:
                shield_damage = -agent_target.update_energy(-actor.attack_damage, NULL)
                self.env._stats.agent_add(actor_id, "shield_damage", shield_damage)
            if shield_damage < actor.attack_damage:
                agent_target.shield = False
                agent_target.frozen = agent_target.freeze_duration
                agent_target.update_energy(-agent_target.energy, NULL)
                self.env._stats.agent_incr(actor_id, "attack.frozen")
                strcpy(stat_name, actor.group_name.c_str())
                strcat(stat_name, ".attack.frozen")
                self.env._stats.agent_incr(actor_id, stat_name)
                strcpy(stat_name, actor.group_name.c_str())
                strcat(stat_name, ".attack.frozen")
                self.env._stats.agent_incr(actor_id, stat_name)
                for item in range(InventoryItem.InventoryCount):
                    actor.update_inventory(item, agent_target.inventory[item])
                    strcpy(stat_name, actor.group_name.c_str())
                    strcat(stat_name, ".")
                    strcat(stat_name, InventoryItemNames[item].c_str())
                    strcat(stat_name, ".stolen")
                    self.env._stats.agent_add(actor_id, stat_name, agent_target.inventory[item])
                    strcpy(stat_name, actor.group_name.c_str())
                    strcat(stat_name, ".")
                    strcat(stat_name, InventoryItemNames[item].c_str())
                    strcat(stat_name, ".gained")
                    self.env._stats.agent_add(actor_id, stat_name, agent_target.inventory[item])
                    agent_target.inventory[item] = 0

            return True

        target_loc.layer = GridLayer.Object_Layer
        cdef MettaObject * object_target = <MettaObject *>self.env._grid.object_at(target_loc)
        if object_target:
            self.env._stats.agent_incr(actor_id, self._stats.target[object_target._type_id].c_str())
            strcpy(stat_name, actor.group_name.c_str())
            strcat(stat_name, ".")
            strcat(stat_name, self._stats.target[object_target._type_id].c_str())
            self.env._stats.agent_incr(actor_id, stat_name)
            object_target.hp -= 1
            strcpy(stat_name, actor.group_name.c_str())
            strcat(stat_name, ".damage.")
            strcat(stat_name, ObjectTypeNames[object_target._type_id].c_str())
            self.env._stats.agent_incr(actor_id, stat_name)
            if object_target.hp <= 0:
                self.env._grid.remove_object(object_target)
                strcpy(stat_name, actor.group_name.c_str())
                strcat(stat_name, ".destroyed.")
                strcat(stat_name, ObjectTypeNames[object_target._type_id].c_str())
                self.env._stats.agent_incr(actor_id, stat_name)
            return True

        return False
