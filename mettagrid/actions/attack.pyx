
from libc.stdio cimport printf

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

        cdef unsigned short shield_damage = 0
        if agent_target:
            actor.stats.incr(self._stats.target[agent_target._type_id])
            actor.stats.incr(self._stats.target[agent_target._type_id], actor.group_name)
            actor.stats.incr(self._stats.target[agent_target._type_id], actor.group_name, agent_target.group_name)

            if agent_target.shield:
                shield_damage = -agent_target.update_energy(-actor.attack_damage, NULL)
                actor.stats.add(b"shield_damage", shield_damage)
            if shield_damage < actor.attack_damage:
                agent_target.shield = False
                agent_target.frozen = agent_target.freeze_duration
                agent_target.update_energy(-agent_target.energy, NULL)

                actor.stats.incr(b"attack.win", actor.group_name)
                actor.stats.incr(b"attack.win", actor.group_name, agent_target.group_name)
                actor.stats.incr(b"attack.loss", agent_target.group_name)
                actor.stats.incr(b"attack.loss", agent_target.group_name, actor.group_name)

                self.env._rewards[actor.agent_id] += agent_target.freeze_reward
                self.env._rewards[agent_target.agent_id] -= agent_target.freeze_reward

                for item in range(InventoryItem.InventoryCount):
                    actor.update_inventory(item, agent_target.inventory[item], &self.env._rewards[actor.agent_id])
                    agent_target.update_inventory(item, -agent_target.inventory[item], &self.env._rewards[agent_target.agent_id])
                    actor.stats.add(InventoryItemNames[item], b"stolen", actor.group_name, agent_target.inventory[item])

            return True

        target_loc.layer = GridLayer.Object_Layer
        cdef MettaObject * object_target = <MettaObject *>self.env._grid.object_at(target_loc)
        if object_target:
            actor.stats.incr(self._stats.target[object_target._type_id])
            actor.stats.incr(self._stats.target[object_target._type_id], actor.group_name)
            object_target.hp -= 1
            actor.stats.incr(b"damage", ObjectTypeNames[object_target._type_id])
            if object_target.hp <= 0:
                self.env._grid.remove_object(object_target)
                actor.stats.incr(b"destroyed", ObjectTypeNames[object_target._type_id])
            return True

        return False
