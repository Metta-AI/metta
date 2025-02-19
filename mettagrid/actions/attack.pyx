from libc.stdio cimport printf

from omegaconf import OmegaConf

from mettagrid.action cimport ActionArg
from mettagrid.actions.actions cimport MettaActionHandler
from mettagrid.grid_object cimport GridLocation, Orientation
from mettagrid.objects.agent cimport Agent
from mettagrid.objects.constants cimport (
    GridLayer,
    InventoryItem,
    InventoryItemNames,
    ObjectTypeNames
)
from mettagrid.objects.metta_object cimport MettaObject

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

        if actor.inventory[InventoryItem.laser] == 0:
            return False

        actor.update_inventory(InventoryItem.laser, -1, &self.env._rewards[actor_id])

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

        cdef bint was_frozen = False
        if agent_target:
            actor.stats.incr(self._stats.target[agent_target._type_id])
            actor.stats.incr(self._stats.target[agent_target._type_id], actor.group_name)
            actor.stats.incr(self._stats.target[agent_target._type_id], actor.group_name, agent_target.group_name)

            if agent_target.group_name == actor.group_name:
                actor.stats.incr(b"attack.own_team", actor.group_name)
            else:
                actor.stats.incr(b"attack.other_team", actor.group_name)

            was_frozen = agent_target.frozen > 0

            if agent_target.inventory[InventoryItem.armor] > 0:
                agent_target.update_inventory(InventoryItem.armor, -1, &self.env._rewards[agent_target.agent_id])
                actor.stats.incr(b"attack.blocked", agent_target.group_name)
                actor.stats.incr(b"attack.blocked", agent_target.group_name, actor.group_name)
            else:
                agent_target.frozen = agent_target.freeze_duration

                if not was_frozen:
                    actor.stats.incr(b"attack.win", actor.group_name)
                    actor.stats.incr(b"attack.win", actor.group_name, agent_target.group_name)
                    actor.stats.incr(b"attack.loss", agent_target.group_name)
                    actor.stats.incr(b"attack.loss", agent_target.group_name, actor.group_name)

                    if agent_target.group_name == actor.group_name:
                        actor.stats.incr(b"attack.win.own_team", actor.group_name)
                    else:
                        actor.stats.incr(b"attack.win.other_team", actor.group_name)

                    for item in range(InventoryItem.InventoryCount):
                        actor.stats.add(InventoryItemNames[item], b"stolen", actor.group_name, agent_target.inventory[item])
                        actor.update_inventory(item, agent_target.inventory[item], &self.env._rewards[actor.agent_id])
                        agent_target.update_inventory(item, -agent_target.inventory[item], &self.env._rewards[agent_target.agent_id])

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
