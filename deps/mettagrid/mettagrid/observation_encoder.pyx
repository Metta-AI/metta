from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map

from mettagrid.grid_object cimport GridObject, ObsType

from mettagrid.objects.constants cimport ObjectType

from mettagrid.objects.agent cimport Agent
from mettagrid.objects.converter cimport Converter
from mettagrid.objects.wall cimport Wall

cdef class ObservationEncoder:
    cdef init(self, unsigned int obs_width, unsigned int obs_height):
        self._obs_width = obs_width
        self._obs_height = obs_height

    def __init__(self) -> None:
        self._offsets.resize(ObjectType.Count)
        self._type_feature_names.resize(ObjectType.Count)

        self._type_feature_names[ObjectType.AgentT] = Agent.feature_names()
        self._type_feature_names[ObjectType.WallT] = Wall.feature_names()

        # These are different types of Converters. The only difference in the feature names
        # is the 1-hot that they use for their type. We're working to simplify this, so we can
        # remove these types from code.
        for type_id in [ObjectType.AltarT, ObjectType.ArmoryT, ObjectType.FactoryT, ObjectType.GeneratorT, ObjectType.LabT, ObjectType.LaseryT, ObjectType.MineT, ObjectType.TempleT]:
            self._type_feature_names[type_id] = Converter.feature_names(type_id)
        
        # Generate an offset for each unique feature name.
        cdef map[string, int] features
        cdef vector[string] feature_names
        for type_id in range(ObjectType.Count):
            for i in range(len(self._type_feature_names[type_id])):
                if features.count(self._type_feature_names[type_id][i]) == 0:
                    features[self._type_feature_names[type_id][i]] = features.size()
                    feature_names.push_back(self._type_feature_names[type_id][i])
        # Set the offset for each feature, using the global offsets.
        for type_id in range(ObjectType.Count):
            for i in range(len(self._type_feature_names[type_id])):
                self._offsets[type_id].push_back(features[self._type_feature_names[type_id][i]])

        self._feature_names = feature_names

    cdef encode(self, GridObject *obj, ObsType[:] obs):
        self._encode(obj, obs, self._offsets[obj._type_id])

    cdef _encode(self, GridObject *obj, ObsType[:] obs, vector[unsigned int] offsets):
        obj.obs(&obs[0], offsets)

    cdef vector[string] feature_names(self):
        return self._feature_names
