import metta.agent.lib.nn_layer_library as nn_layer_library

class ActionType(nn_layer_library.Linear):
    def __init__(self, action_type_size, **cfg):
        super().__init__(**cfg)
        self._output_size = action_type_size


class ActionParam(nn_layer_library.Linear):
    def __init__(self, action_param_size, **cfg):
        super().__init__(**cfg)
        self._output_size = action_param_size
