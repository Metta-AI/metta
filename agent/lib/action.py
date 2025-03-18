import agent.lib.nn_layer_library as nn_layer_library

class ActionType(nn_layer_library.Linear):
    def __init__(self, action_type_size, **cfg):
        self._out_tensor_shape = [action_type_size]
        self._nn_params = {'out_features': action_type_size}
        super().__init__(**cfg)

class ActionParam(nn_layer_library.Linear):
    def __init__(self, action_param_size, **cfg):
        self._out_tensor_shape = [action_param_size]
        self._nn_params = {'out_features': action_param_size}
        super().__init__(**cfg)
        
