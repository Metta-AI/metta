    if 'total_timesteps' in sweep_parameters['train']['parameters']:
        time_param = sweep_parameters['train']['parameters']['total_timesteps']
        min_timesteps = time_param['min']
        param_spaces.append(carbs_param('train', 'total_timesteps', 'log', sweep_parameters,
            search_center=min_timesteps, is_integer=True))

    batch_param = sweep_parameters['train']['parameters']['batch_size']
    default_batch = (batch_param['max'] - batch_param['min']) // 2

    minibatch_param = sweep_parameters['train']['parameters']['minibatch_size']
    default_minibatch = (minibatch_param['max'] - minibatch_param['min']) // 2
