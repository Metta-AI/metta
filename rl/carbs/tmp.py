    if 'total_timesteps' in sweep_parameters['train']['parameters']:
        time_param = sweep_parameters['train']['parameters']['total_timesteps']
        min_timesteps = time_param['min']
        param_spaces.append(carbs_param('train', 'total_timesteps', 'log', sweep_parameters,
            search_center=min_timesteps, is_integer=True))

    batch_param = sweep_parameters['train']['parameters']['batch_size']
    default_batch = (batch_param['max'] - batch_param['min']) // 2

    minibatch_param = sweep_parameters['train']['parameters']['minibatch_size']
    default_minibatch = (minibatch_param['max'] - minibatch_param['min']) // 2

    param_spaces += [
        carbs_param('train', 'learning_rate', 'log', sweep_parameters, search_center=1e-3),
        carbs_param('train', 'gamma', 'logit', sweep_parameters, search_center=0.95),
        carbs_param('train', 'gae_lambda', 'logit', sweep_parameters, search_center=0.90),
        carbs_param('train', 'update_epochs', 'linear', sweep_parameters,
            search_center=1, scale=3, is_integer=True),
        carbs_param('train', 'clip_coef', 'logit', sweep_parameters, search_center=0.1),
        carbs_param('train', 'vf_coef', 'logit', sweep_parameters, search_center=0.5),
        carbs_param('train', 'vf_clip_coef', 'logit', sweep_parameters, search_center=0.1),
        carbs_param('train', 'max_grad_norm', 'linear', sweep_parameters, search_center=0.5),
        carbs_param('train', 'ent_coef', 'log', sweep_parameters, search_center=0.01),
        carbs_param('train', 'batch_size', 'log', sweep_parameters,
            search_center=default_batch, is_integer=True),
        carbs_param('train', 'minibatch_size', 'log', sweep_parameters,
            search_center=default_minibatch, is_integer=True),
        carbs_param('train', 'bptt_horizon', 'log', sweep_parameters,
            search_center=16, is_integer=True),
    ]
