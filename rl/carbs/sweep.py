from omegaconf import DictConfig, OmegaConf

def apply_carbs_suggestion(cfg: OmegaConf, suggestion: DictConfig):
    for key, value in suggestion.items():
        if key == "suggestion_uuid":
            continue
        new_cfg_param = cfg
        key_parts = key.split(".")
        for k in key_parts[:-1]:
            new_cfg_param = new_cfg_param[k]
        param_name = key_parts[-1]
        new_cfg_param[param_name] = value

