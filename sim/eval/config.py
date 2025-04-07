from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import hydra

@dataclass
class DashboardConfig:
    """Configuration settings for the RL policy evaluation dashboard."""   
    # Server configuration
    debug: bool = True
    port: int = 8050
    output_html_path: Optional[str] = None  # Path to save HTML output (None for no saving)
    
    # Dashboard settings
    default_graph_height: int = 500
    default_graph_width: int = 800
    page_title: str = "Metta Policy Dashboard"
    
    # Policy display names
    policy_names: Dict[str, str] = field(default_factory=lambda: {
        "b.daphne.navigation_varied_obstacle_shapes_pretrained.r.1": "varied_obstacles_pretrained",
        "b.daphne.navigation_varied_obstacle_shapes.r.0": "varied_obstacles",
        "navigation_poisson_sparser.r.2": "3_objects_far",
        "navigation_infinite_cooldown_sparser_pretrained.r.0": "inf_cooldown_sparse_pretrained",
        "navigation_infinite_cooldown_sparser.r.0": "inf_cooldown_sparse",
        "navigation_infinite_cooldown_sweep:v46": "inf_cooldown:v46",
        "navigation_poisson_sparser_pretrained.r.6": "3_objects_far_pretrained",
        "navigation_infinite_cooldown_sweep": "inf_cooldown",
        "navigation_infinite_cooldown_sweep.r.0": "inf_cooldown2",
        "b.daveey.t.8.rdr9.3": "daveey.t.8.rdr9.3",
        "b.daveey.t.4.rdr9.3": "daveey.t.4.rdr9.3",
        "b.daveey.t.8.rdr9.mb2.1": "daveey.t.8.rdr9.mb2.1",
        "b.daveey.t.1.pi.dpm": "daveey.t.1.pi.dpm",
        "b.daveey.t.64.dr90.1": "daveey.t.64.dr90.1",
        "b.daveey.t.8.rdr9.sb": "daveey.t.8.rdr9.sb",
    })
    
    # Visualization settings
    pass_threshold: float = 2.95
    metrics_of_interest: List[str] = field(default_factory=lambda: ['episode_reward'])
    matrix_colorscale: str = 'RdYlGn'
    matrix_score_range: Tuple[float, float] = (0, 3)

    @classmethod
    def from_dict_config(cls, config_dict: DictConfig) -> 'DashboardConfig':
        """
        Create a DashboardConfig from a DictConfig.

        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            New DashboardConfig instance with values from config_dict
        """
        # Add _target_ field for instantiate
        config_dict = OmegaConf.create(config_dict)
        OmegaConf.set_struct(config_dict, False)
        config_dict.setdefault('_target_', f"{cls.__module__}.{cls.__name__}")
        return hydra.utils.instantiate(config_dict)