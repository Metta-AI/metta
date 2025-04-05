import copy
import hydra
from omegaconf import DictConfig
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

@dataclass
class ReportConfig:
    """Configuration settings for the RL policy evaluation dashboard."""
    
    # Data sources
    eval_db_uri: str = 'sqlite:///path/to/your/eval_stats.db'
    run_dir: str = './run_dir'
    
    # Server configuration
    debug: bool = True
    port: int = 8050
    
    # Dashboard settings
    default_graph_height: int = 500
    default_graph_width: int = 800
    
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
    
    def to_dict(self) -> Dict:
        """Convert the config to a dictionary for compatibility with existing code."""
        return {
            'eval_db_uri': self.eval_db_uri,
            'run_dir': self.run_dir,
            'debug': self.debug,
            'port': self.port,
            'default_graph_height': self.default_graph_height,
            'default_graph_width': self.default_graph_width,
            'policy_names': self.policy_names,
            'pass_threshold': self.pass_threshold,
            'metrics_of_interest': self.metrics_of_interest,
            'matrix_colorscale': self.matrix_colorscale,
            'matrix_score_range': self.matrix_score_range,
        }

    @classmethod
    def from_dict_config(cls, config_dict: DictConfig) -> 'ReportConfig':
        """
        Create an AppConfig from a dictionary or DictConfig.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            New AppConfig instance with values from config_dict
        """
        # Add _target_ field for instantiate
        config = copy.deepcopy(config_dict)
        if '_target_' not in config or config['_target_'] is None:
            config['_target_'] = f"{cls.__module__}.{cls.__name__}"
        return hydra.utils.instantiate(config)
    