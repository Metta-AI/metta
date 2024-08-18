import unittest
from ast import arg
import tempfile
from train import make_env_func, register_custom_components, parse_custom_args
from sample_factory.train import run_rl
from sample_factory.cfg.arguments import default_cfg
import sys
from sample_factory.enjoy import enjoy

def train_and_eval(args):
    train_args = [
            "--algo=APPO",
            "--experiment=test_griddly_1",
            "--env=GDY-PowerGrid",
            "--num_workers=5",
            "--device=cpu",
            # "--num_workers=1",
            # "--serial=True",
        ] + args

    with tempfile.TemporaryDirectory() as train_dir:
        train_args.append("--train_dir=" + train_dir)

        train_cfg = parse_custom_args(argv=train_args)
        run_rl(train_cfg)
        eval_args = train_args + [
            "--max_num_episodes=10",
            "--no_render",
            "--restart_behavior=restart",
        ]
        eval_config = parse_custom_args(argv=eval_args, evaluation=True)
        status = enjoy(eval_config)
        return status[1]

class TestGridlyTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        register_custom_components()

    def test_2x2(self):
        args = [
            "--train_for_env_steps=300000",
            "--env_num_agents=2",
            "--power_grid_width=4",
            "--power_grid_height=4",
            "--power_grid_chargers_per_agent=1",
            "--power_grid_energy:initial=3",
            "--power_grid_wall_density=0.0",
            "--env_max_steps=5"
        ]
        self.assertGreater(train_and_eval(args), 4.99)

    def test_2x2_predict(self):
        args = [
            "--train_for_env_steps=300000",
            "--env_num_agents=2",
            "--power_grid_width=4",
            "--power_grid_height=4",
            "--power_grid_energy_per_agent=1",
            "--power_grid_wall_density=0.0",
            "--env_max_steps=2"
        ]
        self.assertGreater(train_and_eval(args), 0.99)

    def test_8x8(self):
        args = [
            "--train_for_env_steps=300000",
            "--env_num_agents=2",
            "--power_grid_width=9",
            "--power_grid_height=9",
            "--power_grid_chargers_per_agent=1",
            "--power_grid_energy:initial=16",
            "--power_grid_wall_density=0.0",
            "--env_max_steps=20"
        ]
        self.assertGreater(train_and_eval(args), 0.99)

    def test_8x8_w05(self):
        args = [
            "--train_for_env_steps=80000",
            "--env_num_agents=2",
            "--power_grid_width=9",
            "--power_grid_height=9",
            "--power_grid_energy_per_agent=1",
            "--power_grid_wall_density=0.05",
            "--env_max_steps=16"
        ]
        self.assertGreater(train_and_eval(args), 0.99)

if __name__ == "__main__":
    unittest.main()
