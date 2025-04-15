from rich.table import Table

from metta.rl.pufferlib.trainer import PufferTrainer

from .dashboard import ROUND_OPEN, DashboardComponent, abbreviate, b2, c1, c2, duration, fmt_perf
from .user_stats import UserStats

class Training(DashboardComponent):
    def __init__(self, trainer: PufferTrainer):
        self.trainer = trainer

    def render(self):
        trainer = self.trainer
        profile = trainer.profile

        # Create trainer stats table
        stats_table = Table(box=ROUND_OPEN, expand=True)
        stats_table.add_column(f"{c1}Trainer", justify="left", vertical="top")
        stats_table.add_column(f"{c1}Value", justify="right", vertical="top")
        stats_table.add_row(f"{c2}Agent Steps", abbreviate(trainer.global_step))
        stats_table.add_row(f"{c2}SPS", abbreviate(profile.SPS))
        stats_table.add_row(f"{c2}Epoch", abbreviate(trainer.epoch))
        stats_table.add_row(f"{c2}Uptime", duration(profile.uptime))
        stats_table.add_row(f"{c2}Remaining", duration(profile.remaining))

        # Create performance table
        perf_table = Table(box=ROUND_OPEN, expand=True)
        perf_table.add_column(f"{c1}Performance", justify="left")
        perf_table.add_column(f"{c1}Time", justify="right")
        perf_table.add_column(f"{c1}%", justify="right")
        perf_table.add_row(*fmt_perf("Evaluate", profile.eval_time, profile.uptime))
        perf_table.add_row(*fmt_perf(" Forward", profile.eval_forward_time, profile.uptime))
        perf_table.add_row(*fmt_perf(" Env", profile.env_time, profile.uptime))
        perf_table.add_row(*fmt_perf(" Misc", profile.eval_misc_time, profile.uptime))
        perf_table.add_row(*fmt_perf("Train", profile.train_time, profile.uptime))
        perf_table.add_row(*fmt_perf(" Forward", profile.train_forward_time, profile.uptime))
        perf_table.add_row(*fmt_perf(" Learn", profile.learn_time, profile.uptime))
        perf_table.add_row(*fmt_perf(" Misc", profile.train_misc_time, profile.uptime))

        # Create losses table
        losses_table = Table(box=ROUND_OPEN, expand=True)
        losses_table.add_column(
            f"{c1}Losses",
            justify="left",
        )
        losses_table.add_column(f"{c1}Value", justify="right")
        for metric, value in self.trainer.losses.items():
            losses_table.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")

        # Create user stats component
        user_stats = UserStats(self.trainer.recent_stats, max_stats=8)

        # Create the combined table
        combined_table = Table(box=None, expand=True)
        combined_table.add_row(stats_table, perf_table, losses_table, user_stats.render())

        return combined_table
