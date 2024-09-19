
    def mean_and_log(self):
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
            except:
                del self.stats[k]

            self.stats[k] = v

        if self.wandb is None:
            return

        self.last_log_time = time.time()
        self.wandb.log({
            '0verview/SPS': self.profile.SPS,
            '0verview/agent_steps': self.global_step,
            '0verview/epoch': self.epoch,
            '0verview/learning_rate': self.optimizer.param_groups[0]["lr"],
            **{f'environment/{k}': v for k, v in self.stats.items()},
            **{f'losses/{k}': v for k, v in self.losses.items()},
            **{f'performance/{k}': v for k, v in self.profile},
        })
