
class EvalStatsLogger:
    def __init__(self, eval_stats):
        self.eval_stats = eval_stats

    def log(self):
        pass

class WanddbEvalStatsLogger(EvalStatsLogger):
    def __init__(self, eval_stats, artifact):
        super().__init__(eval_stats, artifact)

    def log(self):
        pass

class FileEvalStatsLogger(EvalStatsLogger):
    def __init__(self, eval_stats, path):
        super().__init__(eval_stats)
        self.path = path

    def log(self):
        pass
