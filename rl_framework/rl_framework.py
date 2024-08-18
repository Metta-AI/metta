

from ast import List
from typing import NamedTuple

class EvaluationResult(NamedTuple):
    reward: float
    frames: List
class RLFramework():
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self):
        raise NotImplementedError

    def evaluate(self) -> EvaluationResult:
        raise NotImplementedError
