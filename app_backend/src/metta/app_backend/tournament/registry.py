from typing import Callable

from metta.app_backend.tournament.commissioners.beta import BetaCommissioner
from metta.app_backend.tournament.interfaces import CommissionerInterface

SEASONS: dict[str, Callable[[], CommissionerInterface]] = {
    "beta": BetaCommissioner,
}
