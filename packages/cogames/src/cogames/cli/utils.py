import logging
import warnings

from pydantic.warnings import UnsupportedFieldAttributeWarning


def init_suppress_warnings() -> None:
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning, module="pydantic")
    # Suppress deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame.pkgdata")

    # Silence PyTorch distributed elastic warning about redirects on MacOS/Windows
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
