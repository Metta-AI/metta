import logging
import warnings

from pydantic.warnings import UnsupportedFieldAttributeWarning


def suppress_noisy_logs() -> None:
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning, module="pydantic")
    # Suppress deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame.pkgdata")

    # Silence PyTorch distributed elastic warning about redirects on MacOS/Windows
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r".*Redirects are currently not supported in Windows or MacOs.*",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
