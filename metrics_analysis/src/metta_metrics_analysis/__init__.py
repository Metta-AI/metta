"""
Metta Metrics Analysis Package.

A comprehensive toolkit for analyzing and comparing machine learning runs from WandB.
"""

__version__ = "0.1.0"


# Delay imports to avoid circular dependencies
def __getattr__(name):
    if name == "DataProcessor":
        from .data_processor import DataProcessor

        return DataProcessor
    elif name == "StatisticalAnalyzer":
        from .statistical_analysis import StatisticalAnalyzer

        return StatisticalAnalyzer
    elif name == "WandBDataCollector":
        from .wandb_data_collector import WandBDataCollector

        return WandBDataCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "WandBDataCollector",
    "DataProcessor",
    "StatisticalAnalyzer",
]
