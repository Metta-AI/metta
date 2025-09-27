from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class PackageInstaller(ABC, Generic[T]):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this package manager is available on the system."""
        pass

    @abstractmethod
    def install(self, packages: list[T]) -> None:
        pass

    @abstractmethod
    def check_installed(self, packages: list[T]) -> bool:
        pass
