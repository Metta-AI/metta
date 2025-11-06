import abc
import typing

T = typing.TypeVar("T")


class PackageInstaller(abc.ABC, typing.Generic[T]):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this package manager is available on the system."""
        pass

    @abc.abstractmethod
    def install(self, packages: list[T]) -> None:
        pass

    @abc.abstractmethod
    def check_installed(self, packages: list[T]) -> bool:
        pass
