from pydantic import BaseModel


class BrewPackageConfig(BaseModel):
    """Configuration for a single package."""

    name: str
    version: str | None = None
    tap: str | None = None
    installed_name: str | None = None
    pin: bool = False

    @property
    def fully_specified_name(self) -> str:
        name = self.name
        if self.version:
            name = f"{name}@{self.version}"
        if self.tap:
            name = f"{self.tap}/{name}"
        return name

    def __repr__(self) -> str:
        return f"{self.fully_specified_name}"


class AptPackageConfig(BaseModel):
    """Configuration for a single package."""

    name: str
    version: str | None = None


class PackageSpec(BaseModel):
    """Package specification across different package managers."""

    brew: BrewPackageConfig | None = None
    apt: AptPackageConfig | None = None


class SystemDepsConfig(BaseModel):
    packages: dict[str, PackageSpec]
