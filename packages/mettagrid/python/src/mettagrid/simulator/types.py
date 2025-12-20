from dataclasses import dataclass


# Resolves circular dependency simulator <-> renderer
@dataclass
class Action:
    name: str
