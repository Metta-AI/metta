from __future__ import annotations

import metta.tools as tools


def resolve_uri(uri: str) -> tools.ResolveUriTool:
    return tools.ResolveUriTool(uri=uri)
