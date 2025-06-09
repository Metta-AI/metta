from typing import Any, Generator


def unroll_nested_dict(d: Any) -> Generator[tuple[str, Any], None, None]:
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v
