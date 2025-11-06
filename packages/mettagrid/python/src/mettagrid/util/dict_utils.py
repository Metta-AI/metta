import typing


def unroll_nested_dict(d: dict[str, typing.Any]) -> typing.Generator[tuple[str, typing.Any], None, None]:
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v
