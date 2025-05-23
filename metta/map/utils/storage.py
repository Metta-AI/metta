


def parse_file_uri(uri: str) -> str:
    if uri.startswith("file://"):
        return uri.split("file://")[1]

    # we don't support any other schemes
    if "://" in uri:
        raise ValueError(f"Invalid URI: {uri}")

    # probably a local file name
    return uri


