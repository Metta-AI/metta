from mettagrid.map.utils.s3utils import get_s3_client, is_s3_uri, parse_s3_uri


def parse_file_uri(uri: str) -> str:
    if uri.startswith("file://"):
        return uri.split("file://")[1]

    # we don't support any other schemes
    if "://" in uri:
        raise ValueError(f"Invalid URI: {uri}")

    # probably a local file name
    return uri


def save_to_uri(text: str, uri: str):
    if is_s3_uri(uri):
        bucket, key = parse_s3_uri(uri)
        s3 = get_s3_client()
        s3.put_object(Bucket=bucket, Key=key, Body=text)

    filename = parse_file_uri(uri)

    with open(filename, "w") as f:
        f.write(text)


def load_from_uri(uri: str) -> str:
    if is_s3_uri(uri):
        bucket, key = parse_s3_uri(uri)
        s3 = get_s3_client()
        response = s3.get_object(Bucket=bucket, Key=key)
        return response["Body"].read().decode("utf-8")

    filename = parse_file_uri(uri)

    with open(filename, "r") as f:
        return f.read()
