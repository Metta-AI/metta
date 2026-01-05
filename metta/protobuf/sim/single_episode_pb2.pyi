from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PureSingleEpisodeRunConfig(_message.Message):
    __slots__ = ("job", "device", "allow_network")
    JOB_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    ALLOW_NETWORK_FIELD_NUMBER: _ClassVar[int]
    job: PureSingleEpisodeJob
    device: str
    allow_network: bool
    def __init__(self, job: _Optional[_Union[PureSingleEpisodeJob, _Mapping]] = ..., device: _Optional[str] = ..., allow_network: bool = ...) -> None: ...

class PureSingleEpisodeJob(_message.Message):
    __slots__ = ("policy_uris", "assignments", "env_raw", "results_uri", "replay_uri", "seed", "max_action_millis")
    POLICY_URIS_FIELD_NUMBER: _ClassVar[int]
    ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
    ENV_RAW_FIELD_NUMBER: _ClassVar[int]
    RESULTS_URI_FIELD_NUMBER: _ClassVar[int]
    REPLAY_URI_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    MAX_ACTION_MILLIS_FIELD_NUMBER: _ClassVar[int]
    policy_uris: _containers.RepeatedScalarFieldContainer[str]
    assignments: _containers.RepeatedScalarFieldContainer[int]
    env_raw: _struct_pb2.Struct
    results_uri: str
    replay_uri: str
    seed: bytes
    max_action_millis: int
    def __init__(self, policy_uris: _Optional[_Iterable[str]] = ..., assignments: _Optional[_Iterable[int]] = ..., env_raw: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., results_uri: _Optional[str] = ..., replay_uri: _Optional[str] = ..., seed: _Optional[bytes] = ..., max_action_millis: _Optional[int] = ...) -> None: ...
