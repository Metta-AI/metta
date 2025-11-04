import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

from metta.sim.utils import get_or_create_policy_ids


def test_get_or_create_policy_ids_uses_epoch_in_policy_name():
    stats_client = MagicMock()
    stats_client.get_policy_ids.return_value = SimpleNamespace(policy_ids={})
    expected_id = uuid.uuid4()
    stats_client.create_policy.return_value = SimpleNamespace(id=expected_id)

    uri = "s3://bucket/example_run/checkpoints/example_run:v5.mpt"
    mapping = get_or_create_policy_ids(stats_client, [(uri, None)])

    stats_client.create_policy.assert_called_once()
    called_kwargs = stats_client.create_policy.call_args.kwargs
    assert called_kwargs["name"] == "example_run:v5"
    assert mapping[uri] == expected_id
