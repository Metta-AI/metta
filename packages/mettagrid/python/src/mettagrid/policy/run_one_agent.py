import socket
import sys

from cogames.cli.utils import suppress_noisy_logs
from mettagrid.config.mettagrid_config import ProtocolConfig
from mettagrid.policy.loader import resolve_policy_class_path
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation, ObservationToken
from mettagrid.util.module import load_symbol

suppress_noisy_logs()


def main():
    """
    This script runs a single agent in its own process.

    The game coordinator communicates with us using a line-based protocol over stdin/stdout.
    The first few lines are for setup, telling us which agent to load and how to configure it.
    - First line: agent ID (hex)
    - Second line: JSON-encoded PolicySpec
    - Third line: JSON-encoded game configuration (PolicyEnvInterface)
    - Final line: "READY"

    We then respond with a "READY" line once our initialization is complete.

    Following that, we receive one observation per line and must respond with our chosen action.
    The observation lines contain hex-encoded bytes:
    - 1 byte for the agent ID, followed by a space
    - 4 bytes for the step number, followed by a space
    - 3 bytes per observation token, followed by a space

    The three bytes in the (raw) observation tokens correspond to:
    - location, where bits 0xF0 hold row, bits 0x0F hold column, and 0xFF means to ignore the observation
    - feature_id, as a reference into the game configuration struct we received at startup
    - value

    We respond with a single line containing our chosen action as a string.
    """

    try:
        sock_fd = int(sys.argv[1])
        sock = socket.socket(fileno=sock_fd)
        file = sock.makefile(mode="rw")
    except (
        IndexError,
        OSError,
    ):
        print(
            "The first command-line argument must be an FD number "
            "corresponding to a socket for communicating with "
            "the parent process",
            file=sys.stderr,
        )
        sys.exit(1)

    setup_lines: list[str] = []

    while True:
        line = file.readline()
        if line == "":  # EOF
            return
        line = line.strip()
        if line == "READY":
            break
        setup_lines.append(line)

    if len(setup_lines) < 3:
        raise RuntimeError("Insufficient setup lines received")

    agent_id = int(setup_lines[0], 16)
    policy_spec = PolicySpec.model_validate_json(setup_lines[1])

    for path in reversed(policy_spec.python_path):
        if path not in sys.path:
            sys.path.insert(0, path)

    policy_env_info = PolicyEnvInterface.model_validate_json(setup_lines[2])

    policy_class = load_symbol(resolve_policy_class_path(policy_spec.class_path))
    policy = policy_class(policy_env_info, **(policy_spec.init_kwargs or {}))  # type: ignore[call-arg]

    if policy_spec.data_path:
        policy.load_policy_data(policy_spec.data_path)

    agent = policy.agent_policy(agent_id=agent_id)

    assembler_protocols: list[ProtocolConfig] = []
    for protocol in policy_env_info.assembler_protocols:
        p2 = ProtocolConfig.model_validate(protocol)
        assembler_protocols.append(p2)
    policy_env_info.assembler_protocols = assembler_protocols

    agent.reset()

    try:
        file.write("READY\n")
        file.flush()
    except BrokenPipeError:
        pass

    while True:
        line = file.readline()
        if line == "":  # EOF
            return

        observation_line = line.strip()
        parts = observation_line.split(" ")
        raw_tokens = parts[2:]  # Skip agent ID and step number
        tokens: list[ObservationToken] = []
        for raw in raw_tokens:
            a = int(raw, 16)
            location_byte = (a >> 16) & 0xFF
            feature_id = (a >> 8) & 0xFF
            value = a & 0xFF
            # See mettagrid side, packages/mettagrid/cpp/include/mettagrid/systems/packed_coordinate.hpp
            row = (location_byte >> 4) & 0x0F
            col = location_byte & 0x0F
            location = (row, col)
            tokens.append(
                ObservationToken(
                    feature=policy_env_info.obs_features[feature_id],
                    location=location,
                    value=value,
                    raw_token=(location_byte, feature_id, value),
                )
            )
        obs = AgentObservation(agent_id=agent_id, tokens=tokens)

        action = agent.step(obs=obs)

        try:
            name, _, _ = str(action.name).partition("\n")
            file.write("{0}\n".format(name))
            file.flush()
        except BrokenPipeError:
            return


if __name__ == "__main__":
    main()
