"""RPC client for communicating with MettaGrid socket server.

This module provides a Python client that speaks the length-prefixed protobuf protocol
used by the MettaGrid RPC server.
"""

import os
import socket
import struct
import sys
from typing import Optional

# Add python directory to path for protobuf imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

from mettagrid.rpc.v1 import mettagrid_service_pb2 as pb


class RPCClient:
    """Client for communicating with MettaGrid RPC server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5858):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self._request_counter = 0

    def connect(self):
        """Establish connection to the RPC server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print(f"Connected to RPC server at {self.host}:{self.port}")

    def disconnect(self):
        """Close connection to the RPC server."""
        if self.socket:
            self.socket.close()
            self.socket = None
            print("Disconnected from RPC server")

    def _send_message(self, message: pb.MettaGridRequest) -> pb.MettaGridResponse:
        """Send a protobuf message with length prefix and receive response."""
        if not self.socket:
            raise RuntimeError("Not connected to server")

        # Serialize message
        data = message.SerializeToString()

        # Send length prefix (4 bytes, big-endian)
        length = len(data)
        self.socket.sendall(struct.pack(">I", length))

        # Send message data
        self.socket.sendall(data)

        # Receive response length prefix
        length_bytes = self._recv_exactly(4)
        response_length = struct.unpack(">I", length_bytes)[0]

        # Receive response data
        response_data = self._recv_exactly(response_length)

        # Parse response
        response = pb.MettaGridResponse()
        response.ParseFromString(response_data)

        # Check for errors
        if response.HasField("error"):
            raise RuntimeError(f"RPC Error: {response.error.message}")

        return response

    def _recv_exactly(self, n: int) -> bytes:
        """Receive exactly n bytes from the socket."""
        data = b""
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                raise RuntimeError("Connection closed by server")
            data += chunk
        return data

    def _next_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_counter += 1
        return f"req_{self._request_counter}"

    def create_game(
        self,
        game_id: str,
        config: pb.GameConfig,
        map_def: pb.MapDefinition,
        seed: int = 42,
    ) -> None:
        """Create a new game instance."""
        request = pb.MettaGridRequest(
            request_id=self._next_request_id(),
            create_game=pb.CreateGameRequest(
                game_id=game_id,
                config=config,
                map=map_def,
                seed=seed,
            ),
        )

        response = self._send_message(request)

        if not response.create_result.ok:
            raise RuntimeError(
                f"Failed to create game: {response.create_result.message}"
            )

        print(f"Game '{game_id}' created successfully")

    def step_game(self, game_id: str, flat_actions: list[int]) -> pb.StepResult:
        """Step the game with the given actions and return observations."""
        request = pb.MettaGridRequest(
            request_id=self._next_request_id(),
            step_game=pb.StepGameRequest(
                game_id=game_id,
                flat_actions=flat_actions,
            ),
        )

        response = self._send_message(request)
        return response.step_result

    def get_state(self, game_id: str) -> pb.StepResult:
        """Get current game state without stepping."""
        request = pb.MettaGridRequest(
            request_id=self._next_request_id(),
            get_state=pb.GetStateRequest(game_id=game_id),
        )

        response = self._send_message(request)
        return response.state_result.snapshot

    def delete_game(self, game_id: str) -> None:
        """Delete a game instance."""
        request = pb.MettaGridRequest(
            request_id=self._next_request_id(),
            delete_game=pb.DeleteGameRequest(game_id=game_id),
        )

        response = self._send_message(request)

        if not response.delete_result.ok:
            raise RuntimeError(
                f"Failed to delete game: {response.delete_result.message}"
            )

        print(f"Game '{game_id}' deleted successfully")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
