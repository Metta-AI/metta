import contextlib
import json
import logging
import os
import pathlib
import queue
import subprocess
import threading
import time
from typing import IO

logger = logging.getLogger(__name__)


class LSPClient:
    """
    This class implements a simple LSP client.

    There are some python libraries for LSP, but they either don't support pyright, or aren't very mature (lack
    documentation, etc.).
    """

    def __init__(self):
        root = pathlib.Path(".").resolve()

        self.id = 0
        self.unprocessed_responses: dict[int, dict] = {}

        self.server = subprocess.Popen(
            ["pyright-langserver", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # avoid blocking if server writes a lot to stderr
            text=False,
            bufsize=0,
        )

        def read_messages(reader: IO[bytes], out_q: queue.Queue[dict]):
            # background reader: pushes decoded JSON-RPC messages into a queue
            while True:
                # read headers
                headers: dict[str, str] = {}
                line = reader.readline()
                if not line:
                    break
                while line not in (b"\r\n", b""):
                    key, val = line.decode("ascii").split(":", 1)
                    headers[key.strip().lower()] = val.strip()
                    line = reader.readline()
                if not headers:
                    continue
                length = int(headers["content-length"])
                body = reader.read(length)
                if not body:
                    break

                msg = json.loads(body.decode("utf-8"))
                if not isinstance(msg, dict):
                    logger.warning(f"Invalid message: {msg}")

                out_q.put(msg)

        # Start background reader
        self.queue = queue.Queue[dict]()
        self.queue_thread = threading.Thread(target=read_messages, args=(self.server.stdout, self.queue), daemon=True)
        self.queue_thread.start()

        # Initialize
        init_id = self.next_id()
        self.send(
            {
                "jsonrpc": "2.0",
                "id": init_id,
                "method": "initialize",
                "params": {
                    "processId": os.getpid(),
                    "rootUri": root.as_uri(),
                    "capabilities": {},  # minimal is fine; expand if you need features
                    "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                    "clientInfo": {"name": "gridworks-pyright-client", "version": "0.1"},
                },
            },
        )

        # The server may send several notifications first; keep reading until we see id=1
        self.recv_id(init_id)

        self.send({"jsonrpc": "2.0", "method": "initialized", "params": {}})

    def shutdown(self):
        shutdown_id = self.next_id()
        self.send({"jsonrpc": "2.0", "id": shutdown_id, "method": "shutdown", "params": None})
        self.recv_id(shutdown_id)

        self.send({"jsonrpc": "2.0", "method": "exit"})

    def next_id(self) -> int:
        self.id += 1
        return self.id

    def send(self, msg: dict):
        data = json.dumps(msg).encode("utf-8")
        assert self.server.stdin
        self.server.stdin.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data)
        self.server.stdin.flush()

    def send_with_id(self, msg: dict):
        msg["id"] = self.next_id()
        self.send(msg)
        return msg["id"]

    def recv_ids(self, wanted_ids: list[int], timeout=30.0) -> dict[int, dict]:
        """Drain queue until we see all the wanted ids."""
        deadline = time.time() + timeout
        responses: dict[int, dict] = {}

        remaining_ids = set(wanted_ids)
        for wanted_id in wanted_ids:
            if wanted_id in self.unprocessed_responses:
                responses[wanted_id] = self.unprocessed_responses[wanted_id]
                del self.unprocessed_responses[wanted_id]
                remaining_ids.remove(wanted_id)

        while remaining_ids:
            remaining = max(0, deadline - time.time())
            if remaining == 0:
                raise TimeoutError(f"Timed out waiting for responses: {remaining_ids}")
            try:
                msg = self.queue.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"Timed out waiting for responses: {remaining_ids}") from None

            msg_id = msg.get("id")

            if msg_id is None:
                # some responses are informational, we don't care about them
                continue

            if msg_id in remaining_ids:
                responses[msg_id] = msg
                remaining_ids.remove(msg_id)
            else:
                # collect responses that we didn't expect, maybe we'll need them later
                # (i.e. if this class is used in multi-threaded code, I'm not sure if we'll need it)
                self.unprocessed_responses[msg_id] = msg

        return responses

    def recv_id(self, wanted_id: int, timeout=10.0) -> dict:
        return self.recv_ids([wanted_id], timeout)[wanted_id]

    @contextlib.contextmanager
    def with_file(self, file_path: pathlib.Path):
        uri = file_path.resolve().as_uri()
        text = file_path.read_text(encoding="utf-8")
        self.send(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": uri,
                        "languageId": "python",
                        "version": 1,
                        "text": text,
                    }
                },
            },
        )
        yield uri
        self.send(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didClose",
                "params": {"textDocument": {"uri": uri}},
            },
        )

    def get_file_symbols(self, file_path: pathlib.Path):
        with self.with_file(file_path) as uri:
            req_id = self.next_id()
            self.send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": "textDocument/documentSymbol",
                    "params": {"textDocument": {"uri": uri}},
                },
            )
            resp = self.recv_id(req_id)
            return resp.get("result")

    def get_hover(self, file_path: pathlib.Path, line: int, column: int):
        with self.with_file(file_path) as uri:
            req_id = self.next_id()
            self.send(
                {
                    "jsonrpc": "2.0",
                    "method": "textDocument/hover",
                    "id": req_id,
                    "params": {"textDocument": {"uri": uri}, "position": {"line": line, "character": column}},
                },
            )
            resp = self.recv_id(req_id)
            return resp.get("result")

    def get_hover_bulk(self, file_path: pathlib.Path, positions: list[tuple[int, int]]):
        with self.with_file(file_path) as uri:
            req_ids: list[int] = []
            for line, column in positions:
                req_id = self.next_id()
                self.send(
                    {
                        "jsonrpc": "2.0",
                        "method": "textDocument/hover",
                        "id": req_id,
                        "params": {"textDocument": {"uri": uri}, "position": {"line": line, "character": column}},
                    },
                )
                req_ids.append(req_id)
            resp = self.recv_ids(req_ids)
            return [resp[req_id].get("result") for req_id in req_ids]
