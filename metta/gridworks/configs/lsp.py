import ast
import contextlib
import json
import logging
import os
import pathlib
import queue
import subprocess
import threading
import time
from typing import IO, Any

logger = logging.getLogger(__name__)


class _StaticLSPClient:
    """Fallback client that infers hover information via the AST."""

    def shutdown(self) -> None:
        return None

    def _function_defs(self, file_path: pathlib.Path) -> dict[int, ast.AST]:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        defs: dict[int, ast.AST] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs[node.lineno - 1] = node
        return defs

    def _hover_from_node(self, node: ast.AST | None) -> dict[str, Any]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            return_annotation = ast.unparse(node.returns)
            hover_value = f"() -> {return_annotation}"
        else:
            hover_value = "() -> Unknown"
        return {"contents": {"value": hover_value}}

    def get_hover_bulk(
        self,
        file_path: pathlib.Path,
        positions: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        defs = self._function_defs(file_path)
        return [self._hover_from_node(defs.get(line)) for line, _ in positions]

    def get_hover(self, file_path: pathlib.Path, line: int, column: int) -> dict[str, Any]:
        return self.get_hover_bulk(file_path, [(line, column)])[0]

    def get_file_symbols(self, file_path: pathlib.Path) -> None:
        logger.debug("Static LSP fallback does not provide file symbols for %s", file_path)
        return None


class LSPClient:
    """
    This class implements a simple LSP client.

    There are some python libraries for LSP, but they either don't support pyright, or aren't very mature (lack
    documentation, etc.).
    """

    def __init__(self) -> None:
        self.id = 0
        self.unprocessed_responses: dict[int, dict[str, Any]] = {}
        self.server: subprocess.Popen[bytes] | None = None
        self.queue: queue.Queue[dict[str, Any]] | None = None
        self.queue_thread: threading.Thread | None = None
        self._fallback_client: _StaticLSPClient | None = None

        try:
            self._start_pyright_server()
        except (FileNotFoundError, TimeoutError, OSError) as exc:
            logger.warning(
                "Unable to start pyright-langserver; falling back to static analysis. Error: %s",
                exc,
            )
            self._teardown_server()
            self._fallback_client = _StaticLSPClient()

    def _start_pyright_server(self) -> None:
        root = pathlib.Path(".").resolve()

        self.server = subprocess.Popen(
            ["pyright-langserver", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=False,
            bufsize=0,
        )

        if not self.server.stdout or not self.server.stdin:
            raise RuntimeError("Pyright language server streams are not available")

        def read_messages(reader: IO[bytes], out_q: queue.Queue[dict[str, Any]]) -> None:
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
                    logger.warning("Invalid message: %s", msg)

                out_q.put(msg)

        # Start background reader
        self.queue = queue.Queue[dict[str, Any]]()
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
                    "capabilities": {},
                    "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                    "clientInfo": {"name": "gridworks-pyright-client", "version": "0.1"},
                },
            },
        )

        # The server may send several notifications first; keep reading until we see id=1
        self.recv_id(init_id)

        self.send({"jsonrpc": "2.0", "method": "initialized", "params": {}})

    def _teardown_server(self) -> None:
        if self.server and self.server.poll() is None:
            with contextlib.suppress(Exception):
                self.server.terminate()
        self.server = None
        self.queue = None
        self.queue_thread = None

    def shutdown(self) -> None:
        if self._fallback_client is not None:
            self._fallback_client.shutdown()
            return

        if not self.server:
            return

        shutdown_id = self.next_id()
        self.send({"jsonrpc": "2.0", "id": shutdown_id, "method": "shutdown", "params": None})
        self.recv_id(shutdown_id)

        self.send({"jsonrpc": "2.0", "method": "exit"})
        self._teardown_server()

    def next_id(self) -> int:
        if self._fallback_client is not None:
            raise RuntimeError("next_id is not supported in static LSP fallback mode")
        self.id += 1
        return self.id

    def send(self, msg: dict[str, Any]) -> None:
        if self._fallback_client is not None:
            raise RuntimeError("send is not supported in static LSP fallback mode")
        if not self.server or not self.server.stdin:
            raise RuntimeError("Pyright language server is not running")
        data = json.dumps(msg).encode("utf-8")
        self.server.stdin.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data)
        self.server.stdin.flush()

    def send_with_id(self, msg: dict[str, Any]) -> int:
        if self._fallback_client is not None:
            raise RuntimeError("send_with_id is not supported in static LSP fallback mode")
        msg["id"] = self.next_id()
        self.send(msg)
        return msg["id"]

    def recv_ids(self, wanted_ids: list[int], timeout: float = 30.0) -> dict[int, dict[str, Any]]:
        if self._fallback_client is not None:
            raise RuntimeError("recv_ids is not supported in static LSP fallback mode")
        if not self.queue:
            raise RuntimeError("Pyright language server is not running")
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
            remaining = max(0.0, deadline - time.time())
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

    def recv_id(self, wanted_id: int, timeout: float = 10.0) -> dict[str, Any]:
        if self._fallback_client is not None:
            raise RuntimeError("recv_id is not supported in static LSP fallback mode")
        return self.recv_ids([wanted_id], timeout)[wanted_id]

    @contextlib.contextmanager
    def with_file(self, file_path: pathlib.Path):
        if self._fallback_client is not None:
            raise RuntimeError("with_file is not supported in static LSP fallback mode")
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

    def get_file_symbols(self, file_path: pathlib.Path) -> Any:
        if self._fallback_client is not None:
            return self._fallback_client.get_file_symbols(file_path)
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

    def get_hover(self, file_path: pathlib.Path, line: int, column: int) -> Any:
        if self._fallback_client is not None:
            return self._fallback_client.get_hover(file_path, line, column)
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

    def get_hover_bulk(self, file_path: pathlib.Path, positions: list[tuple[int, int]]) -> list[Any]:
        if self._fallback_client is not None:
            return self._fallback_client.get_hover_bulk(file_path, positions)
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
