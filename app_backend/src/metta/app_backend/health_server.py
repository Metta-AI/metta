import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)

HEALTH_TIMEOUT_SECONDS = 120

_last_heartbeat: float = 0.0
_heartbeat_lock = threading.Lock()


def update_heartbeat():
    global _last_heartbeat
    with _heartbeat_lock:
        _last_heartbeat = time.time()


def _is_healthy() -> bool:
    with _heartbeat_lock:
        return (time.time() - _last_heartbeat) < HEALTH_TIMEOUT_SECONDS


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            status, body = (200, b"ok") if _is_healthy() else (503, b"unhealthy: loop stale")
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def start_health_server(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info(f"Health server started on port {port}")
    update_heartbeat()
