#!/usr/bin/env python3
"""
Dummy API server that mimics Observatory and Cogweb APIs for local testing.
Runs on port 8080 (Observatory) and 8081 (Cogweb).
"""

import http.server
import json
import socketserver
import threading
import time


class DummyAPIHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Override to reduce noise in logs."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        path = self.path
        print(f"🔍 GET {path}")

        if "/sweeps/" in path:
            # Observatory sweep endpoint
            sweep_name = path.split("/sweeps/")[-1].split("/")[0]
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {
                "exists": True,
                "wandb_sweep_id": f"dummy_sweep_{sweep_name}",
                "name": sweep_name,
                "status": "active",
            }
            self.wfile.write(json.dumps(response).encode())

        elif "/health" in path:
            # Health check endpoint
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {"status": "healthy", "service": "dummy-api"}
            self.wfile.write(json.dumps(response).encode())

        else:
            # Default response
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {"status": "success", "message": "Dummy API response"}
            self.wfile.write(json.dumps(response).encode())

    def do_POST(self):
        """Handle POST requests."""
        path = self.path
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        print(f"📝 POST {path}")

        try:
            json.loads(post_data.decode()) if post_data else {}
        except json.JSONDecodeError:
            pass

        if "/sweeps/" in path and "/create_sweep" in path:
            # Observatory create sweep endpoint
            sweep_name = path.split("/sweeps/")[-1].split("/")[0]
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {"status": "created", "sweep_name": sweep_name, "wandb_sweep_id": f"dummy_sweep_{sweep_name}"}
            self.wfile.write(json.dumps(response).encode())

        elif "/sweeps/" in path and "/runs/next" in path:
            # Cogweb next run endpoint
            sweep_name = path.split("/sweeps/")[-1].split("/")[0]
            run_name = f"{sweep_name}.r.{int(time.time()) % 1000}"
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {"run_name": run_name, "status": "assigned"}
            self.wfile.write(json.dumps(response).encode())

        else:
            # Default POST response
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {"status": "success", "message": "Dummy API POST response"}
            self.wfile.write(json.dumps(response).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server(port, name):
    """Run a server on the specified port."""
    try:
        with socketserver.TCPServer(("", port), DummyAPIHandler) as httpd:
            print(f"🤖 {name} running on port {port}")
            httpd.serve_forever()
    except Exception as e:
        print(f"❌ Failed to start {name} on port {port}: {e}")


def main():
    """Start both Observatory and Cogweb API servers."""
    print("🚀 Starting Dummy API Servers...")

    # Start Observatory API (port 8080)
    observatory_thread = threading.Thread(target=run_server, args=(8080, "Observatory API"), daemon=True)
    observatory_thread.start()

    # Start Cogweb API (port 8081)
    cogweb_thread = threading.Thread(target=run_server, args=(8081, "Cogweb API"), daemon=True)
    cogweb_thread.start()

    print("✅ Dummy API servers started!")
    print("   Observatory API: http://localhost:8080")
    print("   Cogweb API: http://localhost:8081")
    print("   Health check: http://localhost:8080/health")

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down dummy API servers...")


if __name__ == "__main__":
    main()
