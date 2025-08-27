#!/usr/bin/env python3
"""
Simple HTTP server with CORS headers for Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy.
These headers are often needed for SharedArrayBuffer support and other advanced web features.
"""

import http.server
import socketserver
import sys


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS headers."""

    def end_headers(self):
        """Add custom headers before ending headers."""
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


def run_server(port):
    """Run the HTTP server with CORS headers."""
    handler = CORSHTTPRequestHandler

    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Server running at http://localhost:{port}/")
            print("With headers:")
            print("  Cross-Origin-Opener-Policy: same-origin")
            print("  Cross-Origin-Embedder-Policy: require-corp")
            print("Press Ctrl+C to stop the server...")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"Error: Port {port} is already in use. Try a different port.")
        else:
            print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if a port was provided as command line argument
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid port number '{sys.argv[1]}'")
            print(f"Usage: {sys.argv[0]} [port]")
            sys.exit(1)
    else:
        port = 8000

    run_server(port)
